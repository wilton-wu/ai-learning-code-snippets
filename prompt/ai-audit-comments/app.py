import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_apscheduler import APScheduler
from openai import OpenAI

load_dotenv()
BASE_DIR = Path(__file__).parent

# 全局变量，用于跟踪任务执行状态
_task_running = False
_task_lock = threading.Lock()
_last_execution_time = None

# 创建Flask应用
app = Flask(__name__)


# 配置类
class Config:
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR}/comments.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # APScheduler配置
    SCHEDULER_API_ENABLED = False  # 禁用API
    SCHEDULER_EXECUTORS = {
        "default": {"type": "threadpool", "max_workers": 1}  # 限制线程数为1
    }
    SCHEDULER_JOB_DEFAULTS = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 10,  # 错过执行时间的宽限期
    }


def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def relative_time_from_timestamp(timestamp):
    target_time = datetime.fromtimestamp(timestamp)
    now = datetime.now()

    delta = now - target_time
    days = delta.days
    seconds = delta.seconds
    minutes = seconds // 60
    hours = minutes // 60

    if days > 365:
        return f"{days // 365} 年前"
    elif days > 30:
        return f"{days // 30} 个月前"
    elif days > 7:
        return f"{days // 7} 周前"
    elif days > 0:
        return f"{days} 天前"
    elif hours > 0:
        return f"{hours} 小时前"
    elif minutes > 0:
        return f"{minutes} 分钟前"
    elif seconds > 0:
        return f"{seconds} 秒前"
    else:
        return "刚刚"


def print_time(job_id):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} :Task {job_id} executed.")


# 应用配置
app.config.from_object(Config())
app.jinja_env.filters["timesince"] = relative_time_from_timestamp

# 初始化数据库
db = SQLAlchemy(app)

# 初始化调度器但不启动
scheduler = APScheduler()
scheduler.init_app(app)


# 数据模型定义
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    # 评论状态 0:未审核 1:已审核通过 2:已拒绝
    status = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.Integer, nullable=False, default=0)


# 自动审核任务
def auto_audit():
    # 使用线程锁确保任务不会重复执行
    global _task_running, _last_execution_time

    process_id = os.getpid()
    current_time = datetime.now()
    task_id = str(uuid.uuid4())[:8]  # 使用UUID生成唯一ID

    # 打印调试信息，但不执行实际任务
    print(
        f"[进程ID: {process_id}] 收到任务执行请求 [{task_id}]，时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # 检查是否有其他任务正在运行
    with _task_lock:
        if _task_running:
            print(f"[进程ID: {process_id}] 已有其他任务实例在运行，跳过本次执行")
            return

        # 检查上次执行时间
        if (
            _last_execution_time
            and (current_time - _last_execution_time).total_seconds() < 5
        ):
            print(
                f"[进程ID: {process_id}] 距离上次执行时间过短 ({(current_time - _last_execution_time).total_seconds():.2f}秒)，跳过本次执行"
            )
            return

        # 标记任务开始运行
        _task_running = True
        print(
            f"[进程ID: {process_id}] 开始执行自动审核任务 [{task_id}]，时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    try:
        # 在应用上下文中执行数据库操作
        with app.app_context():
            # 记录开始时间
            start_time = datetime.now()

            client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            prompt = """
# 评论内容审核指令

## 任务描述
对用户提交的评论内容进行合规性评估，判断其是否适合在中国境内网络平台发布

## 评估维度
1. **政治敏感性**
   - 是否包含不当政治表述
   - 是否涉及敏感政治人物/事件

2. **宗教文化适配**
   - 是否含有宗教歧视内容
   - 是否符合社会主义核心价值观

3. **法律合规性**
   - 是否违反网络安全法
   - 是否包含违法信息

4. **社会文化规范**
   - 是否存在地域/民族歧视
   - 是否使用侮辱性语言
   - 是否包含低俗内容

## 返回格式要求
```json
{
  "passed": <0|1>,
  "reason": "<审核结果说明>"
} 
"""

            comments = Comment.query.filter_by(status=0).order_by(Comment.id).all()

            # 记录API调用次数
            api_call_count = 0
            for comment in comments:
                api_call_count += 1
                print(
                    f"[进程ID: {process_id}] 正在审核评论ID: {comment.id}, 内容: '{comment.content}' (API调用次数: {api_call_count})"
                )

                generation_response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": comment.content},
                    ],
                    response_format={"type": "json_object"},
                    stream=False,
                )

                result = eval(generation_response.choices[0].message.content)

                comment.status = 1 if result["passed"] == 1 else 2
                db.session.commit()

                if result["passed"] == 1:
                    print(f"[进程ID: {process_id}] {comment.id} 自动审核通过！")
                else:
                    print(
                        f"[进程ID: {process_id}] [{comment.id}] '{comment.content}' 有问题 : {result['reason']} ."
                    )

            # 计算执行时间
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            print(
                f"[进程ID: {process_id}] 本次审核共调用OpenAI API {api_call_count} 次，执行时间: {execution_time:.2f}秒"
            )

        # 更新最后执行时间
        _last_execution_time = current_time
    finally:
        # 无论如何都要标记任务结束
        with _task_lock:
            _task_running = False


# 路由定义
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content = request.form["content"]
        new_comment = Comment(
            content=content, status=0, created_at=int(datetime.now().timestamp())
        )

        try:
            db.session.add(new_comment)
            db.session.commit()
            return redirect("/")
        except Exception as e:
            print(f"添加评论错误: {e}")
            return "There was an issue adding your comment"
    else:
        comments = Comment.query.filter_by(status=1).order_by(Comment.id.desc()).all()
        return render_template("index.html", comments=comments)


@app.route("/delete/<int:id>")
def delete(id):
    comment_to_delete = Comment.query.get_or_404(id)

    try:
        db.session.delete(comment_to_delete)
        db.session.commit()
        return redirect("/")
    except Exception as e:
        print(f"删除评论错误: {e}")
        return "There was a problem deleting that comment"


@app.route("/admin")
def admin():
    comments = Comment.query.filter_by(status=0).order_by(Comment.id).all()
    return render_template("admin.html", comments=comments)


@app.route("/passone/<int:id>")
def passone(id):
    comment_to_pass = Comment.query.get_or_404(id)

    try:
        comment_to_pass.status = 1
        db.session.commit()
        return redirect("/admin")
    except Exception as e:
        print(f"通过评论错误: {e}")
        return "There was a problem passing that comment"


@app.route("/reject/<int:id>")
def rejectone(id):
    comment_to_reject = Comment.query.get_or_404(id)

    try:
        comment_to_reject.status = 2
        db.session.commit()
        return redirect("/admin")
    except Exception as e:
        print(f"拒绝评论错误: {e}")
        return "There was a problem rejecting that comment"


# 检查数据库文件是否存在，不存在则创建数据库和表
def init_db():
    process_id = os.getpid()
    db_path = BASE_DIR / "comments.db"
    if not db_path.exists():
        print(f"[进程ID: {process_id}] 数据库文件不存在，正在创建: {db_path}")
        with app.app_context():
            db.create_all()
            print(f"[进程ID: {process_id}] 数据库表创建成功")
    else:
        print(f"[进程ID: {process_id}] 数据库文件已存在: {db_path}")


if __name__ == "__main__":
    # 确保初始状态正确
    _task_running = False
    _last_execution_time = None

    # 初始化数据库
    init_db()

    # 手动添加任务并启动调度器
    scheduler.add_job(
        id="auto_audit",
        func=auto_audit,
        trigger="interval",
        seconds=30,
        max_instances=1,
        coalesce=True,
    )

    # 启动调度器
    scheduler.start()

    # 启动Flask应用
    app.run(port=5000, debug=False)
