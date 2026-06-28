#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass


# 从 .env 文件解析键值对并加载进进程环境变量中
load_dotenv(override=True)

base_url = os.getenv("ANTHROPIC_BASE_URL")
auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
model = os.getenv("MODEL")

# 凭据校验
if not base_url:
    raise ValueError("缺少 ANTHROPIC_BASE_URL，请设置环境变量")
if not auth_token:
    raise ValueError("缺少 ANTHROPIC_AUTH_TOKEN， 请设置环境变量")
if not model:
    raise ValueError("缺少 MODEL，请设置环境变量")

# 用 auth_token 认证时，必须清除可能残留的 ANTHROPIC_API_KEY
# 否则 SDK 会同时发送 x-api-key 和 Authorization: Bearer，API 返回 401
os.environ.pop("ANTHROPIC_API_KEY", None)

# 当前工作目录
work_dir = Path.cwd()

# 构造一个客户端对象
client = Anthropic(
            base_url = base_url,
            auth_token = auth_token
       )

# 定义系统提示词
system = f"你是一个工作在 {work_dir} 的编码 Agent。使用 bash 来处理任务。执行，不做解释"

# 定义工具列表
tools = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read file content.",
        "input_schema": {
            "type": "object",
            "properties": { "path": { "type": "string" }, "limit": { "type": "integer" } },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": { "path": {"type": "string"}, "content": {"type": "string"} },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": { "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": { "type": "string" } },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": { "pattern": {"type": "string" } },
            "required": ["pattern"]
        }
    }
]



# Bash 工具函数
def run_bash(command: str) -> str:
    # 危险命令拦截
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    # 启动子进程执行命令，阻塞等待它结束
    # 返回一个 CompletedProcess 对象，里面封装了退出码、标准输出、标准错误
    try:
        r = subprocess.run(command, shell=True, cwd=work_dir, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout(120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

# 确保传入路径位于工作目录下，挡住一切越界访问
def safe_path(p: str) -> Path:
    path = (work_dir / p).resolve()  # 拼接并解析成绝对路径
    if not path.is_relative_to(work_dir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

# Read 工具函数
def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# Write 工具函数
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True) # 自动创建父目录
        file_path.write_text(content)  # 覆盖写入内容
        return f"Wrote {len(content.encode())} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

# Edit 工具函数
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

# Glob 工具函数
def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        # 将工作目录作为根目录匹配 pattern，但 glob 函数本身不做越界防护:pattern 里若含 .. 或绝对路径片段,glob 仍可能返回工作区之外的路径。
        # 因此这里需要校验
        for match in g.glob(pattern, root_dir=work_dir):
            if (work_dir / match).resolve().is_relative_to(work_dir):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# 工具分发映射
tools_handlers = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# Agent Loop
def agent_loop(messages: list):

    while True:
        # 向模型发送请求
        response = client.messages.create(
            max_tokens=8000,
            messages=messages,
            model=model,
            system=system,
            tools=tools
        )

        # 模型回复消息加入消息列表
        messages.append({"role": "assistant", "content": response.content})

        # 判断模型是否需要调用工具，即判断 stop_reason 属性是否是 `tool_use`
        if response.stop_reason != "tool_use":
            return

        # 执行每个工具调用，收集结果
        # content 是个列表，因为模型一轮回复可能由多个块组成（先思考、再说话、再调工具）。每个块都是带 type 字段的对象，常见三种：
        # 文本块：TextBlock(type='text', text='模型的回复文字')
        # 思考块（开启 thinking 时出现）：ThinkingBlock(type='thinking', thinking='...思考过程...')
        # 工具调用块：ToolUseBlock(type='tool_use', id='toolu_xxx', name='get_weather', input={'location': '北京'})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.name}\033[0m")
                handler = tools_handlers.get(block.name)    # 根据工具名映射要调用的工具函数
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(str(output)[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # tool_result 的 id 与 tool_use 的 id 一一对应
                    "content": output
                })

        # 将工具调用结果作为 user 信息追加到消息列表回送到模型
        messages.append( {"role": "user", "content": results} )

if __name__ == '__main__':
    print("s02: Tool Use — 在 s01 的基础上加了 4 个工具")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 消息历史
    history = []

    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 用户消息加入对话历史
        history.append({"role": "user", "content": query})

        # 执行 Agent Loop
        agent_loop(history)

        # 打印模型的最后一个文本消息
        # 模型会生成内容块列表，例如
        # TextBlock(
        #    type='text'
        #    text="xxxxxx"
        # )
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)

        print()