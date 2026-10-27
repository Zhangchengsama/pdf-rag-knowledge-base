"""统一入口：python main.py 启动网页，python main.py --cli 进入终端。"""
import argparse
from pathlib import Path
import subprocess
import sys
from config import load_settings


def parse_arguments(argv=None):
    # 选择模式
    # 1. 读取命令行参数。
    # 2. 网页工作模式与终端模式互斥，没指定时启动网页服务。
    parser = argparse.ArgumentParser(description="PDF RAG 知识库")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cli", action="store_true", help="在终端中提问")
    modes.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def launch_web():
    # 启动网页
    # 1. 使用当前 Python 解释器调用 Streamlit 官方入口。
    # 2. 给同一个文件传入网页工作标记，避免再次启动服务。
    command = [sys.executable, "-m", "streamlit", "run",
               str(Path(__file__).resolve()), "--", "--web"]
    return subprocess.run(command).returncode


def main(argv=None):
    # 分派入口
    # 1. 先处理启动参数和帮助信息。
    # 2. 需要运行时再导入对应界面，共享同一份配置。
    args = parse_arguments(argv)
    if args.cli:
        from cli import run_cli
        run_cli(load_settings())
    elif args.web:
        from ui import run_web
        run_web(load_settings())
    else:
        return launch_web()
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
