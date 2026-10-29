"""项目启动入口：python run.py；终端模式使用 --cli。"""
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent / 'src'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from main import main


if __name__ == '__main__':
    raise SystemExit(main())
