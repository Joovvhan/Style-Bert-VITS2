import sys
from pathlib import Path

# decompose.py / alignment.py를 베이스 디렉토리에서 pytest 실행 시에도 찾을 수 있게 함
sys.path.insert(0, str(Path(__file__).parent))
