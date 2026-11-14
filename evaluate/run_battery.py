"""모델 하나를 모든 잣대로 잰다. 밤새 혼자 돌게 만든 것.

B 안과 C 안을 나란히 놓으려면 재는 방법이 완전히 같아야 한다. 사람이
명령을 하나씩 치면 옵션 하나를 빠뜨린다. 그래서 목록을 코드에 박아 둔다.

  python experiments/run_battery.py C_fly871.pt
  python experiments/run_battery.py B_dense871.pt --dense
  python experiments/run_battery.py C_fly871.pt --only robust
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flyracer.paths import RACE_MAP

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAP = str(RACE_MAP)

# (이름, 스크립트, 붙일 인자). --dense 는 필요하면 자동으로 붙는다.
STEPS = [
    ("basic", "evaluate.py", ["--tracks", "45", "--map", MAP]),
    ("obstacles", "experiment_obstacles.py", ["--tracks", "45"]),
    ("dynamic", "experiment_dynamic.py", ["--tracks", "45"]),
    ("faults", "experiment_faults.py", ["--tracks", "12"]),
    ("robust", "experiment_robustness.py", ["--tracks", "20"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--only", nargs="*", default=None,
                        help="이 단계만 돌린다")
    parser.add_argument("--skip", nargs="*", default=(),
                        help="이 단계는 건너뛴다")
    args = parser.parse_args()

    name = Path(args.model).stem
    steps = [s for s in STEPS if (args.only is None or s[0] in args.only)
             and s[0] not in args.skip]
    print("=" * 74)
    print(f"{name} 전체 측정 — {', '.join(s[0] for s in steps)}")
    print("=" * 74, flush=True)

    started = time.time()
    for label, script, extra in steps:
        command = [sys.executable, str(HERE / script), args.model] + list(extra)
        if args.dense:
            command.append("--dense")
        print(f"\n{'=' * 74}\n[{label}] {' '.join(command[2:])}\n{'=' * 74}",
              flush=True)
        step_started = time.time()
        done = subprocess.run(command, cwd=str(ROOT))
        mark = "끝" if done.returncode == 0 else f"실패(코드 {done.returncode})"
        print(f"[{label}] {mark}  {time.time() - step_started:.0f}초", flush=True)

    print(f"\n{name} 전체 측정 끝. {(time.time() - started) / 60:.0f}분")


if __name__ == "__main__":
    main()
