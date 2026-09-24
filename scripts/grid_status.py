"""격자 진행 상황을 한 번만 찍는다.

  .\\.venv\\Scripts\\python.exe grid_status.py

실시간으로 보려면 watch_grid.ps1 을 쓴다.
"""
import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pathlib
import re

# scripts/ 안에서 한 칸 위가 저장소 뿌리다
ROOT = pathlib.Path(__file__).resolve().parent.parent
# 격자 순서대로. 아직 시작 안 한 판도 '대기' 로 보여준다.
ORDER = [f"{caps}_{gam}_s{seed}"
         for seed in (0, 1, 2)
         for gam in ("g99", "g998")
         for caps in ("none", "mid", "high")]

EMPTY = {"state": "대기", "it": "-", "rate": "-", "speed": "-",
         "eta": "-", "best": "-", "now": "-"}


def read(name):
    path = ROOT / "logs" / f"grid_{name}.log"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r", "\n")
    rows = [line for line in text.split("\n") if "강화 " in line]
    if not rows:
        return None
    last = rows[-1]

    def grab(pattern):
        found = re.search(pattern, last)
        return found.group(1) if found else "-"

    valid = [int(v) for v in re.findall(r"검증 +(\d+)%", text)]
    return {
        "state": "끝" if "강화 완료" in text else "진행",
        "it": grab(r"강화 +(\d+/\d+)"),
        "rate": grab(r"완주 +(\d+%)"),
        "speed": grab(r"평균속도 +([\d.]+)"),
        "eta": grab(r"남음 +([\d:]+)"),
        "best": f"{max(valid)}%" if valid else "-",
        "now": f"{valid[-1]}%" if valid else "-",
    }


def main():
    print("격자 18판  —  CAPS 3종 x 감마 2종 x 시드 3개")
    print("=" * 86)
    done = 0
    for name in ORDER:
        row = read(name) or EMPTY
        done += row["state"] == "끝"
        print(f"{name:<16s} {row['state']:<4s} {row['it']:>9s}  "
              f"완주 {row['rate']:>5s}  속도 {row['speed']:>4s}  "
              f"검증 최고 {row['best']:>5s} 지금 {row['now']:>5s}  "
              f"남음 {row['eta']}")
    print("=" * 86)
    print(f"끝난 판 {done}/18")


if __name__ == "__main__":
    main()
