"""본 학습이 저장할 때마다 회차 번호를 붙여 따로 복사해 둔다.

train_rl.py 는 검증 점수가 오를 때마다 T_fly871.pt 를 덮어쓴다. 그래서
학습이 끝나면 '마지막으로 오른 순간' 하나만 남는다. 그 하나가 32트랙
검증셋에서 운 좋게 튄 순간이면 그걸 대회에 들고 가게 된다. R안이
852회차에서 그렇게 뽑혔다.

덮어쓰기 전에 매번 복사해 두면, 끝나고 나서 그 후보들을 전부 넓은
세트(일반 45 x 섭동 + 지그재그 12 + 대회 코스 7노면)에서 다시 재서
고를 수 있다. 한 번 잰 점수로 고르는 게 아니라 여러 후보를 다시 재서
고르는 것이라 운이 끼어들 자리가 없다.

  python keep_checkpoints.py T_fly871.pt
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil
import time
from pathlib import Path

import torch

from flyracer.paths import model as model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="T_fly871.pt")
    parser.add_argument("--every", type=float, default=30.0,
                        help="몇 초마다 들여다볼지")
    args = parser.parse_args()

    src = model_path(args.name)
    keep = src.parent / "checkpoints"
    keep.mkdir(exist_ok=True)

    seen = None
    saved = 0
    print(f"{src.name} 을 지켜본다. 바뀔 때마다 {keep.name}/ 에 복사한다.",
          flush=True)
    while True:
        try:
            stamp = src.stat().st_mtime
        except FileNotFoundError:
            time.sleep(args.every)
            continue
        if stamp != seen:
            # 쓰는 중에 읽으면 깨진 파일을 복사한다. 다 쓸 때까지 기다린다.
            time.sleep(3.0)
            try:
                blob = torch.load(src, map_location="cpu", weights_only=False)
                it = int(blob.get("iteration", 0))
            except Exception:
                time.sleep(args.every)
                continue
            out = keep / f"{src.stem}_it{it:04d}.pt"
            if not out.exists():
                shutil.copy2(src, out)
                saved += 1
                print(f"  {time.strftime('%H:%M:%S')}  {out.name} "
                      f"(모아 둔 것 {saved}개)", flush=True)
            seen = stamp
        time.sleep(args.every)


if __name__ == "__main__":
    main()
