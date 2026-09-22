"""실제 서킷을 트랙으로 쓴다 — F1TENTH 공식 공개 트랙 26개.

우리가 만드는 트랙은 푸리에 곡선과 둥근 다각형이라 모양이 한정돼 있다.
2026 대회 코스를 재 보면 가로세로비 3.42 인데 우리 학습 트랙은 최대 2.91 이라
**대회 코스가 학습 분포 밖**이다. 직접 닮은 트랙을 만들어 보려 했으나
길쭉하게 누르면 코너가 무한히 뾰족해져 전부 버려졌다.

그래서 실제 서킷을 가져온다. f1tenth/f1tenth_racetracks 저장소에
Monza, Silverstone, Catalunya 등 26개가 있고, 각 폴더의 중심선 CSV 가
`x, y, 오른쪽 폭, 왼쪽 폭` 형식이라 우리 make_track 과 같은 것을 준다.

  통로 폭      2.2m 일정   (우리 1.7~4.2m, 대회 1.8m)
  한 바퀴      261~554m    (우리 33~180m, 대회 47.5m)
  최소 코너    0.50~13.5m  (우리 1.13~6.64m)  <- 우리보다 훨씬 급하다

지도 그림(.png)은 못 쓴다. 흰 배경에 벽선만 그려 둔 것이라 트랙 밖도
전부 '주행 가능' 으로 잡힌다. CSV 만 쓴다.

  from real_tracks import track_names
  env = F1TenthEnv(circuit="Monza")
"""

import pathlib

import numpy as np

from f1tenth_sim import _resample

ROOT = pathlib.Path(r"C:\Users\woong\Desktop\f1tenth_racetracks")
POINTS = 600            # 중심선을 몇 점으로 다시 찍을지. 트랙이 길어서 넉넉히.


def track_names(root=ROOT):
    """쓸 수 있는 트랙 이름들."""
    if not pathlib.Path(root).exists():
        return []
    return sorted(d.name for d in pathlib.Path(root).iterdir()
                  if (d / f"{d.name}_centerline.csv").exists())


def load_centerline(name, root=ROOT, points=POINTS, scale=1.0):
    """중심선과 통로 폭을 읽는다. scale 로 트랙 전체 크기를 조절한다."""
    path = pathlib.Path(root) / name / f"{name}_centerline.csv"
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    line = _resample(table[:, :2] * scale, points)
    width = float(np.median(table[:, 2] + table[:, 3])) * scale
    return line, width


# 환경에서 쓰는 법:
#   F1TenthEnv(circuit="Monza")            정지 장애물만
#   MovingObstacleEnv(circuit="Monza")     상대 차까지
# 트랙을 어디서 읽을지는 F1TenthEnv 가 정한다. 여기서 따로 환경을 만들면
# 상대 차 환경과 안 섞인다.
