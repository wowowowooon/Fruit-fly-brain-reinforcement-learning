"""프로젝트 안 파일들이 어디 있는지 한곳에 모아 둔다.

어느 폴더에서 실행하든 같은 곳을 가리킨다. 경로를 파일마다 문자열로
박아 두면 폴더를 옮길 때마다 전부 찾아 고쳐야 한다.

  from flyracer.paths import DATA, MODELS
  pd.read_csv(DATA / "subcircuit_neurons.csv")
"""

import os
from pathlib import Path

# 이 파일은 flyracer/ 안에 있다. 저장소 뿌리는 한 칸 위다.
# parent 로 두면 models/ 와 data/ 를 flyracer/ 밑에서 찾아
# 빈 폴더를 새로 만들고 '학습 전' 이라고 말한다. 안 터져서 더 위험하다.
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"        # 커넥톰에서 받아온 뉴런/연결 표
MODELS = ROOT / "models"    # 학습된 가중치 (.pt)
OUTPUTS = ROOT / "outputs"  # 그림, 렌더 결과 (.png)
LOGS = ROOT / "logs"        # 학습 로그

# 저장소 밖에 두는 외부 데이터. 기본은 저장소 옆 폴더이고,
# 다른 곳에 두었으면 환경 변수로 알려 준다.
#   ../ifac_roboracer_maps/ifac_roboracer.yaml   2026 대회 코스 맵 (시험에만 쓴다)
#   ../f1tenth_racetracks/                       F1TENTH 공개 서킷 (github.com/f1tenth/f1tenth_racetracks)
RACE_MAP = Path(os.environ.get(
    "FLYRACER_RACE_MAP", ROOT.parent / "ifac_roboracer_maps" / "ifac_roboracer.yaml"))
F1TENTH_TRACKS = Path(os.environ.get(
    "FLYRACER_F1TENTH_TRACKS", ROOT.parent / "f1tenth_racetracks"))

for folder in (DATA, MODELS, OUTPUTS, LOGS):
    folder.mkdir(exist_ok=True)


def model(name):
    """모델 이름을 받아 실제 파일 경로로.

    폴더 없이 이름만 주면 models/ 안에서 찾는다. 경로를 직접 준 경우
    (예: 다른 폴더의 실험 모델) 그대로 쓴다.
    """
    path = Path(name)
    return path if path.parent != Path(".") else MODELS / path.name
