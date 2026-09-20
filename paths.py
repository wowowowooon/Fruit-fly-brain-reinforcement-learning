"""프로젝트 안 파일들이 어디 있는지 한곳에 모아 둔다.

어느 폴더에서 실행하든 같은 곳을 가리킨다. 경로를 파일마다 문자열로
박아 두면 폴더를 옮길 때마다 전부 찾아 고쳐야 한다.

  from paths import DATA, MODELS
  pd.read_csv(DATA / "subcircuit_neurons.csv")
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"        # 커넥톰에서 받아온 뉴런/연결 표
MODELS = ROOT / "models"    # 학습된 가중치 (.pt)
OUTPUTS = ROOT / "outputs"  # 그림, 렌더 결과 (.png)
LOGS = ROOT / "logs"        # 학습 로그

for folder in (DATA, MODELS, OUTPUTS, LOGS):
    folder.mkdir(exist_ok=True)


def model(name):
    """모델 이름을 받아 실제 파일 경로로.

    폴더 없이 이름만 주면 models/ 안에서 찾는다. 경로를 직접 준 경우
    (예: 다른 폴더의 실험 모델) 그대로 쓴다.
    """
    path = Path(name)
    return path if path.parent != Path(".") else MODELS / path.name
