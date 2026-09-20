
import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from getpass import getpass
from neuprint import Client, fetch_neurons

# 실행할 때 토큰을 입력받음. 파일에는 저장하지 않음.
token = getpass("neuPrint API 토큰을 붙여넣고 Enter: ").strip()

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print("\n서버 연결 완료. 뉴런 정보를 불러옵니다...")

# 공식 예제에 있는 뉴런 유형 조회
neurons, roi_info = fetch_neurons("DNge104", client=client)

columns = [
    name
    for name in ["bodyId", "type", "instance", "pre", "post"]
    if name in neurons.columns
]

print(f"\n조회된 뉴런 수: {len(neurons)}")
print(neurons[columns].to_string(index=False))

