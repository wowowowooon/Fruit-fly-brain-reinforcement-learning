"""주행에 쓸 만한 초파리 회로를 최대 범위로 받아서 CSV로 저장.

역할 3층으로 태깅한다.
  sensory : 시각 흐름 / 충돌 감지  -> 차로 치면 카메라
  inter   : 중심복합체 (방향, 목표) -> 컨트롤러 내부
  motor   : 하강 뉴런              -> 조향, 가감속
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import DATA

import os
from getpass import getpass

import pandas as pd
from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_adjacencies

# 약한 연결은 노이즈로 보고 버린다.
MIN_WEIGHT = 5

# 역할별로 넓게 잡은 타입 패턴. 나중에 필요 없는 건 여기서 지우면 된다.
GROUPS = {
    "sensory": [
        r"^HS[NES]$",           # 수평 시각 흐름 (좌우 회전)
        r"^VS\d+$",             # 수직 시각 흐름 (피치/롤)
        r"^H2$", r"^Hx$",       # 반대방향 수평 운동
        r"^LPT\d+.*",           # 나머지 lobula plate 광역 세포
        r"^LC4$",               # 접근 물체(looming) 감지
        r"^LPLC[12]$",          # looming, 충돌 임박
        r"^LC6$", r"^LC9$",     # 물체 감지
    ],
    "inter": [
        r"^EPG.*", r"^PEN.*", r"^PEG$",   # 방향 나침반
        r"^ER\d.*", r"^EL$", r"^ExR\d$",  # 방향 기준 입력
        r"^Delta7$",                      # 나침반 정규화
        r"^PFL\d$", r"^PFN.*", r"^PFR.*", # 방향 오차 -> 조향 변환
        r"^FC\d.*", r"^FS\d.*",           # 목표 방향
        r"^hDelta.*", r"^vDelta.*",       # 목표 저장
        r"^LAL\d+.*",                     # 중심복합체 -> DN 중계
    ],
    "motor": [
        r"^DNa\d+$",   # 걷기 조향 (DNa01, DNa02 등)
        r"^DNb\d+$",   # 다리 제어, 리듬 생성 (DNb08)
        r"^DNg\d+$",   # 조향, 전진 속도 (DNg13, DNg100=BDN2)
        r"^DNp\d+$",   # 도피, 급정지 (DNp09)
        # 아래는 DNxx 이름 규칙을 안 따르는 운동 명령 뉴런
        r"^MDN.*",     # 후진, 감속. 차로 치면 브레이크
        r"^oDN\d+$",   # 걷기 개시
    ],
}


def connect():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        token = getpass("neuPrint API 토큰을 붙여넣고 Enter: ").strip()
    return Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)


# 흥분/억제 판정에 쓸 신경전달물질 컬럼. 데이터셋마다 이름이 달라서 있는 걸 쓴다.
NT_COLUMNS = ["consensusNt", "predictedNt", "celltypePredictedNt", "neurotransmitter"]


def fetch_group(role, patterns, client):
    """한 역할에 속한 뉴런을 모두 모은다."""
    frames = []
    for pattern in patterns:
        criteria = NeuronCriteria(type=pattern, regex=True, client=client)
        neurons, _ = fetch_neurons(criteria, client=client)
        if len(neurons) == 0:
            print(f"  {pattern:<16} 없음")
            continue
        print(f"  {pattern:<16} {len(neurons):>5}개  ({neurons['type'].nunique()}종)")

        nt = next((c for c in NT_COLUMNS if c in neurons.columns), None)
        columns = ["bodyId", "type", "instance"] + ([nt] if nt else [])
        group = neurons[columns]
        frames.append(group.rename(columns={nt: "nt"}) if nt else group.assign(nt=None))

    if not frames:
        return pd.DataFrame(columns=["bodyId", "type", "instance", "nt", "role"])

    group = pd.concat(frames).drop_duplicates("bodyId")
    group["role"] = role
    return group


def main():
    client = connect()
    print("서버 연결 완료.\n")

    groups = []
    for role, patterns in GROUPS.items():
        print(f"[{role}]")
        groups.append(fetch_group(role, patterns, client))
        print()

    neurons = pd.concat(groups).drop_duplicates("bodyId").reset_index(drop=True)
    print(f"총 뉴런 {len(neurons)}개")
    print(neurons["role"].value_counts().to_string(), "\n")

    if neurons["nt"].notna().any():
        print("신경전달물질 분포")
        print(neurons["nt"].value_counts().to_string(), "\n")

    print(f"연결표 조회 중 (weight >= {MIN_WEIGHT})...")
    ids = neurons["bodyId"].tolist()
    _, connections = fetch_adjacencies(
        ids, ids,
        min_total_weight=MIN_WEIGHT,
        omit_rois=True,
        client=client,
    )
    print(f"연결 {len(connections)}개\n")

    neurons.to_csv(DATA / "circuit_neurons.csv", index=False)
    connections.to_csv(DATA / "circuit_connections.csv", index=False)
    print("저장: circuit_neurons.csv, circuit_connections.csv")

    density = len(connections) / (len(neurons) ** 2)
    print(f"\n뉴런당 평균 연결: {len(connections) / len(neurons):.1f}")
    print(f"배선 밀도: {density:.3%}  (100% - 밀도 = 희소성)")


if __name__ == "__main__":
    main()

