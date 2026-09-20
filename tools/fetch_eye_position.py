"""시각 뉴런이 시야의 어느 방향을 보는지 받아온다.

초파리 겹눈은 낱눈이 육각 격자로 박혀 있고, 시신경엽 안쪽까지 그 배열이
그대로 이어진다. 커넥톰은 시냅스마다 그 격자 좌표(olHex1, olHex2)를
가지고 있다. 즉 한 세포의 가지가 격자 어디에 뻗어 있는지를 보면
그 세포가 시야의 어느 방향을 담당하는지 알 수 있다.

이 정보가 없으면 뉴런을 시야 구역에 임의로 나눠 넣게 되고, 그러면
초파리 배선을 그대로 가져와도 엉뚱한 단자에 꽂는 셈이 된다.

  set NEUPRINT_TOKEN=...
  python tools/fetch_eye_position.py
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import numpy as np
import pandas as pd
from neuprint import Client

from paths import DATA

# 가지(입력)가 뻗어 있는 곳이 그 세포가 보는 방향이다.
# 로불라(LO)와 로불라판(LOP)이 시야 배열을 유지하는 곳.
VISUAL_ROIS = ("LO", "LOP", "ME")

QUERY = """
MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
WHERE n.bodyId IN [{bodies}]
  AND s.olHex1 IS NOT NULL
  AND s.type = 'post'
RETURN n.bodyId        AS bodyId,
       avg(s.olHex1)   AS hex1,
       avg(s.olHex2)   AS hex2,
       count(s)        AS n_synapse
"""


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("NEUPRINT_TOKEN 환경변수를 먼저 설정하세요")
    client = Client("https://neuprint.janelia.org",
                    dataset="male-cns:v1.0", token=token)

    neurons = pd.read_csv(DATA / "subcircuit_neurons.csv")
    sensory = neurons[neurons.role == "sensory"]
    bodies = [int(b) for b in sensory.bodyId]
    print(f"시각 뉴런 {len(bodies)}개의 시야 위치를 받는다")

    # 한 번에 다 넣으면 질의가 너무 길어진다. 나눠서 받는다.
    chunks = []
    for start in range(0, len(bodies), 60):
        group = bodies[start:start + 60]
        query = QUERY.format(bodies=", ".join(str(b) for b in group))
        chunks.append(client.fetch_custom(query, format="pandas"))
        print(f"  {min(start + 60, len(bodies))}/{len(bodies)}")
    frame = pd.concat(chunks, ignore_index=True)
    print(f"위치를 얻은 뉴런: {len(frame)}개")

    got = frame.merge(sensory[["bodyId", "type", "instance"]], on="bodyId")

    # 육각 좌표를 좌우 각도 하나로 편다.
    # 두 축이 60도로 어긋나 있으므로 그대로 빼면 안 되고 직교로 바꾼다.
    hx = got.hex1 - got.hex1.mean()
    hy = got.hex2 - got.hex2.mean()
    got["azimuth"] = hx - hy * 0.5      # 좌우 방향
    got["elevation"] = hy * np.sqrt(3) / 2   # 위아래 방향

    # 반구별로 0~1 로 정규화한다. 0이 가장 앞, 1이 가장 옆.
    got["side"] = got.instance.astype(str).str[-1:]
    got["front_to_side"] = np.nan
    for side in ("L", "R"):
        rows = got.side == side
        if not rows.any():
            continue
        values = got.loc[rows, "azimuth"]
        span = values.max() - values.min()
        norm = (values - values.min()) / (span if span else 1.0)
        # 오른쪽 눈은 격자 방향이 뒤집혀 있다. 둘 다 '앞=0'이 되게 맞춘다.
        got.loc[rows, "front_to_side"] = norm if side == "L" else 1.0 - norm

    missing = set(bodies) - set(got.bodyId)
    out = DATA / "sensory_position.csv"
    got[["bodyId", "type", "instance", "side", "hex1", "hex2",
         "azimuth", "elevation", "front_to_side", "n_synapse"]].to_csv(
        out, index=False)

    print(f"\n저장: {out.relative_to(DATA.parent)}")
    print(f"  위치 확보 {len(got)}개 / 위치 없음 {len(missing)}개")
    print("\n종류별 확보 현황")
    for name, group in got.groupby("type"):
        total = int((sensory.type == name).sum())
        print(f"  {str(name):10s} {len(group):4d} / {total:4d}개   "
              f"좌우각 {group.azimuth.min():6.1f} ~ {group.azimuth.max():6.1f}")
    if missing:
        rows = sensory[sensory.bodyId.isin(missing)]
        print(f"\n위치가 없는 뉴런의 종류: "
              f"{dict(rows.type.value_counts().head(10))}")
        print("  (광역 세포는 시야 전체를 보므로 위치가 없어도 된다)")


if __name__ == "__main__":
    main()
