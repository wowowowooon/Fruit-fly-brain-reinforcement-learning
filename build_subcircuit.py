"""받아둔 전체 회로에서 '시각 -> 조향'으로 실제 이어지는 부분만 추린다.

경로에 못 걸리는 뉴런은 조향에 기여할 수 없으므로 뺀다.
학습은 하지 않는다. 추리기만 한다.
"""

import networkx as nx
import pandas as pd

# 조향: 좌우 활동 차이가 회전 속도와 비례한다고 측정된 하강 뉴런
STEERING_TYPES = ["DNa01", "DNa02", "DNa03", "DNb06", "DNg13"]

# 구동: 전진 속도, 개시, 감속을 담당하는 하강 뉴런
#   DNg100(=BDN2) 활동이 전진 속도와 직접 비례. 사실상 스로틀
#   DNp09 전진 개시, 강하게 켜지면 정지
#   DNb08 다리 리듬 생성
#   MDN 느린 전진에서 정지를 거쳐 후진까지. 브레이크
DRIVE_TYPES = ["DNg100", "DNp09", "DNb08", "MDN"]

OUTPUT_TYPES = STEERING_TYPES + DRIVE_TYPES

MIN_WEIGHT = 10   # 접점 10개 미만 연결은 버림
MAX_HOPS = 2      # 시각/조향에서 몇 단계까지 훑을지


def within_hops(graph, seeds, hops):
    """seeds에서 hops 단계 안에 닿는 노드."""
    reached = set()
    for distance in range(1, hops + 1):
        for seed in seeds & set(graph):
            reached |= nx.descendants_at_distance(graph, seed, distance)
    return reached


def main():
    neurons = pd.read_csv("circuit_neurons.csv")
    connections = pd.read_csv("circuit_connections.csv")
    print(f"전체: 뉴런 {len(neurons)}개, 연결 {len(connections)}개")

    edges = connections[connections.weight >= MIN_WEIGHT]
    graph = nx.from_pandas_edgelist(
        edges, "bodyId_pre", "bodyId_post", edge_attr="weight",
        create_using=nx.DiGraph,
    )
    print(f"약한 연결 제거 후: 연결 {len(edges)}개 (weight >= {MIN_WEIGHT})\n")

    steering = set(neurons[neurons.type.isin(STEERING_TYPES)].bodyId)
    drive = set(neurons[neurons.type.isin(DRIVE_TYPES)].bodyId)
    motor = steering | drive
    sensory = set(neurons[neurons.role == "sensory"].bodyId)
    print(f"도착점: 조향 {len(steering)}개 + 구동 {len(drive)}개 = {len(motor)}개")

    # 조향 뉴런에서 거꾸로 거슬러 올라가 닿는 것
    upstream = within_hops(graph.reverse(), motor, MAX_HOPS)
    # 그중 시각 뉴런만 출발점으로 인정
    source = sensory & upstream
    # 출발점에서 정방향으로 닿는 것
    downstream = within_hops(graph, source, MAX_HOPS)

    keep = (downstream & upstream) | motor | source
    subgraph = graph.subgraph(keep)

    sub_neurons = neurons[neurons.bodyId.isin(keep)].copy()
    sub_neurons["role"] = sub_neurons.apply(
        lambda r: "motor" if r.bodyId in motor else r.role, axis=1
    )
    # 출력 뉴런은 어느 축을 담당하는지 표시해 둔다.
    sub_neurons["axis"] = sub_neurons.bodyId.map(
        lambda b: "steering" if b in steering else "drive" if b in drive else ""
    )
    sub_edges = edges[
        edges.bodyId_pre.isin(keep) & edges.bodyId_post.isin(keep)
    ]

    print(f"\n추린 결과: 뉴런 {len(sub_neurons)}개, 연결 {len(sub_edges)}개")
    print(f"(전체의 {len(sub_neurons) / len(neurons):.1%})\n")
    print(sub_neurons.role.value_counts().to_string())

    out = sub_neurons[sub_neurons.axis != ""]
    print("\n출력 뉴런")
    for axis in ["steering", "drive"]:
        rows = out[out.axis == axis]
        types = ", ".join(f"{t}({c})" for t, c in rows.type.value_counts().items())
        print(f"  {axis:<9} {len(rows):>3}개  {types}")

    print("\n남은 타입 상위 10종")
    print(sub_neurons.type.value_counts().head(10).to_string())

    sub_neurons.to_csv("subcircuit_neurons.csv", index=False)
    sub_edges.to_csv("subcircuit_connections.csv", index=False)
    print("\n저장: subcircuit_neurons.csv, subcircuit_connections.csv")

    n = len(sub_neurons)
    print(f"\n예상 모델 크기: 약 {n * n * 5 / 1e6:.1f}M 파라미터")
    print(f"배선 밀도: {subgraph.number_of_edges() / n ** 2:.2%}")


if __name__ == "__main__":
    main()
