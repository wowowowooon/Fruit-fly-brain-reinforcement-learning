# data — 초파리 회로 데이터

neuPrint **male-cns v1.0**(초파리 수컷 중추신경계 커넥톰)에서 받은 데이터입니다. `tools/fetch_*.py`, `tools/build_subcircuit.py`로 다시 만들 수 있습니다.

| 파일 | 내용 |
|---|---|
| `circuit_neurons.csv` | 처음 받은 넓은 주행 회로의 뉴런 (`bodyId, type, instance, nt, role`) |
| `circuit_connections.csv` | 위 뉴런 사이의 연결과 시냅스 수 (`bodyId_pre, bodyId_post, weight, weightHP`) |
| `subcircuit_neurons.csv` | 시각 → 조향·구동으로 실제 이어지는 부분만 추린 **최종 회로 871개** 뉴런 |
| `subcircuit_connections.csv` | 최종 회로의 연결 9,992개 |
| `sensory_position.csv` | 시각 뉴런이 보는 시야 방향(겹눈 육각 좌표, 방위각·고도각). 라이다 구역 배정에 사용 |
| `real_track_widths.json` | 실제 트랙 폭 정보 |

`nt`(신경전달물질)로 흥분·억제 극성을 정하고, `role`은 감각/중간/출력 역할입니다.
