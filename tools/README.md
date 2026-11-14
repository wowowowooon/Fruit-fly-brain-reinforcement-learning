# tools — 데이터 만들기와 사전 점검

가끔 한 번씩 돌리는 도구입니다.

| 파일 | 내용 |
|---|---|
| `fetch_driving_circuit.py` | neuPrint에서 주행에 쓸 만한 회로를 넓게 받아 `data/circuit_*.csv`로 저장 (API 토큰 필요, 실행 시 입력) |
| `fetch_eye_position.py` | 시각 뉴런의 시야 방향을 받아 `data/sensory_position.csv`로 저장 |
| `build_subcircuit.py` | 받은 회로에서 시각 → 조향으로 실제 이어지는 부분만 추려 `data/subcircuit_*.csv` 생성 |
| `tune_tracks.py` | 교사가 제대로 도는지 확인하고 트랙 난이도 조정. 모방학습 전에 먼저 통과시킴 |
| `verify_fast_mode.py` | 라이다 빔 수를 줄여 학습해도 결과가 같은지 확인 |
| `bench_vec.py` | 병렬 환경 처리량을 재서 강화학습 설정을 고름 |
