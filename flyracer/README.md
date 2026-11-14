# flyracer — 핵심 패키지

시뮬레이터, 초파리 회로, 주행 알고리즘이 들어 있는 패키지입니다. 다른 폴더의 스크립트는 모두 이 패키지를 불러 씁니다.

| 폴더/파일 | 내용 |
|---|---|
| `sim/` | ROBORACER(F1TENTH) 규격 시뮬레이터. 차량 물리, 라이다·IMU, 트랙 생성, 대회 맵 읽기, 장애물·상대 차, 센서 잡음, 병렬 환경 |
| `brain/` | 커넥톰 CSV를 학습 가능한 CfC 회로로 바꾸는 코드(`build_wiring.py`), 라이다·IMU를 읽고 조향·속도를 내는 운전자(`fly_driver.py`), PPO 정책(`racing_policy.py`), 비교용 일반 신경망(`standard_net.py`) |
| `drive/` | 교사와 비교 기준. 최소곡률 레이싱 라인 + Stanley(`raceline.py`), 회피까지 하는 교사(`racing_teacher.py`), 고전 제어(`controllers.py`), 추월 층(`overtake.py`) |
| `paths.py` | 데이터·모델·출력 파일 위치를 한곳에 모아 둔 파일 |
| `progress.py` | 진행 막대와 로그 출력 |

파일별 자세한 설명은 [`docs/CODE_GUIDE.md`](../docs/CODE_GUIDE.md)에 있습니다.

---
[← 전체 구성으로](../README.md)
