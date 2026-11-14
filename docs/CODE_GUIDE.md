# 초파리 커넥톰 → 자율주행 보금자리

한 줄로 말하면 이렇습니다.  
**초파리 뉴런을 차에 꽂아서 달리게 하는 프로젝트가 아닙니다.**  
나중에 더 큰 생물의 배선도가 나와도 같은 파이프에 갈아 끼울 수 있게, **어댑터와 실험대를 먼저 만드는 프로젝트**입니다.

---

## 폴더 구조

```
fly_brain_test/
├── flyracer/      핵심 코드 (패키지). 서로를 불러 쓴다
│   ├── sim/         시뮬레이터, 트랙, 장애물, 센서
│   ├── brain/       초파리 배선과 신경망
│   ├── drive/       주행 알고리즘 (레이싱 라인, 교사, 고전)
│   ├── paths.py     파일 위치 모음
│   └── progress.py  진행 막대 + 로그
├── train/         학습 (모방 → 강화)
├── evaluate/      성적 재기
├── view/          눈으로 보기
├── scripts/       진행 상황 보기, 체크포인트 모으기
├── tools/         데이터 만들기·성능 재기 (가끔)
├── archive/       다 쓴 1회성 스크립트 (참고용)
├── data/          커넥톰 데이터 (뉴런·연결·시야 위치)
├── models/        학습된 가중치 (.pt) 와 성적표 (.json)
├── outputs/       그림, 주행 궤적, 실험 결과
└── logs/          학습·실험 로그
```

`flyracer/` 만 패키지입니다. 나머지 폴더는 실행하는 스크립트들이고,
각자 맨 위에서 저장소 뿌리를 `sys.path` 에 넣어 `flyracer` 를 찾습니다.

### flyracer/sim — 시뮬레이터

| 파일 | 하는 일 |
|---|---|
| `f1tenth_sim.py` | 차 물리, 라이다·IMU, 트랙 생성. 타이어·서보 한계 |
| `track_map.py` | 대회 맵(.png+.yaml) 읽기, 맵에서 중심선 뽑기 |
| `map_start.py` | 맵 위 출발 자세 찾기, 한 바퀴 궤적 따기 |
| `map_hazards.py` | 대회 코스에 정지 장애물·상대 차 얹기 |
| `moving_obstacles.py` | 움직이는 상대 차 |
| `real_tracks.py` | F1TENTH 공개 서킷 22개 읽기 |
| `sensor_noise.py` | 센서 잡음·고장 흉내 (학습 중 랜덤화 + 실험) |
| `vec_env.py` | 여러 대를 동시에 굴리는 환경 |

### flyracer/brain — 초파리 회로

| 파일 | 하는 일 |
|---|---|
| `build_wiring.py` | 커넥톰 CSV → 학습 가능한 회로. 배선·극성 적용 |
| `fly_driver.py` | 라이다·IMU 를 읽고 조향·속도를 냄. 배선 검사 포함 |
| `racing_policy.py` | PPO 정책·가치 머리 |
| `standard_net.py` | 비교용 일반 신경망 (A안) |

### flyracer/drive — 주행 제어

| 파일 | 하는 일 |
|---|---|
| `raceline.py` | 최소곡률 라인 + Stanley 추종 |
| `racing_teacher.py` | 교사 — 코리도 DP + 속도 계획 + 회피 |
| `controllers.py` | 고전 방식 (Follow the Gap, 순수추종) — 비교 기준 |

### 자주 쓰는 명령

모델마다 관측 채널 수가 다릅니다. **`--motion` 을 꼭 맞춰야 합니다.**
T안은 6, R·B·C안은 3입니다. 틀리면 조용히 어긋나거나 터집니다.

```powershell
# 눈으로 보기 (대회 코스, 장애물 6개 + 상대 차 2대, 끌 때까지 계속)
.\.venv\Scripts\python.exe view\viewer.py --model T_fly871_final.pt --motion 6 `
  --map "C:\Users\woong\Desktop\ifac_roboracer_maps\ifac_roboracer.yaml" `
  --obstacles 6 --rivals 2

# 대회 코스 한 바퀴, 숫자로
.\.venv\Scripts\python.exe view\test_on_map.py "...\ifac_roboracer.yaml" `
  --model T_fly871_final.pt --motion 6

# 회피 장면 세 컷을 그림으로
.\.venv\Scripts\python.exe view\race_avoid.py --model T_fly871_final.pt --motion 6

# 성적 재기
.\.venv\Scripts\python.exe evaluate\evaluate.py T_fly871_final.pt --tracks 45 --motion 6
.\.venv\Scripts\python.exe evaluate\experiment_faults.py T_fly871_final.pt --motion 6
.\.venv\Scripts\python.exe evaluate\race_robust.py --model T_fly871_final.pt --motion 6

# 학습
.\.venv\Scripts\python.exe train\train_imitation.py --out T_fly871.pt --motion 6
.\.venv\Scripts\python.exe train\train_rl.py --iters 3000 --init AB_mid.pt --motion 6 --out T_fly871.pt
```

파일 위치는 `flyracer/paths.py` 를 거칩니다. 폴더를 옮겨도 거기만 고치면 됩니다.

---

## 지금 어디까지 왔나

**C안(초파리 배선) 완성.** 맵 없이 라이다와 IMU 만으로 2026 대회 코스를 완주한다.

| | 값 |
|---|---|
| 처음 보는 트랙 45개 | 39/45 완주 (87%) |
| 좁은 트랙 (폭 2.0m 이하) | 22/23 (96%) |
| **2026 대회 코스** | **완주 12.2초** |
| 실제 사용 파라미터 | 51,302 |

### 초파리에서 가져온 것

| | 출처 |
|---|---|
| 어느 뉴런이 어느 뉴런에 연결되나 (9,992개) | 커넥톰 |
| 흥분(+) / 억제(−) (36%) | 커넥톰 신경전달물질 |
| 연결 세기 초기값 | 커넥톰 접점 수 |
| 시야 방향 (231개) | 겹눈 육각 좌표 |
| 출력 뉴런 선택 | DNa01/02/03·DNb06·DNg13 (조향), DNg100·DNp09·DNb08·MDN (구동) |
| 조향 = 좌우 활동 차이 | 초파리 조향 원리 |
| **연결 세기** | **학습** |

`fly_driver.check_wiring()` 이 학습을 시작하기 전에 배선 수·억제성 개수·극성
유지율을 센다. 하나라도 어긋나면 학습이 시작되지 않는다. 배선이나 극성이
조용히 빠진 채로 몇 시간을 돌린 일이 있어서 넣었다.

### 센서 고장 견딤 (시드 700~711)

| 손상 | 완주 |
|---|---|
| 빔 50% 손실 | 100% |
| 사거리 10m → 5m | 100% |
| 거리 잡음 ±1m | 83% |
| 값 70% 굳음 | 92% |
| 한쪽 시야 차단 | 0% |

한쪽 시야가 막히면 못 달린다. 조향을 좌우 차이로 읽기 때문에 한쪽이
죽으면 차이가 영구히 쏠린다. 대회 코스에서는 일어나지 않는 상황이라
그대로 둔다.

---

## 비교 실험 설계

배선이 기여하는지 보려면 변수를 하나만 두어야 한다.

| | 뉴런 | 배선 | 연결 | 실제 사용 파라미터 | 상태 |
|---|---|---|---|---|---|
| **A** | GRU 256 | 일반 신경망 | — | 228,120 | 모듈만 있음 |
| **B** | 871 | 전부 연결 | 758,641 | 3,045,898 | 아직 |
| **C** | 871 | **초파리** | 9,992 | 51,302 | **완료** |

**B vs C 가 핵심이다.** 뉴런 수가 같고 연결만 다르다. A 는 인과 비교가
아니라 실무 수준과 견줄 만한지 보는 참고선이다.

평가는 시드 700~744 에서 한다. 학습에 쓴 0~399 와 절대 겹치면 안 된다.

---

## 남은 일

1. **B안 학습** — 배선이 기여하는가에 대한 답
2. **레이싱 라인 수정** — 지금 교사가 트랙 75% 에서 중심선만 탄다.
   `_enforce_clearance` 가 점을 하나씩 밀어내 꺾임을 만들고, 그러면
   최적화 결과를 버리고 중심선으로 되돌린다. 랩타임을 줄이려면 여기부터다
3. **C 가지치기** — 실제 주행에서 안 쓰는 뉴런을 찾아 줄인다
4. **동적 장애물·상대 차량** — 접근 감지(LC4/LPLC)가 제 일을 할 무대
5. **ROS2 노드로 감싸 실차에 올리기**

### 실차 올리기 전에 재야 할 값

시뮬의 이 다섯 개는 F1TENTH 표준값이지 우리 차를 잰 값이 아니다.

| 값 | 현재 | 재는 법 |
|---|---|---|
| 타이어 마찰 | 1.0489 | 정상원 선회에서 미끄러지는 속도 |
| 서보 속도 | 3.2 rad/s | 핸들 좌→우 풀스윙 영상 |
| 가속 / 감속 | 4.0 / 6.0 m/s² | 직선 전속 → 급제동 |
| 제어 지연 | 50 ms | 명령 발행 → 바퀴 반응 타임스탬프 |
| 최고 속도 | 7.0 m/s | 직선 전속 |

차량 기하(축거 0.33m, 전폭 0.30m, 전장 0.60m, 라이다 위치 0.31m,
조향 ±40°)는 실측값이다.
