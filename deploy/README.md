# T안을 젯슨에 올리기

`models/T_fly871_final.pt` 를 실차에서 굴리는 데 필요한 것 전부.

## 뭘로 대체하는가

기존 스택에서 주행에 쓰던 것들을 **노드 하나가 대신**한다.

| 안 쓰는 것 | 이유 |
|---|---|
| `stanley_waypoint_follow_node` | 경로를 안 쓴다 |
| `local_planner_node`, `fgm_node` | 경로를 안 쓴다 |
| `static_obstacle_node`, `integrated_obstacle_node` | 정책이 라이다에서 바로 판단 |
| `cartographer`, `amcl`, `tf_manager` | 지도도 위치추정도 안 쓴다 |
| `control_node` (1683줄) | `deploy/hardware.py` 에 필요한 부분만 |

T안은 **라이다와 IMU 만 보고 바로 조향·속도를 낸다.** 중간 단계가 없다.

## 띄우는 것

```
sllidar_node   → /scan        (기존 그대로)
ebimu_driver   → /imu/data    (기존 그대로)
fly_drive_node → ESP32 + VESC 시리얼 직접
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `fly_drive_node.py` | 본체. 스캔·IMU 받아 정책 돌리고 시리얼로 명령 |
| `hardware.py` | ESP32(`S:`) / VESC(duty 패킷) 통신, 속도 PI |
| `obs_bridge.py` | 실차 센서 → 모델 관측. ROS 없이 돌아가 검증이 된다 |
| `check_signs.py` | 차에서 부호 4개 확정 (30분) |
| `measure_response.py` | 서보·차체 반응 실측 → 시뮬과 비교 |
| `test_bridge.py` | 관측 변환 검증 (PC) |
| `test_hardware.py` | 시리얼 프로토콜 검증 (PC) |
| `test_loop.py` | 실차 명령 경로로 대회 코스 주행 (PC) |
| `speed_ctrl.py` | 목표 속도 -> duty. 모터 이득을 실측값으로 안다 |
| `OVERTAKE.md` | 추월 참는 층(`--patience`) 의 측정 결과와 판단 |

## PC 에서 이미 확인한 것

```
관측 변환      270도/360도/점수 제각각/무반사 10% 전부 완주
시리얼         VESC CRC·duty 패킷·응답 파싱, ESP S: 줄, RC 수동/자율
조향 환산      서보 40~140도 ↔ 앞바퀴 ±40도 왕복 일치
추론 속도      3 ms (예산 25 ms)
```

## 차에서 할 것

### 1. 모델과 코드 옮기기

```bash
# 젯슨에서
git clone <이 저장소> ~/fly_racer
scp <PC>:models/T_fly871_final.pt ~/fly_racer/models/
pip3 install pyserial            # torch 는 JetPack 에 맞는 NVIDIA 휠로
```

### 2. 부호 4개 확정 (바퀴 띄우고)

```bash
python3 deploy/check_signs.py --steer
```
찍혀 나온 값을 `fly_drive_node.py` 의 `CHECK` 에 넣는다.

### 3. 추론 속도 확인

젯슨에서 25 ms 안에 끝나야 한다. 모자라면 `--beams 270` (랩 0.03초 손해).

### 4. 첫 주행

```bash
ros2 run sllidar_ros2 sllidar_node --ros-args --params-file <sensor_params.yaml>
ros2 run ebimu_pkg ebimu_driver --ros-args --params-file <sensor_params.yaml>
python3 deploy/fly_drive_node.py --speed-scale 0.3
```

**CH5 를 자율로 올려야 움직인다.** 내리면 즉시 사람이 가져간다.

### 4-1. 상대차와 같이 달릴 때 (선택)

```bash
python3 deploy/fly_drive_node.py --patience
```

앞차를 지나갈 자리가 없으면 속도를 묶어 뒤따른다. 조향은 건드리지 않는다.

**기본은 꺼져 있다.** 좁은 코스에서 상대차와 달릴 때 박는 횟수가 약
4분의 1 줄지만 랩타임을 10% 쯤 쓴다. **상대차가 없으면 순손해다** —
벽만 있는 코스에서도 12.5% 느려지고 얻는 것이 없다.

자세한 측정과 판단 근거는 `OVERTAKE.md`.

### 5. 서보 반응 실측 (재학습 여부 판단)

넓은 데서 1~2 m/s 로 조향을 좌우로 꺾었다 풀었다 30초.

```bash
python3 deploy/measure_response.py <telemetry.csv>
```

시뮬(기울기 제한)과 실차(지수 지연)의 차이가 **5도를 넘으면 재학습**하는 편이 낫다.

## 아직 못 푼 것

**서보·차체 반응 실측.** 기존 텔레메트리 로그 24개가 전부 `yaw_rate_radps`,
`lateral_accel_mps2` 가 비어 있다(IMU 가 안 붙었거나 로깅이 꺼져 있었다).
차에서 30초만 다시 재면 `measure_response.py` 가 바로 답을 낸다.

## 알아 둘 것 — 기존 스택의 조향 설정

`control_node.py` 의 `max_steering_angle_rad = 0.3735` (21.4°) 는 실측과 다르다.
각도기로 재면 앞바퀴는 **±40°(0.698)** 까지 간다. 실차 로그가 증거다.

```
target_steering  최대 0.698 rad (40°)  ← 요구
final_steering   최대 1.000 (포화)     ← 0.698/0.3735 = 1.87 이 잘림
```

지금 차는 코너에서 **조향이 포화된 채로** 달리고 있다. Stanley 가 게인을
×0.428 로 재환산해 우연히 상쇄하고 있었다.

우리 노드는 `control_node` 를 안 거치고 ESP 에 직접 `S` 를 보내므로
이 문제가 없다. `S=a[0]` 이 그대로 앞바퀴 풀락 비율이다.
