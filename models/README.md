# models — 모델 성적표

평가 스크립트(`evaluate/`)가 남긴 결과 JSON입니다. 학습된 가중치(`*.pt`)는 용량 때문에 저장소에 넣지 않았습니다.

**모델 이름**

| 이름 | 뜻 |
|---|---|
| `C_fly871` | C안. 초파리 배선, 뉴런 871개 |
| `B_dense871` | B안. 뉴런 871개를 전부 연결한 대조군 |
| `R_fly871` | R안. 초파리 배선 + 레이싱 교사 |
| `T_fly871_final` | T안. 최종 모델 (실차 주행) |
| `T_cand_itNNNN` | T안 학습 중 NNNN회차 후보 |
| `grid_<섭동>_<할인율>_s<시드>` | 학습 설정 격자 실험 (섭동 none/mid/high, 할인율 0.99/0.998) |

**파일 앞머리**

| 앞머리 | 만든 스크립트 | 내용 |
|---|---|---|
| (없음) | `evaluate.py` | 기본 45트랙 성적 |
| `dynamic_` | `experiment_dynamic.py` | 움직이는 상대 차 |
| `obstacles_` | `experiment_obstacles.py` | 정지 장애물 |
| `robust_` | `experiment_robust_completion.py` | 섭동 반복 완주율 |
| `pick_` | `grid_pick.py` | 설정 선정용 넓은 재측정 |
| `race_` | `grid_race.py` | 대회 코스 반복 측정 |
| `racerobust_` | `race_robust.py` | 대회 코스 노면 × 섭동 21판 |

---
[← 전체 구성으로](../README.md)
