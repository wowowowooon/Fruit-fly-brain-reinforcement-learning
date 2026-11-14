# evaluate — 성적 측정과 실험

모델을 같은 시드, 같은 조건으로 재서 비교합니다. 평가 시드(700~744)는 학습 시드(0~399)와 겹치지 않습니다.

**기본 측정**

| 파일 | 내용 |
|---|---|
| `evaluate.py` | 처음 보는 트랙 45개 완주율·랩타임 |
| `experiment_robust_completion.py` | 아주 작은 섭동을 줘 가며 여러 번 재는 완주율 (한 번 잰 숫자는 믿지 않음) |
| `run_battery.py` | 모델 하나를 아래 모든 잣대로 한꺼번에 측정 |

**강인성 실험**

| 파일 | 내용 |
|---|---|
| `experiment_faults.py` | 센서 고장(빔 손실, 사거리 감소, 잡음, 값 굳음, 시야 차단) |
| `experiment_robustness.py` | 실차에서 달라질 조건(노면 마찰, 제어 지연 등) |
| `experiment_obstacles.py` | 정지 장애물 유무 비교 |
| `experiment_dynamic.py` | 움직이는 상대 차 유무 비교 |
| `experiment_speed.py` | 한 스텝 추론 시간 |
| `experiment_neurons.py` | 주행 중 어떤 뉴런이 언제 켜지는지 기록 |

**대회 코스와 최종 모델 선정**

| 파일 | 내용 |
|---|---|
| `race_robust.py` | 2026 대회 코스, 노면 7종 × 섭동 3단계 = 21판 |
| `grid_pick.py` · `grid_race.py` · `grid_report.py` | 학습 설정 격자 18판을 넓은 세트와 대회 코스로 재고, 미리 정한 규칙으로 하나를 고름 |
| `overtake_check.py` | T안과 Y안(추월 층 추가)을 같은 잣대로 합격/불합격 판정 |

결과 성적표는 `models/*.json`에 저장됩니다.

---
[← 전체 구성으로](../README.md)
