# train — 학습

모방학습으로 운전을 가르친 뒤 강화학습으로 랩타임을 줄이는 두 단계입니다.

| 파일 | 내용 |
|---|---|
| `train_imitation.py` | 1단계. 레이싱 라인 교사(최소곡률 라인 + Stanley)를 따라 하게 하는 모방학습(DAgger) |
| `train_rl.py` | 2단계. 모방학습 결과에서 시작하는 강화학습(PPO). 검증 점수가 오를 때마다 가중치 저장 |

```powershell
.\.venv\Scripts\python.exe train\train_imitation.py --out T_fly871.pt --motion 6
.\.venv\Scripts\python.exe train\train_rl.py --iters 3000 --init AB_mid.pt --motion 6 --out T_fly871.pt
```

`--motion`은 모델마다 다릅니다(T안 6, R·B·C안 3).
