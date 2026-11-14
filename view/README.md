# view — 눈으로 보기

| 파일 | 내용 |
|---|---|
| `viewer.py` | 주행을 실시간으로 보는 뷰어. 장애물·상대 차를 넣을 수 있음 |
| `test_on_map.py` | 실제 대회 맵에서 한 바퀴 돌리고 결과를 숫자로 출력 |
| `render_map_run.py` | 대회 맵 위에 주행 궤적을 그림으로 저장 |
| `race_avoid.py` | 장애물과 상대 차를 피하는 세 장면을 나란히 그림 |

```powershell
.\.venv\Scripts\python.exe view\viewer.py --model T_fly871_final.pt --motion 6 --map <맵.yaml> --obstacles 6 --rivals 2
```

---
[← 전체 구성으로](../README.md)
