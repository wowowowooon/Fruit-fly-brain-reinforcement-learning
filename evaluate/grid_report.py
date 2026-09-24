"""격자 결과를 모아 표로 찍고, 미리 정한 규칙대로 설정 하나를 고른다.

고르는 규칙 (측정 전에 정해 둔 것)
  필수   2026 대회 코스를 노면 7종에서 완주할 것
  1순위  일반 45트랙 완주 (시드 3개 평균)
  동점   1.5판 이내면 시드 간 편차가 작은 쪽
  그래도 랩타임이 짧은 쪽
  떨림은 고르는 데 쓰지 않는다. 참고로만 찍는다.

원래는 '학습 중 검증 최고' 로 고르기로 했다. 그런데 검증셋이 12트랙이라
한 칸이 8.3%p 고 여섯 설정이 94.7~100% 안에 다 들어와 아무것도 안 갈렸다.
게다가 '최고' 는 8번 측정 중 최대값이라 운 좋은 한 번이 박제된다.
그래서 같은 구조(완주 -> 편차 -> 속도)를 유지한 채 재는 세트만 넓혔다.

'최고' 와 '마지막' 을 둘 다 찍는다. '최고' 는 좁은 검증셋의 복권 당첨자라
부풀어 있고, '마지막' 은 고르는 과정이 없어 편향이 없다. 설정 자체를
판단할 때는 '마지막' 이 정직하다.

  python experiments/grid_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import statistics as st

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CAPS = ("none", "mid", "high")
CAPS_NAME = {"none": "CAPS 없음", "mid": "CAPS 보통", "high": "CAPS 강함"}
GAMMAS = ("g99", "g998")
GAM_NAME = {"g99": "감마 0.99", "g998": "감마 0.998"}
SEEDS = (0, 1, 2)


def load(stem):
    pick = MODELS / f"pick_grid_{stem}.json"
    race = MODELS / f"race_grid_{stem}.json"
    if not pick.exists():
        return None
    row = json.loads(pick.read_text(encoding="utf-8"))
    if race.exists():
        r = json.loads(race.read_text(encoding="utf-8"))
        row["race_done_n"] = r["done"]
        row["race_total"] = r["total"]
        row["race_lap"] = r["lap"]
        row["race_worst"] = r["worst_lap"]
        row["race_gap"] = r["gap"]
    return row


def table(kind):
    """kind: '' 이면 최고 정책, '_last' 면 마지막 정책."""
    print()
    print(f"### {'최고 정책 (좁은 검증셋으로 뽑은 것)' if not kind else '마지막 정책 (고르지 않은 것)'}")
    print(f"{'설정':<22}{'일반/45':>11} {'랩':>7} {'지그재그':>8} {'대회/7':>8} "
          f"{'대회랩':>8} {'떨림':>7}")
    print("-" * 78)
    out = {}
    for c in CAPS:
        for g in GAMMAS:
            rows = [load(f"{c}_{g}_s{s}{kind}") for s in SEEDS]
            rows = [r for r in rows if r]
            if not rows:
                continue
            norm = [r["normal_mean"] for r in rows]
            narrow = [r["narrow_done"] for r in rows]
            lap = [r["normal_lap"] for r in rows]
            shake = [r["shake"] for r in rows]
            race_n = [r.get("race_done_n") for r in rows if "race_done_n" in r]
            race_l = [r.get("race_lap", 0.0) for r in rows if r.get("race_done_n")]
            name = f"{CAPS_NAME[c]} / {GAM_NAME[g]}"
            out[(c, g)] = {
                "n": len(rows),
                "norm": st.mean(norm), "norm_sd": st.pstdev(norm),
                "narrow": st.mean(narrow), "lap": st.mean(lap),
                "shake": st.mean(shake),
                "race": st.mean(race_n) if race_n else None,
                "race_lap": st.mean(race_l) if race_l else 0.0,
            }
            v = out[(c, g)]
            race_txt = f"{v['race']:.1f}" if v["race"] is not None else "-"
            race_lap = f"{v['race_lap']:.2f}초" if v["race_lap"] else "-"
            mark = "" if v["n"] == 3 else f"  (시드 {v['n']}개뿐)"
            print(f"{name:<22}{v['norm']:>6.1f}±{v['norm_sd']:<4.1f} "
                  f"{v['lap']:>6.1f}초 {v['narrow']:>6.1f}/12 {race_txt:>8} "
                  f"{race_lap:>8} {v['shake']:>7.3f}{mark}")
    return out


def pick(out):
    """미리 정한 규칙을 그대로 적용한다."""
    ok = {k: v for k, v in out.items()
          if v["race"] is None or v["race"] >= 4}      # 노면 7종 중 과반
    if not ok:
        print("\n대회 코스를 과반 노면에서 완주한 설정이 없다. 필수 조건 미달.")
        ok = out
    top = max(v["norm"] for v in ok.values())
    tied = {k: v for k, v in ok.items() if top - v["norm"] <= 1.5}
    if len(tied) > 1:
        best_sd = min(v["norm_sd"] for v in tied.values())
        tied = {k: v for k, v in tied.items() if v["norm_sd"] <= best_sd + 0.01}
    if len(tied) > 1:
        tied = {min(tied, key=lambda k: tied[k]["lap"]): None}
    return list(tied)[0]


def main():
    done = len(list(MODELS.glob("pick_grid_*.json")))
    print("=" * 78)
    print(f"격자 18판 넓은 재측정   ({done}/36 모델)")
    print("  일반 45트랙 x 섭동 2단계 = 90판 / 지그재그 12트랙 / 대회 코스 노면 7종")
    print("=" * 78)

    best = table("")
    last = table("_last")

    print()
    print("=" * 78)
    for label, out in (("최고 정책 기준", best), ("마지막 정책 기준", last)):
        if not out:
            continue
        c, g = pick(out)
        v = out[(c, g)]
        print(f"{label:<16} -> {CAPS_NAME[c]} / {GAM_NAME[g]}   "
              f"일반 {v['norm']:.1f}±{v['norm_sd']:.1f}  랩 {v['lap']:.1f}초")
    print("=" * 78)

    # R안 기준선
    r = MODELS / "race_R_fly871_v1_keep.json"
    if r.exists():
        d = json.loads(r.read_text(encoding="utf-8"))
        bad = [f"{x['grip']:.2f}" for x in d["runs"] if not x["done"]]
        print(f"\n참고 — R안 대회 코스 {d['done']}/{d['total']} 완주, "
              f"평균 {d['lap']:.2f}초, 최소여유 {d['gap']:+.3f}m"
              + (f", 실패 노면 {','.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
