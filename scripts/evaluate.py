"""Train on a clean baseline, then measure (a) false-positive rate on held-out clean data and
(b) detection rate / latency for injected events, each on a fresh system per seed.
Usage: python -m scripts.evaluate [n_seeds]"""
import sys

from app.bootstrap import build_system

HOLDOUT = 400
SCENARIOS = {
    "concentration (TECH -> 45% QBIT)": dict(kind="concentrate", portfolio_id="TECH", symbol="QBIT", target_weight=0.45),
    "vol spike (4x, 30 ticks)": dict(kind="vol_spike", mult=4.0, duration=30),
    "sector crash (Energy -2%/tick x 10 = -20%)": dict(kind="sector_crash", sector="Energy", drift=-0.02, duration=10),
}


def main(n_seeds=5):
    fp_flagged = fp_total = 0
    results = {k: [] for k in SCENARIOS}
    for seed in range(100, 100 + n_seeds):
        s = build_system(":memory:", model_path=None, baseline_ticks=400, seed=seed)
        t0 = s.engine.tick
        s.engine.run(HOLDOUT)
        fp_total += HOLDOUT * len(s.engine.portfolio_ids)
        fp_flagged += s.db.query("SELECT COUNT(*) c FROM risk_snapshots WHERE tick>? AND is_anomaly=1", (t0,))[0]["c"]
        for label, params in SCENARIOS.items():
            f = build_system(":memory:", model_path=None, baseline_ticks=400, seed=seed)
            f.engine.run(60)
            start = f.engine.tick
            f.engine.schedule(delay=1, **params)
            f.engine.run(40)
            rows = f.db.query("SELECT * FROM alerts WHERE tick>? ORDER BY tick", (start,))
            results[label].append(rows[0]["tick"] - start - 1 if rows else None)
    print(f"clean held-out: {fp_flagged}/{fp_total} snapshots flagged ({fp_flagged / fp_total:.3%}) over {n_seeds} seeds")
    for label, lat in results.items():
        hits = [x for x in lat if x is not None]
        print(f"{label}: detected {len(hits)}/{len(lat)}; latency ticks (5-min bars) per seed = {lat}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
