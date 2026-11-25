# Copyright 2026 Istituto Superiore di Sanità (ISS)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

"""
Step 1 of 2: computes all quantities shown in Fig. 7 (and in Figs. S8 and S12) from the true
dynamics of the toy models, and writes them to fig7_results.json.

Run:  python compute_fig7.py        (about 10-15 minutes on a laptop CPU)
"""
import json, time
import numpy as np
import race_model as race
import wall_model as wall
import population_ssrt as pop

SSDS = [150, 200, 250, 300, 350, 400, 450]
MODELS = {
    "plain": dict(wall.REALISTIC),
    "tilted": dict(wall.REALISTIC, tilt=25.0),
    "pause": dict(wall.REALISTIC, tilt=25.0, pause_lo=0.3, pause_z1=0.7, pause_gain=0.2, pause_gain_z2=0.2),
}
PAUSE_PERIOD = dict(wall.REALISTIC, tilt=25.0, pause_ms=150.0, pause_t_gain=0.2, pause_t_both=True)


def slope_with_se(x, y, B=500, seed=0):
    """Trial-level regression slope and its bootstrap standard error."""
    rng = np.random.default_rng(seed); n = len(x)
    boot = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.polyfit(x, y, 1)[0]), float(np.std(boot))


results = {}
t_start = time.time()

# Wall models (Fig. 7B-E)
for name, prm in MODELS.items():
    print(f"{name} wall ...", flush=True)
    r = {"paired_drt": wall.crn_truth(SSDS, params=prm, M=15000)}
    trials = wall.simulate_trials(N=2500, params=prm, seed=11)
    rt, ssrt, rt_sim = wall.oracle_ssrt(trials, n_trials=700, K=150)
    ok = np.isfinite(rt_sim)
    r["coupling_slope"] = slope_with_se(rt_sim[ok], ssrt[ok])          # SSRT vs simulated RT (Fig. 7D)
    r["slope_vs_empirical_rt"] = slope_with_se(rt, ssrt)
    a = pop.analyse(name, "wall", prm, n_trials=200, K=60, n_ssrt=150, K2=40)
    r["ssrt_mechanistic"], r["ssrt_integration"] = a["ssrt_mechanistic"], a["ssrt_integration"]
    r["pfail"] = a["pfail"]
    results[name] = r

# Race in a box (independent race): reference values for Fig. 7D-E and the per-trial data of Fig. S8
print("race in a box ...", flush=True)
trials = race.simulate_trials(N=2500, seed=11)
rt, ssrt, rt_sim = race.oracle_ssrt(trials, n_trials=700, K=150)
a = pop.analyse("box", "box", None, n_trials=200, K=60, n_ssrt=150, K2=40)
results["box"] = dict(paired_drt={s: 0.0 for s in SSDS},
                      coupling_slope=slope_with_se(rt_sim, ssrt), slope_vs_empirical_rt=slope_with_se(rt, ssrt),
                      ssrt_mechanistic=a["ssrt_mechanistic"], ssrt_integration=a["ssrt_integration"],
                      per_trial=dict(rt=rt.tolist(), ssrt=ssrt.tolist(), rt_sim=rt_sim.tolist()))

# Tilted wall with a pause period instead of a pause region (Fig. S12)
print("pause period ...", flush=True)
results["pause_period"] = {"paired_drt": wall.crn_truth(SSDS, params=PAUSE_PERIOD, M=15000)}

json.dump(results, open("fig7_results.json", "w"), indent=1, default=float)

print("\nSummary")
for k in ["box", "plain", "tilted", "pause"]:
    r = results[k]
    print(f"  {k:7s} paired dRT (ms) {[round(r['paired_drt'][s]) for s in SSDS]}  "
          f"coupling slope {r['coupling_slope'][0]:+.2f} +- {r['coupling_slope'][1]:.2f}  "
          f"SSRT {r['ssrt_mechanistic']:.0f} / {r['ssrt_integration']:.0f} ms (mechanistic / integration)")
print(f"  pause period paired dRT (ms) {[round(results['pause_period']['paired_drt'][s]) for s in SSDS]}")
print(f"\nwrote fig7_results.json  ({(time.time() - t_start) / 60:.1f} min)")
