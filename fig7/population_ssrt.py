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
Stop-signal analyses on the toy models (race in a box and wall models), with their true dynamics.

analyse(kind, params) returns, for SSDs of 150-450 ms:
  - paired Delta-RT (same trial and noise, Stop on vs off)
  - unpaired Delta-RT (stop failures among fresh continuations after the SSD, compared with the
    trial's own RT)
  - P(stop failure) per SSD
  - slopes of per-trial SSRT against empirical RT and against trial-averaged simulated RT
  - population SSRT: from the mean per-trial inhibition curve (mechanistic) and from the
    integration method
"""
import time
import numpy as np
import race_model as race
import wall_model as wall

DT, T, GO = 10, 128, 28
SSDS = [150, 200, 250, 300, 350, 400, 450]


def binned_slope(x, y, nb=9):
    """Slope of a linear fit across nb equal-sized bins of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 * nb:
        return np.nan, np.full(nb, np.nan), np.full(nb, np.nan)
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    k = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
    mx = np.array([x[k == j].mean() for j in range(nb)])
    my = np.array([y[k == j].mean() for j in range(nb)])
    return np.polyfit(mx, my, 1)[0], mx, my


def make_sim(kind, prm):
    """Functions to draw trials and run them, for kind "box" (race model) or "wall"."""
    p = wall.P_(dict(prm or {}, post="freeze"))

    def draw(rng, n):
        if kind == "box":
            return dict(v=np.full(n, race.V_MU), vs=np.full(n, race.VS_MU),
                        e1=rng.standard_normal((n, T)), e2=rng.standard_normal((n, T)))
        z0 = np.stack([p["z1_sd0"] * rng.standard_normal(n), p["z2_sd0"] * rng.standard_normal(n)], 1)
        return dict(v=p["v_mu"] + p["v_sd"] * rng.standard_normal(n), z0=z0, u=rng.random(n),
                    e1=rng.standard_normal((n, T)), e2=rng.standard_normal((n, T)))

    def run(d, ssd_bin):
        if kind == "box":
            _, rt, out = race.latent_paths(d["v"], d["vs"], d["e1"], d["e2"], ssd_bin, post="freeze")
        else:
            _, rt, out = wall.latent_trials(d["v"], d["z0"], d["u"], d["e1"], d["e2"], ssd_bin, p)
        return rt, out
    return draw, run


def take(d, idx):
    return {k: v[idx] for k, v in d.items()}


def splice(d_emp, rng, K, cut_bin):
    """K continuations of each trial: noise copied up to cut_bin (exclusive), fresh afterwards."""
    n = len(d_emp["v"])
    rep = {k: np.repeat(v, K, 0) for k, v in d_emp.items()}
    cut = np.repeat(np.atleast_1d(cut_bin), K) if np.ndim(cut_bin) else np.full(n * K, cut_bin)
    after = np.arange(T)[None] >= cut[:, None]
    for k in ("e1", "e2"):
        rep[k] = np.where(after, rng.standard_normal(rep[k].shape), rep[k])
    return rep


def analyse(name, kind, prm, n_trials=400, K=100, n_ssrt=150, K2=40, max_back=40, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    draw, run = make_sim(kind, prm)
    res = dict(paired={}, unpaired={}, pfail={})
    # paired Delta-RT and P(stop failure): the same draws without and with a Stop signal
    for ssd in SSDS:
        d = draw(rng, 20000)
        rg, _ = run(d, np.full(20000, -1))
        rs, os_ = run(d, np.full(20000, ssd // DT))
        f = (os_ == 1) & np.isfinite(rg) & (rg > ssd)
        res["paired"][ssd] = float(np.mean(rs[f] - rg[f]))
        res["pfail"][ssd] = float(np.mean(os_ == 1))
    rt_go_dist, _ = run(draw(rng, 20000), np.full(20000, -1))
    rt_go_dist = rt_go_dist[np.isfinite(rt_go_dist)]
    # unpaired Delta-RT: stop failures among fresh continuations vs the trial's own RT
    emp = draw(rng, n_trials)
    rt_emp, _ = run(emp, np.full(n_trials, -1))
    for ssd in SSDS:
        ok = np.flatnonzero(np.isfinite(rt_emp) & (rt_emp > ssd))
        cont = splice(take(emp, ok), rng, K, GO + ssd // DT)
        rc, oc = run(cont, np.full(len(ok) * K, ssd // DT))
        rc, oc = rc.reshape(len(ok), K), oc.reshape(len(ok), K)
        vals = [rc[i, oc[i] == 1].mean() - rt_emp[ok[i]] for i in range(len(ok)) if (oc[i] == 1).sum() >= 5]
        res["unpaired"][ssd] = float(np.mean(vals)) if vals else float("nan")
    # per-trial inhibition curves, per-trial SSRT, and the population inhibition curve
    curves, rts, ssrts, rtsims = [], [], [], []
    for i in np.flatnonzero(np.isfinite(rt_emp))[:n_ssrt]:
        tc = GO + int(rt_emp[i] // DT) - 1                  # movement bin
        js = np.arange(1, max_back + 1)
        js = js[tc - js >= GO - 1]
        if len(js) == 0:
            continue
        ssd_bins = (tc - js + 1) - GO                         # Stop from bin tc-j+1: SSRT = j*DT
        base = take(emp, np.full(len(js), i))
        cont = splice(base, rng, K2, ssd_bins + GO)
        rc, oc = run(cont, np.repeat(ssd_bins, K2))
        pm = (oc == 1).reshape(len(js), K2).mean(1)
        curve = np.full(max_back, np.nan); curve[js - 1] = pm; curves.append(curve)
        below = np.flatnonzero(pm < 0.5)
        if len(below) == 0:
            continue
        j = js[below[0]]
        cg = splice(take(emp, [i]), rng, 30, ssd_bins[below[0]] + GO)
        rg, _ = run(cg, np.full(30, -1))
        rts.append(rt_emp[i]); ssrts.append(j * DT); rtsims.append(np.nanmean(rg))
    rts, ssrts, rtsims = map(np.array, (rts, ssrts, rtsims))
    res["ssrt_slope_vs_empirical_rt"] = float(binned_slope(rts, ssrts)[0])
    res["ssrt_slope_vs_simulated_rt"] = float(binned_slope(rtsims, ssrts)[0])
    mean_curve = np.nanmean(np.array(curves), 0)
    below = np.flatnonzero(mean_curve < 0.5)
    res["ssrt_mechanistic"] = float((below[0] + 1) * DT) if len(below) else float("nan")
    integ = {s: float(np.quantile(rt_go_dist, res["pfail"][s]) - s) for s in SSDS if 0.1 <= res["pfail"][s] <= 0.9}
    res["ssrt_integration_by_ssd"] = integ
    res["ssrt_integration"] = float(np.mean(list(integ.values()))) if integ else float("nan")
    res["rt_go_mean"] = float(rt_go_dist.mean())
    print(f"  {name}: population SSRT {res['ssrt_mechanistic']:.0f} ms (mechanistic), "
          f"{res['ssrt_integration']:.0f} ms (integration)  [{time.time() - t0:.0f} s]", flush=True)
    return res
