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
Wall models of action stopping: two-dimensional Markov models in which Go and Stop share a single
decision boundary (the "wall").

Latent state z = (z1, z2), time step DT = 10 ms, Go cue at bin GO.

  Go context      drift v along z1 (trial-specific, v ~ N(v_mu, v_sd^2)) and noise on both axes.
                  Each trial starts from (z1_0, z2_0), with z1_0 ~ N(0, z1_sd0^2) and z2_0 ~ N(0, z2_sd0^2).
  Stop context    adds a field of magnitude vs along +z2 (rotated by stop_angle towards +z1).
  Wall            z1 = 1 - tan(tilt) * z2. tilt = 0: plain wall; tilt > 0: tilted wall, closer to
                  the origin on the Stop side.
  Exits           a trajectory reaching the wall with z2 > sep is a successful stop, otherwise a
                  movement (RT = arrival time). The stop exit exists only after a Stop signal.
  Pause region    under the Stop context, for pause_lo <= z1 < pause_z1, the Go drift is multiplied
                  by pause_gain and the Stop field by pause_gain_z2.
  Pause period    under the Stop context, for the first pause_ms after Stop onset, the Go drift is
                  multiplied by pause_t_gain (and the Stop field too if pause_t_both is True).

Main functions
  latent_trials(...)   simulate trials from given random draws (for paired comparisons)
  simulate_trials(...) simulate a set of trials with random Stop signals
  crn_truth(...)       paired Delta-RT (same trial and noise, Stop on vs off) and P(stop failure)
  oracle_ssrt(...)     per-trial SSRT and trial-averaged simulated RT, from the true dynamics
"""
import numpy as np

DT, T, GO = 10, 128, 28          # ms per bin, bins per trial, bin of the Go cue
TH = 1.0                          # position of the wall along z1 (for z2 = 0)
SQ = np.sqrt(DT)

DEFAULTS = dict(v_mu=1 / 500, v_sd=0.0, sig1=0.012, sig2=0.004, z1_sd0=0.0, z2_sd0=0.12,
                vs=1 / 250, stop_angle=0.0, tilt=0.0, sep=0.45,
                pause_gain=1.0, pause_lo=0.0, pause_z1=0.5, pause_gain_z2=1.0,
                pause_ms=0.0, pause_t_gain=1.0, pause_t_both=False,
                post="continue", post_gain=2.0, k_ret=0.05)

# Parameters used in the paper: most trial-to-trial variability is between trials (initial z1 and
# trial-specific Go drift) rather than step-by-step noise.
REALISTIC = dict(sig1=0.006, z1_sd0=0.15, v_sd=0.0004)


def P_(params):
    """Full parameter set: DEFAULTS updated with params."""
    p = dict(DEFAULTS)
    p.update(params or {})
    return p


def in_pause_region(z1, p):
    """1 inside the pause region along z1, 0 outside (always 0 if no pause region is set)."""
    if p["pause_gain"] == 1.0 and p["pause_gain_z2"] == 1.0:
        return np.zeros_like(z1, dtype=float)
    return ((z1 >= p["pause_lo"]) & (z1 < p["pause_z1"])).astype(float)


def run_paths(z0, v, e1, e2, go_on, stop_on, p):
    """Vectorized dynamics from arbitrary start states.
    z0 (n,2) start states; v (n,) Go drifts; e1, e2 (n,S) standard normals; go_on, stop_on (n,S)
    boolean context masks. Returns Z (n,S,2), outcome (0 running, 1 movement, 2 stop) and the
    step index of the wall arrival (-1 if none).
    After the decision the state either keeps moving along the winning direction (post="continue")
    or stays at the wall (movement) / relaxes towards the start (stop) (post="freeze"); this does
    not affect outcomes or RTs."""
    n, S = e1.shape
    z1, z2 = z0[:, 0].astype(float).copy(), z0[:, 1].astype(float).copy()
    out, hit = np.zeros(n, int), np.full(n, -1)
    Z = np.zeros((n, S, 2))
    ts = np.zeros(n)                                        # steps since Stop onset
    a, tl = np.deg2rad(p["stop_angle"]), np.tan(np.deg2rad(p["tilt"]))
    for k in range(S):
        if p["post"] == "continue":
            mvd, std = (out == 1) & go_on[:, k], (out == 2) & go_on[:, k]
            z1 = np.where(mvd, z1 + p["post_gain"] * v * DT + p["sig1"] * SQ * e1[:, k], z1)
            z2 = np.where(std, z2 + p["post_gain"] * p["vs"] * DT + p["sig2"] * SQ * e2[:, k], z2)
        run = out == 0
        g, s = run & go_on[:, k], run & stop_on[:, k]
        region = s * in_pause_region(z1, p)
        gain = 1.0 - (1.0 - p["pause_gain"]) * region
        gain2 = 1.0 - (1.0 - p["pause_gain_z2"]) * region
        ts = ts + s
        tfac = np.where(s & (ts * DT <= p["pause_ms"]), p["pause_t_gain"], 1.0)
        gain = gain * tfac
        if p["pause_t_both"]:
            gain2 = gain2 * tfac
        z1 = np.where(g, z1 + v * DT * gain + s * gain2 * p["vs"] * DT * np.sin(a) + p["sig1"] * SQ * e1[:, k], z1)
        z2 = np.where(g, z2 + s * gain2 * p["vs"] * DT * np.cos(a) + p["sig2"] * SQ * e2[:, k], z2)
        at_wall = run & g & (z1 >= TH - tl * z2)
        is_stop = (z2 > p["sep"]) & s                       # the stop exit exists only under Stop
        st, mv = at_wall & is_stop, at_wall & ~is_stop
        out[st], out[mv], hit[st | mv] = 2, 1, k
        if p["post"] == "freeze":
            back = out == 2
            z1 = np.where(back & ~st, z1 * (1 - p["k_ret"]), z1)
            z2 = np.where(back & ~st, z2 * (1 - p["k_ret"]), z2)
        Z[:, k, 0], Z[:, k, 1] = z1, z2
    return Z, out, hit


def _draw(rng, n, p):
    """Random draws for n trials: Go drifts, start states, a uniform per trial, and the noise."""
    v = p["v_mu"] + p["v_sd"] * rng.standard_normal(n)
    z2_0 = p["z2_sd0"] * rng.standard_normal(n)
    z0 = np.stack([p["z1_sd0"] * rng.standard_normal(n), z2_0], 1)
    return v, z0, rng.random(n), rng.standard_normal((n, T)), rng.standard_normal((n, T))


def latent_trials(v, z0, u, e1, e2, ssd_bin, p):
    """Whole trials from given draws. z0 (n,2) start states; ssd_bin (n,) Stop onset in bins after
    the Go cue, -1 = no Stop. u is not used by the dynamics (kept so draws can be reused as given
    by _draw). Returns Z (n,T,2), RT in ms (nan if no movement) and outcome."""
    n = len(v)
    t = np.arange(T)[None]
    go_on = np.broadcast_to(t >= GO, (n, T))
    stop_on = (ssd_bin[:, None] >= 0) & (t >= GO + ssd_bin[:, None])
    Z, out, hit = run_paths(np.asarray(z0, float), v, e1, e2, go_on, stop_on, p)
    rt = np.where(out == 1, (hit - GO + 1) * DT, np.nan)
    return Z, rt, out


def simulate_trials(N=3000, p_stop=0.25, seed=0, params=None, ssd_range=(150, 450)):
    """N trials, a fraction p_stop with a Stop signal at an SSD drawn uniformly in ssd_range (ms).
    As in the task, a scheduled Stop trial in which the movement occurred before the SSD is
    recorded as a no-stop trial."""
    p = P_(params)
    rng = np.random.default_rng(seed)
    rng.integers(2, size=N)                                  # reach direction (unused here)
    is_stop = rng.random(N) < p_stop
    ssd_bin = np.where(is_stop, rng.integers(int(ssd_range[0]) // DT, int(ssd_range[1]) // DT + 1, N), -1)
    v, z0, u, e1, e2 = _draw(rng, N, p)
    Z, rt, out = latent_trials(v, z0, u, e1, e2, ssd_bin, p)
    early = is_stop & (out == 1) & (rt <= ssd_bin * DT)
    is_stop = is_stop & ~early
    return dict(Z=Z, rt=rt, outcome=out, is_stop=is_stop,
                ssd=np.where(is_stop, ssd_bin * DT, np.nan), v=v, params=p)


def crn_truth(ssds_ms, params=None, M=20000, seed=1, return_pfail=False):
    """Paired Delta-RT at each SSD (ms): each trial is run twice with the same draws, without and
    with a Stop signal; Delta-RT = RT(stop failure) - RT(same trial without Stop), averaged over
    stop failures whose Go twin had not moved before the SSD. With return_pfail, also returns
    P(stop failure | not yet moved at the SSD)."""
    p = P_(params)
    rng = np.random.default_rng(seed)
    res, pf = {}, {}
    for ssd in ssds_ms:
        v, z0, u, e1, e2 = _draw(rng, M, p)
        _, rt_go, _ = latent_trials(v, z0, u, e1, e2, np.full(M, -1), p)
        _, rt_st, o_st = latent_trials(v, z0, u, e1, e2, np.full(M, ssd // DT), p)
        f = (o_st == 1) & np.isfinite(rt_go) & (rt_go > ssd)
        res[ssd] = round(float(np.mean(rt_st[f] - rt_go[f])), 1) if f.any() else float("nan")
        alive = ~(np.isfinite(rt_go) & (rt_go <= ssd))
        pf[ssd] = round(float(np.mean(o_st[alive] == 1)), 3)
    return (res, pf) if return_pfail else res


def _pmove_curve(data, i, p, rng, K, max_back):
    """p(move) for a virtual Stop signal injected 1..max_back bins before trial i's movement bin,
    and the RTs of Go-context rollouts from the same states."""
    tc = GO + int(data["rt"][i] // DT) - 1
    b = tc - np.arange(1, max_back + 1)
    b = b[b >= GO]
    if len(b) == 0:
        return tc, b, None, None
    nb, S = len(b), T - 1 - b.min()
    z0 = np.repeat(data["Z"][i, b], K, 0)
    v = np.full(nb * K, data["v"][i])
    rng.random(nb * K)                                       # kept for reproducibility of the random stream
    e1, e2 = rng.standard_normal((nb * K, S)), rng.standard_normal((nb * K, S))
    on = np.arange(S)[None] < np.repeat(T - 1 - b, K)[:, None]
    _, out, _ = run_paths(z0, v, e1, e2, on, on, p)
    pm = (out == 1).reshape(nb, K).mean(1)
    _, outg, hitg = run_paths(z0, v, e1, e2, on, np.zeros_like(on), p)
    rt_go = np.where(outg == 1, (np.repeat(b, K) + 1 + hitg - GO + 1) * DT, np.nan).reshape(nb, K)
    return tc, b, pm, rt_go


def oracle_ssrt(data, n_trials=300, K=200, max_back=60, seed=2):
    """Per-trial SSRT with the true dynamics, as described in the Methods: the latest virtual Stop
    time SSD* at which p(move) < 0.5, and SSRT = RT - SSD*. Returns (RT, SSRT, trial-averaged
    simulated RT), the last being the mean RT of Go-context rollouts from the state at SSD*."""
    p, rng = data["params"], np.random.default_rng(seed)
    idx = np.flatnonzero(~data["is_stop"] & (data["outcome"] == 1))[:n_trials]
    rts, ssrts, rtsims = [], [], []
    for i in idx:
        tc, b, pm, rt_go = _pmove_curve(data, i, p, rng, K, max_back)
        if pm is None:
            continue
        below = np.flatnonzero(pm < 0.5)
        if len(below) == 0:
            continue
        j = below[0]
        rts.append(data["rt"][i]); ssrts.append((tc + 1 - b[j]) * DT)
        rtsims.append(np.nanmean(rt_go[j]))
    return np.array(rts), np.array(ssrts), np.array(rtsims)
