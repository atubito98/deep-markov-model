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
Independent race model (IRM) as a two-dimensional Markov process: the "race in a box".

  g  Go accumulator, drifts after the Go cue;  g reaching 1 first -> movement (RT = crossing time)
  s  Stop accumulator, drifts after the SSD;   s reaching 1 first -> successful stop
  The two accumulators never interact.

Time step DT = 10 ms, Go cue at bin GO.

Main functions
  latent_paths(...)    simulate trials from given random draws (for paired comparisons)
  simulate_trials(...) simulate a set of trials with random Stop signals
  oracle_ssrt(...)     per-trial SSRT and trial-averaged simulated RT, from the true dynamics
"""
import numpy as np

DT, T, GO = 10, 128, 28          # ms per bin, bins per trial, bin of the Go cue
TH, SIG, SIG_S = 1.0, 0.012, 0.010
V_MU, VS_MU = 1 / 500, 1 / 200
SQ = np.sqrt(DT)
POST_DRIFT = 3 / 500


def latent_paths(v, vs, eg, es, ssd_bin, post="continue"):
    """Vectorized race over trials. v, vs (N,) Go and Stop drifts; eg, es (N,T) standard normals
    (indexed by absolute bin, so reusing them gives paired trials); ssd_bin (N,) Stop onset in bins
    after the Go cue, -1 = no Stop. Returns Z (N,T,2), RT in ms (nan if no movement) and outcome
    (0 unfinished, 1 movement, 2 stop). After the decision the state either keeps moving along the
    winning axis (post="continue") or freezes (post="freeze"); this does not affect outcomes or RTs."""
    N = len(v)
    g, s = np.zeros(N), np.zeros(N)
    out, rt = np.zeros(N, int), np.full(N, np.nan)
    Z = np.zeros((N, T, 2))
    for t in range(T):
        if post == "continue" and t > 0:
            mvd, std = out == 1, out == 2
            g = np.where(mvd, g + POST_DRIFT * DT + SIG * SQ * eg[:, t], g)
            s = np.where(std, s + POST_DRIFT * DT + SIG_S * SQ * es[:, t], s)
        run = (out == 0) & (t >= GO)
        stop_on = (ssd_bin >= 0) & (t >= GO + ssd_bin)
        g = np.where(run, g + v * DT + SIG * SQ * eg[:, t], g)
        s = np.where(run & stop_on, s + vs * DT + SIG_S * SQ * es[:, t], s)
        st = run & (s >= TH)
        mv = run & ~st & (g >= TH)
        out[st], out[mv], rt[mv] = 2, 1, (t - GO + 1) * DT
        Z[:, t] = np.stack([g, s], 1)
    return Z, rt, out


def simulate_trials(N=3000, p_stop=0.25, seed=0, ssd_range=(150, 450), post="freeze"):
    """N trials, a fraction p_stop with a Stop signal at an SSD drawn uniformly in ssd_range (ms).
    As in the task, a scheduled Stop trial in which the movement occurred before the SSD is
    recorded as a no-stop trial."""
    rng = np.random.default_rng(seed)
    rng.integers(2, size=N)                                  # reach direction (unused here)
    is_stop = rng.random(N) < p_stop
    ssd_bin = np.where(is_stop, rng.integers(int(ssd_range[0]) // DT, int(ssd_range[1]) // DT + 1, N), -1)
    v, vs = np.full(N, V_MU), np.full(N, VS_MU)
    rng.standard_normal(2 * N)                               # kept for reproducibility of the random stream
    eg, es = rng.standard_normal((N, T)), rng.standard_normal((N, T))
    Z, rt, out = latent_paths(v, vs, eg, es, ssd_bin, post=post)
    early = is_stop & (out == 1) & (rt <= ssd_bin * DT)
    is_stop = is_stop & ~early
    return dict(Z=Z, rt=rt, outcome=out, is_stop=is_stop,
                ssd=np.where(is_stop, ssd_bin * DT, np.nan), v=v, vs=vs)


def oracle_ssrt(data, n_trials=300, K=200, max_back=60, seed=2):
    """Per-trial SSRT with the true dynamics, as described in the Methods: the latest virtual Stop
    time SSD* at which p(move) < 0.5, and SSRT = RT - SSD*. Returns (RT, SSRT, trial-averaged
    simulated RT), the last being the mean RT of Go-only rollouts from the state at SSD*."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(~data["is_stop"] & (data["outcome"] == 1))[:n_trials]
    H = 120                                                 # rollout horizon (bins)
    first = lambda x: np.where((x >= TH).any(-1), (x >= TH).argmax(-1), H)
    rts, ssrts, rtsims = [], [], []
    for i in idx:
        tc = GO + int(data["rt"][i] // DT) - 1               # bin where g crossed the threshold
        v = data["v"][i]
        b = tc - np.arange(1, max_back + 1)
        b = b[b >= GO]
        if len(b) == 0:
            continue
        g0 = data["Z"][i, b, 0][:, None, None]
        go = g0 + np.cumsum(v * DT + SIG * SQ * rng.standard_normal((len(b), K, H)), -1)
        rng.standard_normal(len(b) * K)                      # kept for reproducibility of the random stream
        st = np.cumsum(VS_MU * DT + SIG_S * SQ * rng.standard_normal((len(b), K, H)), -1)
        tg, ts = first(go), first(st)
        pmove = (tg < ts).mean(1)
        below = np.flatnonzero(pmove < 0.5)
        if len(below) == 0:
            continue
        j = below[0]                                         # latest virtual Stop time with p(move) < 0.5
        rts.append(data["rt"][i]); ssrts.append((tc + 1 - b[j]) * DT)
        rtsims.append((b[j] - GO + 1 + tg[j].mean()) * DT)
    return np.array(rts), np.array(ssrts), np.array(rtsims)
