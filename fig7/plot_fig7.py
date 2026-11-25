
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
Step 2 of 2: draws Fig. 7 from fig7_results.json (written by compute_fig7.py), plus the two
related supplementary figures.

Outputs
  fig7_wall_model.png               Fig. 7 (main result)
  figS8_ssrt_estimator_null.png     Fig. S8: per-trial SSRT estimator under independence
  figS12_pause_region_vs_period.png Fig. S12: pause region versus pause period

Run:  python plot_fig7.py        (a few seconds)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import race_model as race
import wall_model as wall

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "sans-serif"})
R = json.load(open("fig7_results.json"))
GREY, BLUE, RED, ORANGE = "#8a8a8a", "#2f6fb5", "#c0392b", "#d9822b"
COL = {"plain": GREY, "tilted": BLUE, "pause": ORANGE, "box": "k"}
LAB = {"plain": "Plain wall", "tilted": "Tilted wall", "pause": "Tilted wall + pause region"}
DT, T, GO = wall.DT, wall.T, wall.GO
SSD_EXAMPLE = 250                       # Stop-signal delay of the example trajectories (ms)


def sample(kind, prm=None, ssd=SSD_EXAMPLE, n=600, seed=4):
    """n trials run twice with the same draws (no Stop / Stop at ssd), for the example trajectories."""
    r = np.random.default_rng(seed)
    if kind == "box":
        v, vs = np.full(n, race.V_MU), np.full(n, race.VS_MU)
        e1, e2 = r.standard_normal((n, T)), r.standard_normal((n, T))
        Zg, rg, og = race.latent_paths(v, vs, e1, e2, np.full(n, -1), post="freeze")
        Zs, _, os_ = race.latent_paths(v, vs, e1, e2, np.full(n, ssd // DT), post="freeze")
        return dict(Zg=Zg, rg=rg, og=og, Zs=Zs, os=os_, p=None)
    p = wall.P_(dict(prm, post="freeze"))
    v = p["v_mu"] + p["v_sd"] * r.standard_normal(n)
    z0 = np.stack([p["z1_sd0"] * r.standard_normal(n), p["z2_sd0"] * r.standard_normal(n)], 1)
    u = r.random(n); e1, e2 = r.standard_normal((n, T)), r.standard_normal((n, T))
    Zg, rg, og = wall.latent_trials(v, z0, u, e1, e2, np.full(n, -1), p)
    Zs, _, os_ = wall.latent_trials(v, z0, u, e1, e2, np.full(n, ssd // DT), p)
    return dict(Zg=Zg, rg=rg, og=og, Zs=Zs, os=os_, p=p)


def pick(S, ssd, n_succ=2, n_fail=2):
    """Trials still in play at the SSD: a few successful stops and a few stop failures, chosen
    among those with Go RTs closest to the median."""
    alive = np.isfinite(S["rg"]) & (S["rg"] > ssd + 30)
    succ = np.flatnonzero(alive & (S["os"] == 2)); fail = np.flatnonzero(alive & (S["os"] == 1))
    mid = lambda idx, k: idx[np.argsort(np.abs(S["rg"][idx] - np.median(S["rg"][idx])))[:k]] if len(idx) else idx
    return mid(succ, n_succ), mid(fail, n_fail)


def draw_wall(ax, S, ssd, inset=False, n_go=8):
    p = S["p"]; tl = np.tan(np.deg2rad(p["tilt"])); lw = 0.9 if inset else 1.6
    if p["pause_gain"] < 1:
        ax.axvspan(p["pause_lo"], p["pause_z1"], color=ORANGE, alpha=0.18, lw=0)
    for i in np.flatnonzero(S["og"] == 1)[:n_go]:
        k = int(GO + S["rg"][i] // DT - 1)
        ax.plot(S["Zg"][i, GO - 1:k + 1, 0], S["Zg"][i, GO - 1:k + 1, 1], color="#bdbdbd", lw=0.7 if inset else 0.9, zorder=1)
    succ, fail = pick(S, ssd)
    for idx, col in [(succ, BLUE), (fail, RED)]:
        for i in idx:
            Z = S["Zs"][i]
            hit = np.flatnonzero(Z[:, 0] >= 1 - tl * Z[:, 1]); k = hit[0] if len(hit) else T - 1
            ax.plot(Z[GO - 1:k + 1, 0], Z[GO - 1:k + 1, 1], color=col, lw=lw, zorder=3)
            ks = GO + ssd // DT                                   # state at Stop-signal onset
            ax.plot(Z[ks, 0], Z[ks, 1], "o", ms=2.5 if inset else 4, mfc="white", mec=col, mew=1, zorder=4)
    z2 = np.linspace(-0.7, 1.7, 50); ax.plot(1 - tl * z2, z2, color="k", lw=0.9 if inset else 1.2)
    ax.plot([1 - tl * p["sep"] - 0.05, 1 - tl * p["sep"] + 0.05], [p["sep"]] * 2, color="k", lw=2 if inset else 3)
    return tl


def draw_box(ax, S, ssd):
    for i in np.flatnonzero(S["og"] == 1)[:8]:
        k = int(GO + S["rg"][i] // DT - 1)
        ax.plot(S["Zg"][i, GO - 1:k + 1, 0], S["Zg"][i, GO - 1:k + 1, 1] - 0.004 * i, color="#bdbdbd", lw=0.7)
    succ, fail = pick(S, ssd, 1, 1)
    for idx, col in [(succ, BLUE), (fail, RED)]:
        for i in idx:
            Z = S["Zs"][i]
            hit = np.flatnonzero((Z[:, 0] >= 1) | (Z[:, 1] >= 1)); k = hit[0] if len(hit) else T - 1
            ax.plot(Z[GO - 1:k + 1, 0], Z[GO - 1:k + 1, 1], color=col, lw=0.9)
    ax.axvline(1, color="k", lw=0.9); ax.axhline(1, color="k", lw=0.9)
    ax.set(xlim=(-0.05, 1.15), ylim=(-0.08, 1.15), xticks=[], yticks=[])
    ax.set_xlabel("Go", fontsize=7, labelpad=1); ax.set_ylabel("Stop", fontsize=7, labelpad=1)


def label_wall_panel(ax, tl, title, note, letter):
    ax.set(xlim=(-0.15, 1.75), ylim=(-0.6, 1.5), xticks=[], yticks=[])
    ax.set_xlabel("$z_1$ (Go)"); ax.set_ylabel("$z_2$")
    ax.set_title(title, fontsize=10)
    ax.text(1 - tl * 1.3 + 0.06 if tl else 1.03, 1.3 if tl else 1.2, "stop", fontsize=8)
    ax.text(1 - tl * -0.25 + 0.06 if tl else 1.03, -0.25, "move", fontsize=8)
    ax.annotate("", xy=(0.05, 1.25), xytext=(0.05, 0.85), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.text(0.09, 1.0, "Stop field", fontsize=7.5, color=BLUE)
    ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=7.5, color="0.3")
    ax.text(-0.06, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")


MODELS = {"plain": dict(wall.REALISTIC), "tilted": dict(wall.REALISTIC, tilt=25.0),
          "pause": dict(wall.REALISTIC, tilt=25.0, pause_lo=0.3, pause_z1=0.7, pause_gain=0.2, pause_gain_z2=0.2)}
SSDS = [int(s) for s in R["plain"]["paired_drt"]]
KEYS = ["box", "plain", "tilted", "pause"]
TICKS = ["IRM", "plain", "tilted", "tilted\n+pause"]

# ---------------------------------------------------------------- Fig. 7
fig = plt.figure(figsize=(11, 8.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.38, wspace=0.38)

# A: plain wall, with the race in a box as inset
axA = fig.add_subplot(gs[0, 0:2])
tl = draw_wall(axA, sample("plain", MODELS["plain"]), SSD_EXAMPLE)
label_wall_panel(axA, tl, "Plain wall: equivalent to the independent race",
                 "the Stop field leaves progress\nalong $z_1$ unchanged: escapes\nkeep their Go timing", "A")
ins = axA.inset_axes([0.69, 0.50, 0.28, 0.44])
draw_box(ins, sample("box"), SSD_EXAMPLE)
ins.set_title("race in a box (IRM)", fontsize=7, pad=2)

# B: tilted wall, with the pause region as inset
axB = fig.add_subplot(gs[0, 2:4])
tl = draw_wall(axB, sample("tilted", MODELS["tilted"]), SSD_EXAMPLE)
label_wall_panel(axB, tl, "Tilted wall: geometry couples Go and Stop",
                 "pushed by the Stop field,\ntrajectories meet the wall sooner:\nescapes are faster", "B")
ins2 = axB.inset_axes([0.69, 0.50, 0.28, 0.44])
draw_wall(ins2, sample("pause", MODELS["pause"], seed=7), SSD_EXAMPLE, inset=True, n_go=5)
ins2.set(xlim=(-0.15, 1.45), ylim=(-0.6, 1.5), xticks=[], yticks=[])
ins2.set_title("+ pause region", fontsize=7, pad=2)

# C: paired Delta-RT
axC = fig.add_subplot(gs[1, 0:2])
for k in ["plain", "tilted", "pause"]:
    axC.plot(SSDS, [R[k]["paired_drt"][str(s)] for s in SSDS], "o-", ms=3, color=COL[k], label=LAB[k])
axC.axhline(0, color="k", lw=0.6, ls=":")
axC.set(xlabel="SSD (ms)", ylabel="paired $\\Delta$RT (ms)", title="Context independence")
axC.legend(fontsize=6.5, frameon=False, loc="upper right")
axC.text(-0.14, 1.08, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")

# D: Go-Stop coupling
axD = fig.add_subplot(gs[1, 2])
axD.bar(range(4), [R[k]["coupling_slope"][0] for k in KEYS], yerr=[R[k]["coupling_slope"][1] for k in KEYS],
        color=[COL[k] for k in KEYS], capsize=3)
axD.axhline(0, color="k", lw=0.6)
axD.set_xticks(range(4)); axD.set_xticklabels(TICKS, fontsize=7.5)
axD.set(ylabel="slope, SSRT vs simulated RT", title="Go–Stop coupling")
axD.text(-0.3, 1.08, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")

# E: population SSRT
axE = fig.add_subplot(gs[1, 3]); x = np.arange(4)
axE.bar(x - 0.18, [R[k]["ssrt_mechanistic"] for k in KEYS], 0.36, color="#555555", label="mechanistic")
axE.bar(x + 0.18, [R[k]["ssrt_integration"] for k in KEYS], 0.36, color="#bbbbbb", label="integration")
axE.set_xticks(x); axE.set_xticklabels(TICKS, fontsize=7.5)
axE.set(ylabel="population SSRT (ms)", title="Population SSRT"); axE.legend(fontsize=6.5, frameon=False)
axE.text(-0.3, 1.08, "E", transform=axE.transAxes, fontsize=12, fontweight="bold")
fig.savefig("fig7_wall_model.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- Fig. S8
o = R["box"]["per_trial"]; rt, ssrt, rt_sim = map(np.array, (o["rt"], o["ssrt"], o["rt_sim"]))
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
for a, xv, xl, key in [(ax[0], rt, "empirical RT (ms)", "slope_vs_empirical_rt"),
                       (ax[1], rt_sim, "trial-averaged simulated RT (ms)", "coupling_slope")]:
    q = np.quantile(xv, np.linspace(0, 1, 10)); kk = np.clip(np.searchsorted(q, xv, side="right") - 1, 0, 8)
    mx = [xv[kk == j].mean() for j in range(9)]; my = [ssrt[kk == j].mean() for j in range(9)]
    se = [ssrt[kk == j].std(ddof=1) / np.sqrt((kk == j).sum()) for j in range(9)]
    a.errorbar(mx, my, yerr=se, fmt="o", ms=3, capsize=2, color=ORANGE)
    b, c = np.polyfit(xv, ssrt, 1); xx = np.array([min(mx), max(mx)]); a.plot(xx, b * xx + c, "k--", lw=1)
    sl, sse = R["box"][key]
    a.set(xlabel=xl, ylabel="per-trial SSRT (ms)", title=f"slope {sl:+.2f} $\\pm$ {sse:.2f}")
plt.tight_layout(); fig.savefig("figS8_ssrt_estimator_null.png", dpi=200)

# ---------------------------------------------------------------- Fig. S12
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(SSDS, [R["tilted"]["paired_drt"][str(s)] for s in SSDS], "o-", ms=3, color=BLUE, label="tilted wall")
ax.plot(SSDS, [R["pause"]["paired_drt"][str(s)] for s in SSDS], "o-", ms=3, color=ORANGE, label="+ pause region (state space)")
ax.plot(SSDS, [R["pause_period"]["paired_drt"][str(s)] for s in SSDS], "s--", ms=3, color="#7a4fb0",
        label="+ pause period (150 ms after Stop)")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set(xlabel="SSD (ms)", ylabel="paired $\\Delta$RT (ms)"); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); fig.savefig("figS12_pause_region_vs_period.png", dpi=200)

print("wrote fig7_wall_model.png, figS8_ssrt_estimator_null.png, figS12_pause_region_vs_period.png")
