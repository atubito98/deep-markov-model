# Wall model of action stopping: code for Fig. 7

This folder contains the code that reproduces **Fig. 7** of the paper, together with the two related supplementary figures (**Fig. S8** and **Fig. S12**). All results are computed from the toy models themselves; no data are needed.

## Contents

| File | Description |
|---|---|
| `race_model.py` | Independent race model as a two-dimensional process ("race in a box") |
| `wall_model.py` | Wall models: plain wall, tilted wall, pause region, pause period |
| `population_ssrt.py` | Stop-signal analyses on the toy models (stop-failure probability, population SSRT) |
| `compute_fig7.py` | Step 1: computes all quantities shown in the figures and saves them to `fig7_results.json` |
| `plot_fig7.py` | Step 2: draws the figures from `fig7_results.json` |
| `environment.yml`, `requirements.txt` | Software requirements |

## Requirements

Python 3.11 (any version from 3.9 on should work), NumPy ≥ 1.26 and matplotlib ≥ 3.8. No GPU is needed.

With conda:

```bash
conda env create -f environment.yml
conda activate wall-model
```

or with pip:

```bash
pip install -r requirements.txt
```

## Reproducing Fig. 7

Run the two scripts in order, from this folder:

```bash
python compute_fig7.py    # step 1: about 10-15 minutes on a laptop CPU
python plot_fig7.py       # step 2: a few seconds
```

Step 1 prints a summary of the main values and writes `fig7_results.json`. Step 2 writes three figures:

| Output | Figure |
|---|---|
| **`fig7_wall_model.png`** | **Fig. 7, the main result**: (A, B) example trajectories in the plain and tilted wall models; (C) paired ΔRT; (D) Go–Stop coupling; (E) population SSRT |
| `figS8_ssrt_estimator_null.png` | Fig. S8: the per-trial SSRT estimator applied to an independent race |
| `figS12_pause_region_vs_period.png` | Fig. S12: pause region versus pause period |

All random seeds are fixed, so repeated runs give the same values.

## License

Copyright 2026 Istituto Superiore di Sanità (ISS)

Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the European Commission - subsequent versions of the EUPL (the "Licence"); you may not use this work except in compliance with the Licence. You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12

Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the Licence for the specific language governing permissions and limitations under the Licence.
