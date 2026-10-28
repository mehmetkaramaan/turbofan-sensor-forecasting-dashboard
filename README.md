# Turbofan Sensor Forecasting — Python ML/DL Experiment

A Python time-series experiment comparing ten machine learning and deep learning models for **next-cycle T50 sensor prediction** using NASA C-MAPSS FD001. The browser dashboard is a companion visualization of the recorded experiment.

**Start with the model code:** [Python training and evaluation](train.py) · [English notebook](turbofan_sensor_forecasting.ipynb) · [Recorded result table](recorded_model_comparison.csv)

## Reproduce the Python experiment

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

Alternatively, open `turbofan_sensor_forecasting.ipynb` in Jupyter or Kaggle and run the cells in order. The notebook and script use the same saved Kaggle model settings; the script uses Matplotlib's non-interactive backend for local execution. Training downloads the pinned FD001 Parquet files and writes figures, predictions, model comparisons, learning histories and an experiment manifest under `turbofan_outputs/`.

### Experimental design

- Forecast T50 at cycle t+1 from the previous 30 observed cycles; windows never cross engine boundaries.
- Reserve 20 complete official training engines for validation; fit scalers and select variable input channels using only the remaining 80 engines.
- Exclude engine ID, cycle index, RUL and dataset/split labels from model inputs.
- Compare KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, LightGBM, MLP and SimpleRNN against persistence.
- Select the candidate by validation RMSE. Use original °R units for held-out test MAE/RMSE; also report R², macro-engine MAE and measured computational cost.
- MLP and SimpleRNN minimize standardized-target MSE with Adam, at most 35 epochs, batch size 128 and validation early stopping with patience 6.

`recorded_model_comparison.csv` was extracted from the saved executed Kaggle output; publishing it did not rerun training. CatBoost had the lowest validation RMSE (4.1168 °R), while Extra Trees had a slightly lower test RMSE (4.1185 vs 4.1227 °R). Test ranking was not used to replace validation-based selection. No repeated-seed significance claim is made.

The original notebook's training, split, architecture and hyperparameters are preserved in the English source. Cached notebook outputs were removed to avoid mixing old Turkish figures with rerun results. `experiment_provenance.json` records the source notebook hash and publication changes. Requirements are compatible version ranges, not the original environment lock; hardware/library changes can change timings and numerical results. The script produces evaluation artifacts; it does not currently export fitted model weights for deployment.

### Source and visualization

`train.py` and the notebook contain the actual model-fitting and evaluation workflow. `build.py` only assembles the dashboard. `index.html` is a generated artifact of `dashboard.template.html` plus `benchmark-data.json`; `.gitattributes` marks that generated file accordingly, while the handwritten dashboard template remains visible in language statistics. The GitHub Pages entry point stays at the repository root so the existing demo URL continues to work.

An interactive, fully offline visualization of NASA C-MAPSS FD001 engine sensor trajectories and next-cycle temperature forecasts, developed by **Mehmet Karaman**.

**[Launch the live dashboard](https://mehmetkaramaan.github.io/turbofan-sensor-forecasting-dashboard/)**

[![Interactive turbofan animation showing the engine cross-section and six sensor channels](dashboard-preview.png)](https://mehmetkaramaan.github.io/turbofan-sensor-forecasting-dashboard/)

*Click the preview to launch the interactive animation. Explore engine sensor dynamics, switch forecasting models, and inspect next-cycle T50 predictions.*

The dashboard combines a schematic turbofan cross-section with recorded benchmark measurements. It connects time-series modeling, machine learning, and an engineering interpretation of engine sensor behavior.


## Launch and explore the engine animation

### [▶ OPEN THE INTERACTIVE ENGINE ANIMATION](https://mehmetkaramaan.github.io/turbofan-sensor-forecasting-dashboard/)

The public demo runs directly in a browser through **GitHub Pages**; visitors do not need a GitHub account or a Python environment. GitHub's `index.html` file view displays source code, so open the live link above to see the **Play** button.

1. Start with **Engine 7** to access ten learned models and the persistence baseline.
2. Click **Play** to animate the schematic fan and airflow while recorded sensor values and charts advance through operating cycles.
3. Use **Next cycle**, **Reset**, the timeline slider, and **Playback speed** to inspect individual states.
4. Choose one to four model checkboxes under **Model comparison**. Each gets its own forecast and residual panel. The star marks its next, unrevealed target.
5. Follow each model’s residuals and MAE/RMSE as measurements are revealed. Switch the evaluation interval between all history and the last 10/30/60 cycles. Select another sensor to compare training-standardized trajectories.
6. Use **Show full observed trajectory** for retrospective inspection or **Download CSV** to export the measurements revealed so far.

**How the animation relates to the results:** the tables below summarize the full held-out test set, while the live counters summarize only the selected engine and revealed forecast targets. Engine 7 has stored model predictions; the other 99 engines currently provide six-channel measurement playback. The browser does not fit models. Motor geometry and motion are schematic, while the displayed sensor values come from the FD001 benchmark.

## Unified model selection and individual panels

Choose **one to four models** in the single **Model comparison** selection area. There is no separate active-model dropdown. Each checked model creates exactly one dedicated panel with its next-cycle T50 forecast, MAE, RMSE, observed-versus-forecast chart, and its own residual chart. Four selected models produce four panels; one produces one. Panels share forecast and residual scales for comparison.

The **Error evaluation interval** can use all revealed targets or the last **10, 30, or 60 operating cycles**. The evaluation interval changes error aggregation; it does not change the trained model's fixed 30-cycle input window. Every selected model uses the same revealed target cycles of the same engine. Future targets remain excluded even with full-trajectory view enabled.

The summary table defaults to selected models. **All 11 models - table only** expands the ranking without changing the checked models or panel count. Relative RMSE reduction uses persistence over the same target interval; negative values mean worse performance than persistence in that interval. Engine-specific scores may differ from the full-test results below. Forecasts are stored for engine 7; other engines display sensor measurements only.

Use **Download CSV** to export revealed measurements with a forecast column for each selected model. **Export comparison** downloads the current metric table. The on-page guide explains the 17-input, next-cycle T50 task, MAE/RMSE, residual signs, cycle navigation, engine and sensor selection, playback speed, and what requires retraining. Sensor values and model hyperparameters are not modified by the dashboard.

## Console layout

A neutral charcoal interface separates playback controls from the engine overview and model comparison workspace. Thermal accents identify engine measurements; muted model colours distinguish forecast and residual panels. Method and scientific-scope notes expand on demand. The interface remains self-contained and works offline.

## Interactive features

- Explore **100 test engines** and six sensor channels: T50, T30, P30, Nf, Nc, and BPR.
- Play, pause, reset, step forward, change playback speed, and scrub through operating cycles.
- Compare **ten trained models and a persistence baseline** using stored experiment forecasts for engine 7.
- Inspect the 30-cycle input window, next-cycle forecast, residuals, cumulative MAE and RMSE, and standardized sensor trajectories.
- Reveal the full observed trajectory for retrospective inspection, or keep future measurements hidden.
- Download revealed measurements and available forecasts as CSV.
- Run without API keys, external JavaScript libraries, or a backend. All interface text, chart labels, accessibility labels, and CSV headers are English.

## Forecasting task

**Input:** the previous 30 operating cycles. **Target:** T50 at the next cycle, the low-pressure turbine outlet temperature in degrees Rankine (°R).

The underlying experiment compared KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, LightGBM, MLP, and SimpleRNN against the last-observation persistence baseline. Engines were separated between training and validation; preprocessing statistics were fitted only on training engines. The official test engines were held out.

The saved full-test experiment reported CatBoost RMSE **4.123 °R**, MAE **3.294 °R**, and R² **0.631**, compared with persistence RMSE **5.670 °R** (approximately **27.3% lower RMSE**). CatBoost was selected by validation error. These are experiment results, not metrics recomputed by the web application. Dashboard metrics cover only the selected engine and targets revealed so far, so they differ from aggregate test metrics.

## Recorded experiment results

The following tables are transcribed from the supplied executed Kaggle notebook, `nasa-turbofan-10-model-ile-sens-r-tahmini (1).ipynb`. They report the saved run; no models were retrained to create this README. All ten learned models are compared with the same persistence baseline.

### Dataset and evaluation design

| Partition | Engines | Forecast windows |
| --- | --- | --- |
| Training | 80 | 13,990 |
| Validation | 20 | 3,641 |
| Official held-out test | 100 | 10,096 |

The original FD001 training file contains 20,631 rows from 100 engines; the official test file contains 13,096 rows from another 100 engines. A 30-cycle window predicts the next cycle's T50 value. Windows stay within each engine, and engine-level separation prevents overlapping trajectories from leaking across training and validation. Scaling uses only training engines. Engine identifiers, cycle indices, RUL labels, and dataset/split metadata are excluded from model features.

### Predictive performance

Rows are sorted by **test RMSE for reporting only**. Model selection used validation RMSE, not test ranking. Lower MAE/RMSE and higher R² are better.

| Model | Test MAE (°R) | Test RMSE (°R) | Test R² | Validation RMSE (°R) | RMSE reduction vs. persistence |
| --- | --- | --- | --- | --- | --- |
| Extra Trees | 3.2941 | 4.1185 | 0.6314 | 4.1305 | 27.36% |
| Random Forest | 3.2959 | 4.1215 | 0.6309 | 4.1602 | 27.31% |
| CatBoost | 3.2940 | 4.1227 | 0.6307 | 4.1168 | 27.29% |
| XGBoost | 3.3020 | 4.1300 | 0.6294 | 4.1441 | 27.16% |
| LightGBM | 3.2984 | 4.1312 | 0.6292 | 4.1493 | 27.14% |
| Gradient Boosting | 3.3073 | 4.1369 | 0.6281 | 4.1463 | 27.04% |
| MLP | 3.4002 | 4.2559 | 0.6064 | 4.3466 | 24.94% |
| SimpleRNN | 3.4062 | 4.2623 | 0.6052 | 4.2572 | 24.82% |
| KNN | 3.4133 | 4.2629 | 0.6051 | 4.2623 | 24.81% |
| Decision Tree | 3.9378 | 4.9267 | 0.4726 | 5.0225 | 13.11% |
| Persistence (last value) | 4.5458 | 5.6698 | 0.3015 | 5.7100 | 0.00% |

**Selection and interpretation.** CatBoost achieved the lowest validation RMSE (**4.1168 °R**) and was therefore selected. Its test RMSE was **4.1227 °R**, compared with **5.6698 °R** for persistence: approximately **27.29% lower RMSE**. Its MAE was **3.2940 °R**, equivalent to approximately **1.83 °C of temperature difference**. R² = **0.6307** describes explained variance; it is not a classification accuracy percentage.

Extra Trees produced the lowest observed test RMSE (**4.1185 °R**), with Random Forest and CatBoost extremely close. That small separation does not establish a statistically reliable winner: this notebook reports one engine split and run, without repeated-split uncertainty estimates. Boosted and bagged tree ensembles outperformed the MLP and SimpleRNN configurations in this experiment; this does not establish that recurrent models are generally inferior.

### Computational cost and engine-level error

| Model | Training time (s) | Prediction time (ms/sample) | Mean per-engine MAE (°R) |
| --- | --- | --- | --- |
| Extra Trees | 31.7764 | 0.0141 | 3.3058 |
| Random Forest | 173.1372 | 0.0125 | 3.3090 |
| CatBoost | 7.4337 | 0.0050 | 3.3039 |
| XGBoost | 6.1253 | 0.0025 | 3.2960 |
| LightGBM | 5.6044 | 0.0062 | 3.2970 |
| Gradient Boosting | 231.2875 | 0.0045 | 3.3048 |
| MLP | 5.4318 | 0.0544 | 3.3833 |
| SimpleRNN | 17.2109 | 0.0935 | 3.4262 |
| KNN | 0.0048 | 0.3104 | 3.4376 |
| Decision Tree | 6.1846 | 0.0005 | 3.9946 |
| Persistence (last value) | 0.0000 | 0.0000 | 4.5423 |

Timings describe the notebook's execution environment, batch sizes, and implementation. They are not universal hardware benchmarks or isolated real-time latency guarantees. KNN's short fitting time does not imply low inference cost. Mean per-engine MAE gives each engine equal weight, whereas the overall test MAE weights individual forecast windows; the two need not match. RMSE reduction is `100 × (1 − model RMSE / persistence RMSE)` using the notebook's reported values.

### Model families

| Family | Models | Role in the comparison |
| --- | --- | --- |
| Baseline | Persistence | Predicts the next T50 as the latest observed T50 |
| Neighbor-based ML | KNN | Compares the current input window with similar training windows |
| Single-tree ML | Decision Tree | Provides an interpretable nonlinear partitioning baseline |
| Bagged/randomized tree ensembles | Random Forest, Extra Trees | Combine many trees to reduce variance |
| Gradient-boosted tree ensembles | Gradient Boosting, XGBoost, CatBoost, LightGBM | Combine successive trees to learn residual structure |
| Feed-forward neural network | MLP | Learns nonlinear mappings from a flattened input window |
| Recurrent neural network | SimpleRNN | Processes the input as a sequence of sensor vectors |

The recorded MLP architecture was Flatten → Dense(64) → Dense(32) → Dense(1), with **34,817 trainable parameters**. The SimpleRNN architecture was SimpleRNN(32) → Dense(16) → Dense(1), with **2,145 trainable parameters**. These architectures refer to the evaluated notebook models; the browser dashboard replays stored forecasts.

## Data and stored predictions

The dashboard bundles all 100 FD001 test-engine trajectories for the six displayed sensors. **Stored model forecasts are available for engine 7 only**, preserved from the actual modeling experiment. Other engines display measurements without fabricated forecasts. Models are not trained or executed in the browser.

Sensor scaling uses the training-only 80-engine partition (20 validation engines selected using NumPy `default_rng(42)` from sorted training-engine IDs). The colored bars use training 2nd–98th percentiles; the comparison chart uses training means and standard deviations.

Sources:

- [NASA C-MAPSS degradation modeling](https://c3.ndc.nasa.gov/dashlink/resources/14/)
- [FD001 dataset mirror](https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine), pinned revision `4d14a0bf6248235bce31cbdacb297e03e0f75d52`.

## Scientific interpretation

This is a **data-replay and forecasting visualization**, not a thermodynamic simulator or a validated digital twin. Geometry, blade rotation, airflow particles, and combustor illumination are illustrative. They do not solve thrust, combustion physics, internal temperature fields, or physical rotational dynamics. The T50 indicator represents the LPT outlet sensor, not combustor temperature.

An operating cycle is not a second. Sensor colors are relative measurement ranges, not diagnostic or safety thresholds. The dashboard does not estimate remaining useful life, and test-trajectory endpoints do not imply failure.

## Run locally

Open `index.html` directly in a modern browser. No installation is required. Alternatively:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000`. To rebuild the self-contained HTML using Python's standard library:

```sh
python3 build.py
```

## Repository structure

| File | Purpose |
| --- | --- |
| `train.py` | Full Python model training, diagnostics and evaluation |
| `turbofan_sensor_forecasting.ipynb` | English experiment notebook |
| `recorded_model_comparison.csv` | Recorded Kaggle metrics, extracted without retraining |
| `experiment_provenance.json` | Source hash, dataset revision and publication provenance |
| `requirements.txt` | Python experiment dependencies |
| `index.html` | Self-contained interactive dashboard, including data |
| `dashboard.template.html` | Readable interface, SVG, styling, and JavaScript source |
| `benchmark-data.json` | Bundled measurements, stored predictions, and training statistics |
| `build.py` | Reproducible assembly of the offline HTML |
| `.nojekyll` | Serve static files directly with GitHub Pages |

GitHub's repository file preview does not execute HTML or JavaScript. Use the **live GitHub Pages link** above, or download and open `index.html`.
