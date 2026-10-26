# Turbofan Sensor Forecasting Dashboard

An interactive, fully offline visualization of NASA C-MAPSS FD001 engine sensor trajectories and next-cycle temperature forecasts, developed by **Mehmet Karaman**.

**[Launch the live dashboard](https://mehmetkaramaan.github.io/turbofan-sensor-forecasting-dashboard/)**

The dashboard combines a schematic turbofan cross-section with recorded benchmark measurements. It connects time-series modeling, machine learning, and an engineering interpretation of engine sensor behavior.

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
| `index.html` | Self-contained interactive dashboard, including data |
| `dashboard.template.html` | Readable interface, SVG, styling, and JavaScript source |
| `benchmark-data.json` | Bundled measurements, stored predictions, and training statistics |
| `build.py` | Reproducible assembly of the offline HTML |
| `.nojekyll` | Serve static files directly with GitHub Pages |

GitHub's repository file preview does not execute HTML or JavaScript. Use the **live GitHub Pages link** above, or download and open `index.html`.
