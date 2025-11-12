# brightstar

**Project to output dashboard on ASIN-levels and Vendor levels for personal optimization**

## 🚀 Overview

This project ingests weekly vendor export data (ASIN-level, vendor-level), normalizes it, and produces analytical dashboards and forecasting. It covers:

* Data ingestion & normalization (Phase 1)
* Scoring engine and vendor scorecards
* Forecasting of vendor metrics (GMS, SoROOS, ASP, etc)
* Commentary generation and dashboard output (Excel workbook)
* Modular architecture to support large volume exports (1 000+ SKUs)

## 🧩 Features

* Automatic detection of export layout (wide vs long)
* Normalization of week labels to fiscal week codes
* Configurable via `config.yaml` for column numbers or names
* Scorecard engine with fine granularity (1/1000 moves)
* Forecast engine (Prophet, LSTM ensemble) for metric trends and potential gains
* Dashboard engine that assembles configurable Excel workbooks with summary, scorecards, commentary, and forecasts
* Phase 6 pipeline orchestrator that stitches every phase together with resumable CLI controls
* Modular codebase split by phases (1–6) for ingestion, scoring, forecasting, commentary, dashboard, and orchestration

## 📁 Repository Structure

```
brightstar/
├── README.md  
├── requirements.txt  
├── config.yaml  
├── src/  
│   ├── phase1_ingestion.py  
│   ├── phase2_scoring.py  
│   └── …  
├── tests/  
├── .github/  
│   └── workflows/  
└── docs/  
```

## 🛠️ Getting Started

### Prerequisites

* Python 3.12 (recommended)
* `requirements.txt` installed:

  ```bash
  pip install -r requirements.txt
  ```

### Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/brightstar.git
   cd brightstar
   ```
2. Update `config.yaml` to reflect your folder paths, column numbers, identifier columns.
3. Place your weekly export file into the `raw_data_dir` defined in config.
4. Run Phase 1 ingestion:

   ```bash
   python -m src.phase1_ingestion --config config.yaml
   ```

   The script prints a phase summary (rows processed, number of weeks, last week ingested), writes a normalized dataset to the
   processed directory, and persists an ingestion state JSON for subsequent phases.

5. Run Phase 2 scoring to generate vendor/ASIN scorecards:

   ```bash
   python -m src.phase2_scoring --config config.yaml
   ```

   This reads the normalized dataset, applies the configurable metric weights/directions, and writes CSV/Parquet scorecards plus
   an audit trail under `data/processed/scorecards/`.

6. Run Phase 3 commentary to produce automated narratives:

   ```bash
   python -m src.phase3_commentary --config config.yaml
   ```

   The commentary engine compares week-over-week movements, applies the config-driven templates, and writes CSV/Parquet outputs
   under `data/processed/commentary/` ready for dashboard integration.

7. Run Phase 4 forecasting to project key metrics and potential gains:

   ```bash
   python -m src.phase4_forecasting --config config.yaml
   ```

   The forecasting engine loads normalized history, applies the ensemble (Prophet, LSTM, baseline) models per vendor/ASIN, and
   saves CSV/Parquet outputs under `data/processed/forecasts/` alongside run metadata summarizing methods and history coverage.

8. Run Phase 5 dashboard generation to consolidate insights into Excel:

   ```bash
   python -m src.phase5_dashboard --config config.yaml
   ```

   The dashboard generator loads normalized data, scorecards, commentary, and forecasts, enriches vendor scorecards with linked
   insights, and emits a timestamped workbook under `data/processed/dashboard/` with styled sheets and run summaries.

9. Automate everything with the Phase 6 pipeline (optional when developing, recommended in production):

   ```bash
   python -m src.brightstar_pipeline --config config.yaml
   # or install the package locally and use the console script
   # brightstar-pipeline --config config.yaml --resume
   ```

   The orchestrator validates configuration, executes Phases 1–5 in sequence, supports resumable runs (`--resume`) that skip
   phases with existing outputs, and prints a consolidated run summary.

   A convenience shell wrapper is also available:

   ```bash
   scripts/run_pipeline.sh path/to/config.yaml --resume
   ```

### Tests

Run unit and integration tests to check ingestion, scoring, commentary, forecasting, dashboard, and pipeline orchestration logic:

```bash
pytest -v
```

CI and local developers both execute the full pipeline integration test to ensure end-to-end reproducibility using the bundled
sample data and configuration templates.

## 📊 Usage

After ingestion, following phases calculate scores, run forecasts, and output dashboard workbook. Example:

```bash
python -m src.phase2_scoring --config config.yaml
python -m src.phase3_commentary --config config.yaml
python -m src.phase4_forecasting --config config.yaml
python -m src.phase5_dashboard --config config.yaml
```

Outputs are placed in the `processed_dir` as defined in config. Scorecards are stored under `data/processed/scorecards/`,
commentary narratives under `data/processed/commentary/`, forecast projections under `data/processed/forecasts/`, and Excel
dashboards under `data/processed/dashboard/`.

## 🧰 Phase 6 – Full Pipeline Automation

The `brightstar_pipeline` module coordinates every phase with logging, error handling, and optional resume semantics. Use it
whenever you need deterministic reproducibility or production scheduling:

```bash
python -m src.brightstar_pipeline --config config.yaml --fail-fast
```

Key flags:

* `--phases phase1,phase3` – run a custom subset.
* `--start-phase phase3 --end-phase phase5` – execute a contiguous slice.
* `--resume` – skip phases that already produced outputs according to `config.yaml` paths.

When installed via `setup.py`, the `brightstar-pipeline` console script provides the same interface without referencing the
module path explicitly.

## 📦 Packaging & Deployment

* `setup.py` enables editable installs (`pip install -e .`) and exposes the `brightstar-pipeline` console entry point.
* `scripts/run_pipeline.sh` offers a portable shell wrapper for CI/CD or cron scheduling.
* `.github/workflows/ci-python.yml` now runs unit tests and an end-to-end pipeline smoke test on every push/PR to guarantee the
  shipped configuration and sample data remain reproducible.

## 📈 Forecasting & Scoring Engine

* Scoring: Fine granularity per vendor, per metric, scaling to 1/1000 precision.
* Forecasting: Combines time-series models (Prophet, LSTM) in an ensemble with configurable weights for better prediction of future metric values (e.g., GMS growth, SoROOS improvement).
* Commentary: Automated text templates populated with calculated results (Week-over-Week growth, vendor spotlight, potential points to gain).

## ⚙️ Configuration

The `config.yaml` allows controlling:

* Column number or name for each metric (supports updates easily)
* Week detection mode (date-range pattern, fiscal week map)
* Score thresholds, icon sets (supports 5-icon system, not just 3)
* Forecast horizon (3–4 weeks)
* Output workbook sheet names and layout

## 💡 Future Improvements

* Add richer icon-sets for conditional formatting (5 icons instead of 3)
* Expand dashboard to columns A :P for extended metrics
* Add user prompt interface to dynamically map columns if layout changes
* Publish web-based interactive dashboard (in addition to Excel)
* Enable scheduling of weekly runs (cron or workflow)


## 📝 Known Issues / Data Types

### Value Coercion and Parquet Format

Phase 1 ingestion automatically coerces values to ensure Parquet compatibility:

* **Currency values** like `$1,234.56` are converted to floats (1234.56)
* **Percentage values** like `43.92%` are converted to decimals (0.4392)
* **Comma-separated numbers** like `1,234` are converted to floats (1234.0)
* **Invalid values** that cannot be parsed become NaN

This coercion happens before writing to Parquet to avoid dtype errors. The system validates that ASIN and vendor_code columns exist and contain meaningful unique values before proceeding.

### Parquet vs CSV Output

**Parquet is strongly preferred** for normalized output because:
* It preserves numeric types without ambiguity
* It supports schema validation
* It's more efficient for large datasets
* It's the default format in `config.yaml`

If Parquet writing fails (rare), the system automatically falls back to CSV with explicit options:
* UTF-8 encoding
* Minimal quoting
* Unix-style line terminators (`\n`)

When reading CSV files back, use explicit dtypes to preserve schema:

```python
df = pd.read_csv(
    'normalized_phase1.csv',
    dtype={'asin': 'string', 'vendor_code': 'string'},
    keep_default_na=False
)
```

### Masterfile Handling

The masterfile (ASIN → vendor mapping) supports:
* Both `.xlsx` and `.csv` formats
* Case-insensitive column matching (e.g., "ASIN", "asin", "Asin" all work)
* Multi-sheet Excel files (prefers 'Info', then 'Info2', then first sheet)
* Automatic deduplication by ASIN (keeps last occurrence)


## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (if created) for guidelines.

