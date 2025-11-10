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
* Generates Excel workbook output with sheets for scorecards, commentary, forecasts
* Modular codebase split by phases (1–6) for ingestion, scoring, forecasting, dashboard

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

* Python 3.10+ (configured via environment)
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
   python src/phase1_ingestion.py
   ```

### Tests

Run unit tests to check ingestion logic:

```bash
pytest -v
```

## 📊 Usage

After ingestion, following phases calculate scores, run forecasts, and output dashboard workbook. Example:

```bash
python src/phase2_scoring.py  
python src/phase3_forecasting.py  
```

Outputs are placed in the `processed_dir` as defined in config, e.g. Excel workbook `VendorDashboard_v1.0.xlsx`.

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


## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (if created) for guidelines.

