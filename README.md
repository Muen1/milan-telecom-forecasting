# Mobile Network Traffic Forecasting – Milan Telecom Dataset

**Author**: Cynthia Mutie 
**Course**: ML Techniques I  
**Date**: 21 st May 2026  

This repository contains the complete implementation for the assignment *"Comparative Time Series Analysis and Forecasting of Mobile Network Traffic"*. The project analyses 10‑minute aggregated mobile Internet activity (CDR‑based) over a 100×100 grid covering the city of Milan (10,000 areas) during November and December 2013. Three forecasting models – ARIMA, LSTM, and GRU – are implemented and compared for one‑step‑ahead prediction of traffic in three selected squares.


##  Repository Structure

```text
milan-telecom-forecasting/
├── data/
│ ├── raw/ 
│ ├── grid/ # grid_data.csv (for heatmap)
│ └── processed/ # Generated after running load_data.py
│ ├── target_series.csv # Time series for squares 4159, 4556, 5059
│ └── total_traffic.npy # Total traffic per square (10,000 values)
├── src/
│ ├── load_data.py # Efficient streaming loader (Task 1)
│ ├── generate_task2_figures.py # Creates all EDA figures (Task 2)
│ ├── train_evaluate.py # Trains and evaluates ARIMA, LSTM, GRU (Task 3)
│ ├── prepare_data.py # Data preprocessing & scaling
│ ├── sarima_model.py # ARIMA model wrapper
│ ├── nn_models.py # LSTM & GRU builders, sequence creation
│ └── validate_data.py # Auxiliary scripts
├── reports/
│ ├── figures/ # All plots (generated automatically)
│ └── model_performance.csv # Metrics (MAE, MAPE, RMSE, times)
├── requirements.txt 
├── README.md 
└── .gitignore 
```

---

##  Dataset Access

The raw data (~5 GB) is not included in this repository due to size limits. You must download it from the Harvard Dataverse:

- **Telecommunications activity dataset** (Milan):  
  [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)  
  Only the columns `Square id`, `Time Interval`, and `Internet traffic activity` are needed.

After downloading, place all daily `.txt` files (e.g., `sms-call-internet-mi-2013-11-01.txt`) into `data/raw/`.  
The grid file (if used) goes into `data/grid/grid_data.csv` – but the project generates a spatial heatmap directly from square IDs, so the grid file is optional.


##  Installation & Environment

### Prerequisites

- Python 3.9 – 3.11 (3.11 recommended, tested with 3.11.9)
- 8 GB RAM minimum (16 GB recommended)
- Operating system: Windows 10/11, macOS, or Linux

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/milan-telecom-forecasting.git
cd milan-telecom-forecasting
```

### Step 2: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

requirements.txt
```text
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
tensorflow
joblib
tqdm
```

## Execution Guide 
The workflow is split into three phases.

### Data Loading & Processing (Task 1)

This step reads all daily files, aggregates total traffic per square, and extracts the time series for the three target squares (4159, 4556, and the square with highest total traffic).

**Run**:
```bash
python src/load_data.py
```

**Output**: data/processed/target_series.csv (~12 MB) and data/processed/total_traffic.npy (80 KB).

### Exploratory Data Analysis (Task 2)

Generates all required figures (PDF, time series, decomposition, ACF/PACF, heatmap, anomalies).

**Run**
```bash
python src/generate_task2_figures.py
```

**Output**: Six PNG files inside reports/figures/.

### Model Training & Evaluation (Task 3)

Trains ARIMA (statistical), LSTM, and GRU on each of the three squares, evaluates on the week 16–22 December 2013, saves prediction plots and performance table.

**Run**
```bash
python src/train_evaluate.py
```

**Output**:

* Three combined prediction plots (`predictions_square_*.png`) in `reports/figures/`

* `reports/model_performance.csv` (MAE, MAPE, RMSE, training/prediction times)

* All scripts are self‑contained and will automatically create the `reports/figures/` directory if missing.

## Results Summary

The best performing model for the high‑traffic square (9510) is **LSTM**:

* RMSE = 39.65

* MAE = 17.61

* Training time ≈ 70 seconds (CPU)

Full quantitative results are in `reports/model_performance.csv`.
The report (PDF) and video demonstration provide detailed discussion and failure analysis.

## YouTube Video

## Report 

## Hardware and Software used.

* CPU: Intel Core i5‑8250U @ 1.60 GHz (4 cores)

* RAM: 8 GB

* OS: Windows 11

* Python: 3.11.9

* Libraries: pandas 3.0.3, numpy 2.4.6, statsmodels 0.14.6, tensorflow 2.21.0, scikit‑learn 1.6.1

## Troubleshooting

* MemoryError during SARIMA: The code uses ARIMA (non‑seasonal) to avoid memory issues. If you see a memory error, ensure you are using the provided `sarima_model.py` which calls `ARIMA.`

* Missing figures: Run `mkdir -p reports/figures` before the scripts.

* TensorFlow GPU warnings: They are harmless; the code runs on CPU. To enable GPU acceleration on Windows, use WSL2 or the TensorFlow‑DirectML plugin.

* Date parsing errors: The raw data uses millisecond timestamps; our `load_data.py` correctly parses them with `unit='ms'`.
