好，我给你一份**整合后的最终版 README**：

* 已经把你原来的内容 + 我刚写的版本融合在一起
* 加上了 **clone 仓库**、**Windows / macOS / Linux 三套环境配置**
* 结构、用词都偏“课程项目 + 老师验收友好”

你可以直接把下面整段保存为 `README.md`。

---

````markdown
# Flight Price Analysis and Optimal Booking Time

This project studies how airline ticket prices evolve as the departure date approaches, and builds a machine-learning model to **predict flight prices** and estimate an **optimal booking window**.

The work follows an end-to-end ML pipeline:

> Data → Feature Engineering → Model Training & Comparison → Interpretation (PDP etc.) → Model Serving (FastAPI) → (Optional) Dockerization

The project is framed under **“AI for Social Good”**:  
by making price dynamics more transparent and providing better booking recommendations, it can help students, families, and low-income travellers plan cheaper trips and avoid overpaying for flights.

---

## 1. Repository Structure

Final project layout:

```text
Analysis-of-Flight-Prices-and-Optimal-Booking-Time/
├─ README.md
├─ requirements.txt
├─ Dockerfile                         # Optional, for containerization
├─ train.py                           # Training script: builds and saves the best model
├─ api.py                             # FastAPI app for serving predictions
│
├─ data/
│   └─ Clean_Dataset.csv              # Cleaned dataset (from Kaggle)
│
├─ models/
│   └─ model.joblib                   # Saved best pipeline (created by train.py)
│
├─ output/                            # Training & evaluation artifacts
│   ├─ model_results.csv
│   ├─ model_comparison_r2.png
│   ├─ model_comparison_mae.png
│   ├─ price_vs_days_left.png
│   └─ feature_importance_top20.png   # Only for tree-based best model
│
├─ notebook/
│   └─ Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb
│
└─ templates/
    └─ demo.html                      # Simple web UI for the API demo
````

> Notes:
>
> * `data/Clean_Dataset.csv` and `models/model.joblib` may **not** be stored in Git due to size constraints.
> * They can be recreated by downloading the dataset from Kaggle and running `train.py`.

---

## 2. Dataset

We use the public **Flight Price Prediction** dataset from Kaggle:

* Kaggle link:
  `https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction`

After basic cleaning, the project uses:

* `data/Clean_Dataset.csv`

The training script expects columns including (but not limited to):

* `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`,
  `stops`, `class`, `duration`, `days_left`, `price`
* Optional columns such as `flight` or technical index columns (e.g. `Unnamed: 0`) are dropped in preprocessing if present.

---

## 3. Getting Started

### 3.1 Clone the repository

```bash
git clone https://github.com/<your-username>/Analysis-of-Flight-Prices-and-Optimal-Booking-Time.git
cd Analysis-of-Flight-Prices-and-Optimal-Booking-Time
```

(Replace `<your-username>` with your actual GitHub username if needed.)

---

## 4. Environment Setup

The project is developed and tested with:

* **Python 3.10**
* Main libraries:

  * `pandas`, `numpy`
  * `matplotlib`
  * `scikit-learn`
  * `xgboost`, `lightgbm`
  * `joblib`
  * `fastapi`, `pydantic`, `uvicorn`

### 4.1 Create and activate a virtual environment (Windows)

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate

# (Optional) upgrade pip
python -m pip install --upgrade pip
```

### 4.2 Create and activate a virtual environment (macOS / Linux)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate

# (Optional) upgrade pip
python -m pip install --upgrade pip
```

> If `python3` is not available, try using `python` instead, depending on your system configuration.

### 4.3 Install dependencies

Once the virtual environment is activated:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
pandas
numpy
matplotlib
scikit-learn
xgboost
lightgbm
joblib
fastapi
pydantic
uvicorn
```

---

## 5. Exploratory Analysis (Jupyter Notebook)

Main notebook:

* `notebook/Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb`

The notebook contains:

* **Data exploration & cleaning**

  * Removing technical index columns (e.g. `Unnamed: 0`)
  * Handling missing values (if any)

* **Feature engineering**

  * Mapping `stops` from `"zero" / "one" / "two_or_more"` → `0 / 1 / 2`
  * Creating `route = source_city + "-" + destination_city`
  * Defining target `log_price = log1p(price)` to reduce skew

* **Model comparison**

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * XGBoost Regressor
  * LightGBM Regressor

* **Evaluation metrics**

  * `R²` (on the **log_price** scale)
  * `MAE` (on the **original price** scale)

* **Interpretability**

  * Feature importance (for tree-based models)
  * Partial Dependence Plot (PDP) for `days_left`
    (visualising how predicted prices change as days before departure change)

* **Booking recommendation**

  * Empirical price curve vs. `days_left`
  * Estimated “optimal booking window” (e.g. around 20–50 days before departure)

### 5.1 Running the notebook

1. Activate the virtual environment:

   ```bash
   # Windows
   .\.venv\Scripts\activate

   # or macOS / Linux
   source .venv/bin/activate
   ```

2. Start Jupyter (if running outside VS Code):

   ```bash
   jupyter notebook
   ```

   Or in VS Code:

   * Open the `.ipynb` file.
   * Select the `.venv` Python 3.10 environment as the kernel.
   * Run all cells from top to bottom.

3. Make sure `data/Clean_Dataset.csv` exists so that both the notebook and `train.py` can load it.

---

## 6. Training Script (`train.py`)

The script `train.py` encapsulates the final training pipeline used in the notebook.

### 6.1 What `train.py` does

1. **Path setup**

   * Ensures the following folders exist:

     * `data/`, `models/`, `output/`, `notebook/`
   * Loads data from:
     `data/Clean_Dataset.csv`

2. **Data loading and preprocessing**

   * Checks that all required columns exist.
   * Maps `stops` to numeric codes:

     * `"zero" → 0`, `"one" → 1`, `"two_or_more" → 2` (stored in `stops_encoded`)
   * Creates `route = source_city + "-" + destination_city`.
   * Computes `log_price = log1p(price)` as the regression target.

3. **Feature matrix and target vector**

   * Feature columns:

     ```text
     ["airline", "source_city", "destination_city",
      "departure_time", "arrival_time",
      "class", "duration", "days_left",
      "stops_encoded", "route"]
     ```

   * Target column: `log_price`.

4. **Train–test split**

   * Uses an 80/20 split with a fixed random seed:

     ```python
     train_test_split(X, y, test_size=0.2, random_state=42)
     ```

5. **Preprocessing pipeline**

   * Numerical features: `["duration", "days_left", "stops_encoded"]`
   * Categorical features:
     `["airline", "source_city", "destination_city", "departure_time",
       "arrival_time", "class", "route"]`
   * Uses a `ColumnTransformer`:

     * **Numeric**: `SimpleImputer(strategy="median")` + `StandardScaler`
     * **Categorical**: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

6. **Model zoo and evaluation**

   Trains 5 regression models:

   * `LinearRegression`
   * `DecisionTreeRegressor`
   * `RandomForestRegressor`
   * `XGBRegressor`
   * `LGBMRegressor`

   For each model:

   * Fits on the training set using the full pipeline (preprocessing + model).

   * Predicts `log_price` on the test set, then converts back to **original price**:

     ```python
     y_pred_log = pipe.predict(X_test)
     y_pred_price = np.expm1(y_pred_log)
     y_true_price = np.expm1(y_test)
     ```

   * Computes on the test set:

     * `R²` (on `log_price`)
     * `MAE` (on original `price`)

   * Stores results in `output/model_results.csv`.

7. **Selecting and saving the best model**

   * Selects the model with the **lowest MAE** on the test set (original price scale).
   * Saves the full pipeline (preprocessing + best model) to:

     ```text
     models/model.joblib
     ```

8. **Generated plots (saved in `output/`)**

   * `model_comparison_r2.png` – bar chart of R² for all models.
   * `model_comparison_mae.png` – bar chart of MAE (original price) for all models.
   * `price_vs_days_left.png` – scatter plot: ticket price vs. days_left (sampled subset).
   * `feature_importance_top20.png` – top 20 features by importance for tree-based best model
     (Random Forest / XGBoost / LightGBM only).

### 6.2 How to run `train.py` (Windows / macOS / Linux)

From the project root:

```bash
# 1. Activate venv
# Windows
.\.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# 2. Run training
python train.py
```

After successful training, you should see:

* A printed summary of each model’s performance.
* `models/model.joblib` created or updated.
* Several files in `output/`:

  * `model_results.csv`
  * `model_comparison_r2.png`
  * `model_comparison_mae.png`
  * `price_vs_days_left.png`
  * (Optional) `feature_importance_top20.png`

---

## 7. Serving the Model (`api.py` with FastAPI)

The script `api.py` exposes the trained model as a simple REST API using FastAPI.

### 7.1 API overview

* Loads `models/model.joblib` (the best pipeline saved by `train.py`).
* Defines a `POST /predict` endpoint accepting flight details and returning a predicted ticket price.
* Provides:

  * `GET /health` – health check endpoint.
  * `GET /docs` – Swagger UI (interactive API documentation).
  * `GET /demo` – a simple web form (from `templates/demo.html`) calling `/predict` via JavaScript.

The request body for `POST /predict` includes:

* `airline`: string
* `source_city`: string
* `destination_city`: string
* `departure_time`: string (e.g. `"Morning"`, `"Evening"`)
* `arrival_time`: string (e.g. `"Afternoon"`, `"Night"`)
* `stops`: `"zero" | "one" | "two_or_more"`
* `travel_class`: `"Economy" | "Business"`
* `duration`: float (hours)
* `days_left`: int (days until departure)

### 7.2 Running the API server (Windows / macOS / Linux)

From the project root:

```bash
# Activate venv
# Windows
.\.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# Make sure models/model.joblib exists (run train.py first if needed)
python train.py  # optional, if not yet trained

# Start FastAPI server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:

* Base URL: `http://127.0.0.1:8000`
* Health check: `http://127.0.0.1:8000/health`
* Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`
* Demo UI: `http://127.0.0.1:8000/demo`

### 7.3 Example request (curl)

**Windows (PowerShell / cmd)**

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
     -H "Content-Type: application/json" ^
     -d "{
           \"airline\": \"IndiGo\",
           \"source_city\": \"Delhi\",
           \"destination_city\": \"Mumbai\",
           \"departure_time\": \"Morning\",
           \"arrival_time\": \"Evening\",
           \"stops\": \"one\",
           \"travel_class\": \"Economy\",
           \"duration\": 2.5,
           \"days_left\": 30
         }"
```

**macOS / Linux**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "airline": "IndiGo",
           "source_city": "Delhi",
           "destination_city": "Mumbai",
           "departure_time": "Morning",
           "arrival_time": "Evening",
           "stops": "one",
           "travel_class": "Economy",
           "duration": 2.5,
           "days_left": 30
         }'
```

Example JSON response:

```json
{
  "predicted_price": 4523.17
}
```

(Exact value depends on the random seed and model training.)

---

## 8. Reproducibility Checklist (for Instructor / TA)

This section summarises the **exact steps** to reproduce the project results.

### 8.1 Environment setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/Analysis-of-Flight-Prices-and-Optimal-Booking-Time.git
   cd Analysis-of-Flight-Prices-and-Optimal-Booking-Time
   ```

2. Create and activate a Python 3.10 virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the Kaggle dataset and place the cleaned CSV as:

   ```text
   data/Clean_Dataset.csv
   ```

### 8.2 Retrain the model

From the project root:

```bash
python train.py
```

Expected:

* `models/model.joblib` created/updated.
* `output/model_results.csv` and several `.png` plots generated.

Both the notebook and `train.py` use:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Therefore, re-running `train.py` reconstructs the **same 80/20 split** and re-computes metrics (`R²`, `MAE`) on the same test set (up to small floating-point differences).

### 8.3 Run the API and demo

From the project root:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open in a browser:

* `http://127.0.0.1:8000/docs` – interactive API docs
* `http://127.0.0.1:8000/demo` – demo web page
* `http://127.0.0.1:8000/health` – health check

---

## 9. AI for Social Good Context

This project contributes to **AI for Social Good** in the following ways:

* **Cost savings for travellers**
  By modelling price dynamics and identifying an optimal booking window, users can plan flights more efficiently and avoid overpriced tickets.

* **Support for budget-constrained groups**
  Students, low-income families, and NGOs often have strict travel budgets. A transparent, data-driven booking recommendation tool can help them allocate resources more fairly.

* **Potential integration into public platforms**
  The model and API can be integrated into travel advisory systems, university travel portals, or NGO logistics tools to provide personalised booking guidance.

---

## 10. Team & Contributions

* **DingTianqi 1010730**

  * Data cleaning and feature engineering
  * Notebook analysis and visualizations

* **DuanXu 1010728**

  * Model design and comparison
  * Training script (`train.py`) and evaluation

* **WangZhengrong 1010724**

  * FastAPI implementation (`api.py`)
  * Dockerfile and deployment setup
  * Report writing and AI for Social Good framing

---

## 11. Notes for Markers / TA

* The **main notebook** for reviewing the analysis is:
  `notebook/Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb`.

* The **end-to-end reproducible pipeline** is encapsulated in:

  * `train.py` – model training and saving
  * `api.py` – model serving
  * `requirements.txt` – dependency specification

To fully reproduce the reported performance:

1. Place the cleaned dataset in `data/Clean_Dataset.csv`.
2. Create and activate a Python 3.10 virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python train.py` to:

   * perform the 80/20 train–test split with `random_state=42`,
   * train all candidate models,
   * select the best model by **MAE** on the 20% test set (original price scale),
   * and save the final pipeline to `models/model.joblib`.

You may then start the FastAPI service (`api.py`) and use `/predict` or the `/demo` page to interact with the model.

---

```
