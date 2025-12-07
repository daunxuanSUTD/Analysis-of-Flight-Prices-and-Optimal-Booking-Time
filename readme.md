
# Flight Price Analysis and Optimal Booking Time

This project studies how airline ticket prices evolve as the departure date approaches, and builds a machine-learning model to **predict flight prices** and estimate an **optimal booking window**.

The work follows an end-to-end ML pipeline:

> Data → Feature Engineering → Model Training & Comparison → Interpretation (PDP etc.) → Model Serving (FastAPI) → (Optional) Dockerization

The project is framed under **“AI for Social Good”**:  
by making price dynamics more transparent and providing better booking recommendations, it can help students, families, and low-income travellers plan cheaper trips and avoid overpaying for flights.

---

## 1. Repository Structure

Project root

```text
.
├─ README.md
├─ requirements.txt
├─ Dockerfile                             # Optional, for containerization
├─ Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb
├─ train.py                               # Training script: builds and saves the best model
├─ api.py                                 # FastAPI app for serving predictions
├─ Clean_Dataset.csv                      # Local data file (not versioned; download from Kaggle)
├─ model.joblib                           # Saved best pipeline (created by train.py, not versioned if large)
└─ figures/                               # Optional: exported figures used in the report
    ├─ pipeline_diagram.png
    ├─ model_comparison.png
    └─ pdp_days_left_vs_price.png
````

> Note: `Clean_Dataset.csv` and `model.joblib` may **not** be stored in Git due to size constraints.
> They are local artifacts that can be recreated by downloading the dataset from Kaggle and running `train.py`.

---

## 2. Dataset

We use the public **Flight Price Prediction** dataset from Kaggle:

* Kaggle link:
  `https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction`

After basic cleaning, the project uses:

* `Clean_Dataset.csv` placed in the **project root** (same folder as `train.py`).

The training script expects:

* Columns including (but not limited to):
  `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `stops`, `class`, `duration`, `days_left`, `price`, `flight` (if present), and any technical index column (e.g. `Unnamed: 0`, which will be dropped if it exists).

---

## 3. Environment Setup

The project is developed and tested with:

* **Python 3.10**
* Main libraries:

  * `pandas`, `numpy`
  * `matplotlib`, `seaborn`
  * `scikit-learn`
  * `xgboost`, `lightgbm`
  * `joblib`
  * `fastapi`, `pydantic`, `uvicorn`

### 3.1 Create and activate a virtual environment (Windows)

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate

# (Optional) upgrade pip
python -m pip install --upgrade pip
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
joblib
fastapi
pydantic
uvicorn
```

---

## 4. Reproducing the Analysis (Jupyter Notebook)

Main notebook:

* `Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb`

This notebook contains:

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

* **Evaluation metrics** (on the **original price scale**):

  * `R²`
  * `MAE`

* **Interpretability**

  * Feature importance (for tree-based models)
  * Partial Dependence Plot (PDP) for `days_left`
    (visualizing how predicted prices change as days before departure change)

* **Booking recommendation**

  * Empirical price curve vs. `days_left`
  * Estimated “optimal booking window” (e.g. around 20–50 days before departure)

### 4.1 Running the notebook

1. Activate the virtual environment:

   ```bash
   .\.venv\Scripts\activate
   ```

2. Start Jupyter (if running outside VS Code):

   ```bash
   jupyter notebook
   ```

   Or in VS Code:

   * Open the `.ipynb` file.
   * Select the Python 3.10 virtual environment as the kernel.
   * Run all cells from top to bottom.

3. Ensure that `Clean_Dataset.csv` is present in the project root so that the notebook and `train.py` can load it.

---

## 5. Training Script (`train.py`)

The script `train.py` encapsulates the final training logic used in the notebook.

### 5.1 What `train.py` does

1. **Data loading and preprocessing**

   * Reads `Clean_Dataset.csv`.
   * Drops `Unnamed: 0` if present.
   * Maps `stops` to numeric codes (0/1/2).
   * Creates `route = source_city + "-" + destination_city`.
   * Computes `log_price = log1p(price)` as the regression target.

2. **Feature matrix and target vector**

   * Drops `["price", "log_price", "flight", "source_city", "destination_city"]` from features.
   * Uses `log_price` as the target (`y`).
   * Ensures `days_left` is numeric (`float`).

3. **Preprocessing pipeline**

   * Numerical features: `["duration", "days_left"]`
   * Categorical features: all remaining columns.
   * Uses `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categoricals and `"passthrough"` for numericals.
   * Configured to output a `pandas.DataFrame` for better interpretability and compatibility.

4. **Model comparison**

   * Compares multiple regression models (Linear / Tree / Random Forest / XGBoost / LightGBM).
   * 80/20 train–test split with `random_state=42`.
   * Evaluates model performance on the **original price** scale using:

     * `R²`
     * `MAE`
   * Selects the model with the **lowest MAE**.

5. **Saving the best model**

   * Saves the full pipeline (preprocessing + best model) to:

     ```text
     model.joblib
     ```

   * This file may be large and is typically **not stored in Git**, but is regenerated by running `train.py`.

### 5.2 How to run `train.py`

From the project root:

```bash
# Activate venv first
.\.venv\Scripts\activate

# Run training
python train.py
```

After successful training, you should see a summary of each model’s performance and a message similar to:

```text
Saved best model pipeline to model.joblib
```

---

## 6. Serving the Model (`api.py` with FastAPI)

The script `api.py` exposes the trained model as a simple REST API using FastAPI.

### 6.1 API overview

* Loads `model.joblib` (the best pipeline saved by `train.py`).
* Defines a `POST /predict` endpoint accepting flight details and returning a predicted ticket price.

The request body includes:

* `airline`: string
* `source_city`: string
* `destination_city`: string
* `departure_time`: string (e.g. `"Morning"`, `"Evening"`)
* `arrival_time`: string (e.g. `"Afternoon"`, `"Night"`)
* `stops`: `"zero" | "one" | "two_or_more"`
* `travel_class`: `"Economy" | "Business"`
* `duration`: float (hours)
* `days_left`: int (days until departure)

### 6.2 Running the API server

From the project root:

```bash
# Activate venv
.\.venv\Scripts\activate

# Make sure model.joblib exists (run train.py first if needed)
python train.py  # optional, if not yet trained

# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

The service will be available at:

* Base URL: `http://localhost:8000`
* Interactive API docs (Swagger UI): `http://localhost:8000/docs`

### 6.3 Example request (PowerShell / cmd)

```bash
curl -X POST "http://localhost:8000/predict" ^
     -H "Content-Type: application/json" ^
     -d "{
           \"airline\": \"Indigo\",
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

Example JSON response:

```json
{
  "predicted_price": 4523.17
}
```

---

## 7. Reproducibility & Environment Details

* **Python version:** 3.10

* **Random seed:** 42 (for data splitting and model initialization where applicable)

* **Core libraries:**

  * `scikit-learn`
  * `xgboost`
  * `lightgbm`
  * `pandas`, `numpy`

* **Reproducing figures:**

  * All plots (model comparison, PDP, etc.) are generated in
    `Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb`.
  * Important plots can be exported and stored in the `figures/` folder for the report.

**How to reproduce the 80/20 test split and reported scores**

Both the notebook and `train.py` use:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

This means that when the TA:

1. Downloads the dataset from Kaggle and saves it as `Clean_Dataset.csv` in the project root, and
2. Runs:

   ```bash
   .\.venv\Scripts\activate
   python train.py
   ```

the script will **recreate the same 80/20 split**, use the **same 20% of records as the test set**, and recompute the evaluation metrics (`R²`, `MAE`) on the original price scale. The reported results in the notebook and PDF can therefore be reproduced (up to minor floating-point differences).

---

## 8. AI for Social Good Context

This project contributes to **AI for Social Good** in the following ways:

* **Cost savings for travellers**
  By modelling price dynamics and identifying an optimal booking window, users can plan flights more efficiently and avoid overpriced tickets.

* **Support for budget-constrained groups**
  Students, low-income families, and NGOs often have strict travel budgets. A transparent, data-driven booking recommendation tool can help them allocate resources more fairly.

* **Potential integration into public platforms**
  The model and API can be integrated into travel advisory systems, university travel portals, or NGO logistics tools to provide personalised booking guidance.

---

## 9. Team & Contributions 

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

## 10. Notes for Markers / TA

* The **main notebook** for reviewing the analysis is:
  `Machine Learning Analysis of Flight Prices and Optimal Booking Time_fin.ipynb`.
* The **end-to-end reproducible pipeline** is encapsulated in:

  * `train.py` (model training & saving)
  * `api.py` (model serving)
  * `requirements.txt` (dependencies)
* To **fully reproduce the reported performance**:

  1. Download the Kaggle dataset and save it locally as `Clean_Dataset.csv` in the project root.
  2. Create and activate the Python 3.10 virtual environment.
  3. Install dependencies using `pip install -r requirements.txt`.
  4. Run `python train.py` to:

     * perform the 80/20 train–test split with `random_state=42`,
     * train all candidate models,
     * select the best model by MAE on the 20% test set,
     * and save the final pipeline to `model.joblib`.

