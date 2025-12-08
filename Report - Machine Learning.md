## Report - Machine Learning Analysis of Flight Prices and Optimal Booking Time

**Authors:** Duan Xu, Ding Tianqi, Wang Zhenrong
**Course:** Production-ready Machine Learning
**Date:** November 30, 2025

---

### Executive Summary

This report presents a comprehensive machine learning analysis aimed at predicting flight ticket prices and identifying the optimal time to book. In the dynamic aviation industry, ticket prices fluctuate significantly based on demand, timing, and service class, creating opacity for consumers. Our objective was to develop a robust, reproducible predictive model and to translate its insights into **actionable booking rules**.

We used a cleaned dataset containing over 300,000 flight records, with features such as airline, route, flight duration, days left until departure, and service class. To stabilise the heavily right-skewed price distribution, we modelled the **log-transformed price** using `log1p(price)` as the regression target. Categorical variables (e.g. airline, city, time of day) were one-hot encoded, while numeric variables (duration, days_left, stops) were scaled appropriately.

We benchmarked five algorithms under a unified preprocessing pipeline:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

All models were evaluated on a held-out 20% test set (with `random_state = 42` for reproducibility) using:

* **R² on the log-price scale** (how much variance in log price we explain), and
* **MAE on the original price scale** (average absolute error in currency units).

The Random Forest Regressor emerged as the best model, achieving **R² ≈ 0.985** and **MAE ≈ 1,067 INR**, substantially outperforming the baseline Linear Regression model (R² ≈ 0.884, MAE ≈ 4,551 INR). This confirms that a non-linear ensemble model is far better suited than a simple linear model for this problem.

Key findings from our model interpretation include:

1. **Service Class Dominance**
   The distinction between Economy and Business class is the single most influential predictor of price, followed by days_left and flight duration.

2. **Critical Non-linear Booking Effect**
   Using Partial Dependence Plots (PDP) and piecewise / polynomial regression on days_left, we found:

   * A **structural breakpoint** around **15 days** before departure, after which prices increase sharply (the “price shock zone”).
   * A **theoretical minimum** of the fitted price curve around **37 days** before departure.

3. **Practical Optimal Window: 20–50 Days**
   Combining the breakpoint (≈15 days) and the theoretical minimum (≈37 days), we recommend a practical booking window of **20–50 days** before departure. This window avoids last-minute volatility while still capturing the cheaper “early-bucket” fares.

From an **AI for Social Good** perspective, our work aims to reduce information asymmetry between airlines and passengers. By providing transparent analysis and a simple web-based tool that visualises prices over days_left, we help budget-constrained travellers (e.g. students, families, NGOs) make more informed and cost-effective decisions.

---

### 1 Background and Introduction

Air travel pricing is notoriously volatile and complex. Unlike fixed-price retail goods, airline tickets are subject to dynamic pricing algorithms that adjust fares in real time based on supply, demand, competitor pricing, and temporal factors. For the average consumer, this volatility often results in frustration and sub-optimal purchasing decisions. Questions such as “Should I book now or wait?” are common but difficult to answer without data-driven insights.

The importance of solving this problem extends beyond individual savings. For travel agencies and aggregators, accurate price prediction models can enhance user experience, drive customer loyalty, and support more transparent revenue management strategies.

This project addresses the **information asymmetry** between airlines and passengers. By leveraging historical flight data and supervised machine learning techniques, we aim to reverse-engineer the determinants of flight prices and quantify how timing, route, and class affect cost. The potential impact includes:

* empowering consumers to make informed financial decisions,
* supporting travel platforms in offering fairer, more transparent price guidance, and
* contributing to the broader **AI for Social Good** agenda by making complex pricing logic more understandable.

---

### 2 Related Work

Price prediction in the travel industry is a long-standing research area. Traditional approaches often relied on **time-series forecasting** (e.g. ARIMA), focusing purely on historical price signals for a single route. While such models capture temporal patterns, they struggle to incorporate rich feature interactions between **static attributes** (e.g. origin, destination, service class) and **dynamic attributes** (e.g. days_left, demand shocks).

More recent work adopts **machine learning regressors** that can handle high-dimensional, mixed-type data. The literature suggests that ensemble methods such as **Random Forests** and **Gradient Boosting** often outperform single models and classical time-series techniques, due to their ability to model non-linear relationships and complex feature interactions.

In parallel, commercial platforms like **Google Flights**, **Skyscanner**, and **Hopper** provide users with price trend visualisations and heuristic advice (“Prices are low”, “Buy now”, or “Wait”). However, these systems are largely **black-box**: they offer high-level guidance without exposing the underlying models or feature contributions. Some products (e.g. Hopper) focus on “buy or wait” decisions but do not clearly articulate how variables like class, route, or days_left quantitatively drive prices.

Our work builds on these lines of research and practice but differs in three ways:

1. We explicitly model **log-transformed prices** with a transparent feature pipeline,
2. We focus on the **interpretability** of the “optimal booking window” via PDP and piecewise regression, and
3. We expose a simple **open FastAPI service + demo UI** that visualises the full price curve over days_left, rather than just a “buy / wait” label.

---

### 3 Problem Formulation and Overview

We formulate flight price prediction as a **supervised regression** problem. Let:

* ( X ) denote the feature vector describing a flight, including
  airline, source city, destination city, departure time, arrival time, number of stops, service class, duration, and days_left.
* ( Y ) denote the continuous target variable representing the ticket price.

Due to the heavy right skew of monetary values, we model the **log-transformed price**:

[
Y' = \log(1 + Y)
]

where ( \log(1 + Y) ) is implemented as `np.log1p(price)` in code. This transformation stabilises variance, reduces the influence of extreme outliers, and improves model training.

Our goal is to learn a function

[
f : X \mapsto \widehat{Y'}
]

such that the error between predicted log-price (\widehat{Y'}) and true log-price (Y') is minimised. During evaluation, we map predictions back to the original price scale:

[
\widehat{Y} = \exp(\widehat{Y'}) - 1
]

and compute:

* **R²** on the log-price scale, and
* **Mean Absolute Error (MAE)** on the original price scale.

We perform an **80/20 train–test split** with `random_state = 42` to ensure reproducibility. All models are trained on the same split to enable fair comparison.

---

### 4 Data Description

The dataset used for this analysis is `Clean_Dataset.csv`, containing approximately **300,153 entries and 12 columns** after cleaning. It is derived from a public Kaggle flight price dataset and includes:

* **Categorical features**:
  `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `stops`, `class`
* **Numeric features**:
  `duration` (flight time in hours), `days_left` (days between booking and departure)
* **Target**:
  `price` (ticket price in INR)

#### 4.1 Data Exploration Highlights

Exploratory analysis revealed:

* **Target Distribution**
  The `price` column is heavily right-skewed with a long tail of expensive tickets, motivating the use of `log1p(price)` as the regression target.

* **Categorical Dominance**
  Many key predictors (airline, cities, time of day, stops, class) are categorical, requiring appropriate encoding to feed into regression models.

* **Correlation with Days Left**
  A scatter plot of price vs. days_left showed a clear non-linear pattern: prices tend to be lower in a “middle” region (e.g. around 1–2 months before departure) and higher both very close to departure and very far in advance for some routes.

These insights guided our choice of features, transformations, and models.

---

### 5 Details of Solution

#### 5.1 Methods and Tools

We adopt a standard production ML stack:

* **Python** 3.10
* **Libraries**:
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `joblib`, `fastapi`, `uvicorn`
* **Development environment**:
  Jupyter Notebook for exploration, plus standalone scripts `train.py` and `api.py` for reproducible training and serving.
* **Version control**: Git + GitHub.

All models share a **common preprocessing pipeline** defined via `ColumnTransformer` and `Pipeline` in scikit-learn.

#### 5.2 Feature Engineering

Key feature engineering steps:

1. **Stops Encoding**

   * The textual `stops` feature is mapped to an ordinal numeric variable:

     * `"zero" → 0`, `"one" → 1`, `"two_or_more" → 2`
   * The original `stops` is retained as categorical; the numeric `stops_encoded` is used as a numeric feature.

2. **Route Construction**

   * We create a composite `route` feature:
     [
     \text{route} = \text{source_city} + "-" + \text{destination_city}
     ]
   * This captures route-specific effects beyond individual source and destination.

3. **Target Transformation**

   * Define `log_price = log1p(price)` as the regression target to address skewness.

4. **Feature Groups**

   * **Numeric**: `duration`, `days_left`, `stops_encoded`
   * **Categorical**: `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `class`, `route`

#### 5.3 Model Specifications and Regularization

We benchmark five models, each wrapped in the same preprocessing pipeline:

1. **Linear Regression**

   * Baseline model with no explicit regularization (ordinary least squares).
   * Useful as a simple benchmark: it assumes a linear relationship between one-hot encoded features and log price.

2. **Decision Tree Regressor**

   * Captures non-linear effects and interactions.
   * Regularization via depth and leaf constraints (e.g. `max_depth`, `min_samples_leaf`) to reduce overfitting.

3. **Random Forest Regressor**

   * Ensemble of decision trees trained on bootstrap samples with feature subsampling.
   * Hyperparameters include:

     * `n_estimators` (number of trees, e.g. 100),
     * `max_depth` (to control complexity),
     * `min_samples_leaf` (to smooth leaves),
     * `n_jobs = -1` to use all CPU cores.
   * This model reduces variance compared to a single tree and typically performs well on tabular data.

4. **XGBoost Regressor**

   * Gradient boosting ensemble with tree-based weak learners.
   * Key hyperparameters (learning rate, number of estimators, maximum depth) act as **explicit regularization** controls.

5. **LightGBM Regressor**

   * Gradient boosting with histogram-based splits, optimised for speed and large datasets.
   * Similar hyperparameter roles to XGBoost.

All models are trained using an **80/20 train–test split**. While full k-fold cross-validation would further stabilise estimates, we focus on a single, reproducible split due to time and compute constraints. We interpret model performance and complexity using **bias–variance trade-off** intuition:

* Linear Regression: high bias, low variance
* Decision Tree: low bias, high variance
* Random Forest / Gradient Boosting: more balanced, benefiting from ensembling.

#### 5.4 System Implementation and Code Repository

The complete, documented source code is hosted in our GitHub repository:

* `https://github.com/daunxuanSUTD/Analysis-of-Flight-Prices-and-Optimal-Booking-Time`

##### Training Pipeline

* `train.py` encapsulates the full pipeline:

  * Loads `data/Clean_Dataset.csv`
  * Applies feature engineering (including `stops_encoded`, `route`, and `log1p(price)`)
  * Trains all candidate models
  * Evaluates them on a fixed 20% test set (`random_state = 42`)
  * Selects the best model (lowest MAE in original price)
  * Saves the complete preprocessing + model pipeline to:

    * `models/model.joblib`
  * Exports evaluation artefacts to `output/`:

    * `model_results.csv`, `model_comparison_r2.png`, `model_comparison_mae.png`, `price_vs_days_left.png`, and (for tree models) `feature_importance_top20.png`.

##### Model Serving and Demo UI

We implement a **FastAPI** service in `api.py` with endpoints:

* `GET /health` – health check.
* `POST /predict` – single-point prediction; returns a predicted price for a full flight configuration including `days_left`.
* `POST /booking_plan` – enumerates `days_left` from 1 to 60 for a fixed configuration, predicts the price for each, and returns:

  * the full **price curve** (list of `{days_left, predicted_price}`), and
  * the **cheapest predicted day** and its price.
* `GET /demo` – a simple web front-end (`templates/demo.html`) using Chart.js to:

  * let users select route, airline, class, etc.,
  * call `/booking_plan` from the browser,
  * plot the **predicted price vs. days_left curve** and highlight the cheapest point.

This system implementation connects the **modeling work** with a practical, interactive tool that illustrates the “optimal booking time” concept in real time.

---

### 6 Evaluation

#### 6.1 Model Performance Results

All models are evaluated on a **held-out test set** (20% of the data) using:

* **R² (log-price)**
  Measures the proportion of variance in log-price explained by the model.
* **MAE (INR, original price)**
  Measures average absolute deviation between predicted and true prices, in real currency terms.

A simplified comparison is shown below:

| Model             | R² (log price) | MAE (INR, price) |
| ----------------- | -------------- | ---------------- |
| Random Forest     | 0.985          | 1,067.48         |
| Decision Tree     | 0.977          | 1,158.21         |
| XGBoost           | 0.965          | 2,342.85         |
| LightGBM          | 0.962          | 2,473.75         |
| Linear Regression | 0.884          | 4,550.68         |

The **Random Forest** model achieves the best performance, with the **highest R²** and **lowest MAE**, substantially outperforming the Linear Regression baseline. This confirms that:

* price dynamics are strongly non-linear, and
* interactions between features (e.g. class × days_left, route × duration) are important.

We also considered using cross-validation to estimate generalisation performance more robustly. Due to time and compute limits, we instead rely on a single, reproducible train–test split and note this as a limitation (Section 9.1).

#### 6.2 Feature Importance & Model Interpretation

To interpret the Random Forest, we extract **feature importance scores** (based on impurity decrease). The top features include:

* **Class (Economy / Business)** – the dominant contributor, confirming that service class is the primary cost driver.
* **Days Left** – ranked near the top, validating the central role of booking lead time.
* **Duration** – shorter flights often command a higher price, especially for busy business routes.
* **Route** – captures city-pair specific effects such as business vs leisure markets.

Although we initially framed this section as a “bias audit”, in practice the available dataset does not contain sensitive attributes (e.g. income, demographics). Therefore, our analysis focuses on **model transparency** via feature importance rather than a full fairness audit.

As a partial step towards “bias awareness”, we examined residual patterns by airline and route. We did not observe extreme systematic errors for any single carrier or city-pair, but we recognise that:

* the dataset is restricted to Indian routes, and
* GDP/income proxies are not included,

so a complete fairness audit is outside the scope of this project.

---

### 7 Discussion of Results

#### 7.1 The Price–Time Curve

To understand how booking time affects prices, we combined:

* **Partial Dependence Plots (PDP)** of predicted log-price vs. days_left, and
* **Polynomial + piecewise regression** fitted on aggregated price vs. days_left data.

The analysis revealed a clear **non-linear** pattern:

* **Breakpoint Detection (~15 days)**
  A piecewise linear model identified a structural break around **15 days** before departure.

  * Beyond this breakpoint (d < 15), prices increase sharply, consistent with last-minute “business traveller” fares.
  * Before this breakpoint (d ≥ 15), the curve is flatter and more stable.

* **Theoretical Minimum (~37 days)**
  A convex polynomial fit (on the smoothed curve) suggests the **theoretical minimum** occurs **around 37 days** before departure. This corresponds to airlines releasing lower buckets of seats 1–2 months ahead.

* **Practical Booking Window (20–50 days)**
  Combining these insights, we propose **20–50 days** as a **robust practical booking window**:

  * Starting from **20 days** gives a safety margin before the 15-day “danger zone”.
  * Extending up to **50 days** recognises that prices remain reasonably low and stable across a broader window, which offers flexibility for travellers.

This reconciles all three numbers:

* 15 days → “price shock” breakpoint / danger zone boundary
* 37 days → theoretical minimum of the fitted curve
* 20–50 days → recommended **operational window** that balances risk and flexibility

and thereby aligns the mathematical analysis with user-facing advice.

#### 7.2 Business and Social Implications

From a commercial perspective, our model could be integrated into:

* booking platforms (to generate price-sensitive recommendations),
* travel agencies’ internal tools (to advise clients on when to purchase), or
* meta-search engines (as an interpretable “price trend” widget).

From a social impact perspective, being able to **estimate and visualise booking windows** helps travellers who:

* cannot afford to “just book whenever”, and
* need to plan around salary cycles, exam dates, or family responsibilities.

By open-sourcing both the pipeline and the demo, we contribute to **transparency** in an industry where pricing mechanisms are often opaque.

---

### 8 Recommendations

Based on our modelling and mathematical analysis, we propose the following recommendations for travellers:

1. **Book 20–50 Days Before Departure**
   To secure the most favourable fares, travellers should target a booking window between **20 and 50 days** prior to the travel date, with the theoretical optimum around **37 days**.

2. **Avoid the “Danger Zone” (≤15 Days)**
   Booking within **15 days** of departure exposes travellers to large price spikes and limited seat inventory. Prices in this zone tend to be driven by last-minute demand, especially from business travellers.

3. **Focus on Timing and Class, Not Just Brand**
   Our feature importance analysis suggests that **service class and timing** dominate brand effects. While airline loyalty may matter for comfort or mileage, travellers seeking the lowest fare should prioritise:

   * travelling in Economy, and
   * booking within the recommended window
     over sticking to a specific carrier.

4. **Use Tools that Expose Price Curves**
   Instead of relying only on “buy/wait” labels, users should consult tools (like our demo) that show **full price vs. days_left curves**, making the trade-offs explicit.

---

### 9 Limitations and Future Work

#### 9.1 Limitations

* **Geographical Scope**
  The dataset is limited to routes between Indian cities (e.g. Delhi, Mumbai, Bangalore), so generalizability to international markets or other regions is not guaranteed.

* **Seasonality and Holidays**
  The dataset does not explicitly encode holidays or travel seasons. Major events (e.g. Diwali, school holidays) likely affect prices but are not represented in the current features.

* **External Factors**
  Macro-factors such as fuel prices, inflation, and sudden policy changes are excluded. Our model only learns from the provided historical prices and available features.

* **Evaluation Protocol**
  We use a single 80/20 train–test split with `random_state = 42`. While this ensures reproducibility, it does not capture the full uncertainty that k-fold cross-validation could provide.

#### 9.2 Future Technical Work

To improve and extend this work, we propose:

1. **Time-Series and Sequential Modelling**
   Incorporate time-series models (e.g. LSTM or temporal convolutional networks) that explicitly model sequential price changes over time for specific routes.

2. **External Data Integration**
   Enrich features with:

   * public holiday calendars,
   * school vacation periods,
   * fuel surcharge / jet fuel price indices.

3. **Route-Specific Models**
   Train separate models for different route types (e.g. business vs. leisure routes) to see whether optimal booking windows differ by market segment.

#### 9.3 Ethical Considerations

While our current dataset does not include sensitive personal attributes, future extensions (e.g. personalised pricing detection) may:

* require collecting data about user devices, locations, or browsing patterns;
* raise privacy concerns and possibly fall under regulations (e.g. GDPR).

If such data are used, future work should:

* explicitly define fairness metrics,
* conduct systematic bias audits, and
* explore how to **detect and discourage unfair price discrimination** rather than amplify it.

---

### 10 Proposal for Future Research

Beyond the present scope, we propose a line of research on **“Personalised Dynamic Pricing Detection”**. The core question is whether airlines or intermediaries are dynamically adjusting prices based on user-level signals such as:

* search history,
* device type,
* geolocation, or
* inferred income proxies.

A possible research design would involve:

* constructing multiple synthetic “digital identities” that differ only in selected attributes,
* querying real-time prices for identical flights from each identity, and
* statistically testing for systematic price differences.

Detecting such personalised pricing would have significant **ethical and regulatory implications**. In particular, it could inform discussions on consumer protection, algorithmic transparency, and fairness in digital marketplaces.

Our current work can serve as an analytical foundation and engineering template (data pipeline + model + API + UI) for that future direction.

---
