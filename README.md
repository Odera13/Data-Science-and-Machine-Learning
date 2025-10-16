# 🧠 **Machine Learning Experiments & Model Comparison Repository**

**By Enos — Data-Driven Strategist | Web Developer | E-commerce & SaaS Consultant**

---

## 🎯 **Overview**

This repository is a curated collection of **end-to-end machine learning experiments**, each notebook designed to demonstrate **clarity**, **reproducibility**, and **comparative insight** in model building.

Across these projects, you’ll see one consistent theme:

> **Performance only has meaning when comparison is fair, data preparation is rigorous, and reasoning is transparent.**

Every notebook within this repo was created not just to train a model, but to **sharpen decision-making** — understanding *why* one model performs better than another, *when* to use statistical versus deep learning approaches, and *how* to maintain reproducibility across experiments.

---

## 📚 **Notebooks Included**

### 1. ❤️ **Random Forest Model Comparison on Heart Disease Data**

This notebook explores the performance of three `RandomForestClassifier` models using the Heart Disease dataset.  
It emphasizes the critical importance of **consistent data splits** when evaluating models.

**Key Highlights:**
- Data loaded from GitHub and split using a fixed random seed for reproducibility.  
- Custom evaluation function measuring Accuracy, Precision, Recall, and F1-score.  
- Comparison between:
  - Baseline model (default hyperparameters)
  - `RandomizedSearchCV` optimized model
  - `GridSearchCV` refined model  
- Performance visualized through a bar chart comparing the three models.

**Result:**  
`GridSearchCV` achieved the highest performance (**Accuracy: 86.89%, F1: 0.88**) — demonstrating that **targeted hyperparameter tuning outperforms defaults**.

> **Core Lesson:**  
> Reproducibility and consistent data splits are non-negotiable for credible model comparison.

---

### 2. 🐶 **End-to-End Dog Breed Classification with TensorFlow Hub**

An end-to-end deep learning pipeline using **TensorFlow 2.x** and **MobileNetV2** (from TensorFlow Hub) to classify **120 dog breeds** from images — directly inspired by the **Kaggle Dog Breed Identification Challenge**.

**Core Dependencies:**  
`TensorFlow`, `TensorFlow Hub`, `Pandas`, `NumPy`, `Matplotlib`, `Scikit-learn`, `os`, and `datetime`.

**Workflow Overview:**
1. Data preprocessing and one-hot encoding of labels.  
2. Image loading, normalization, batching, and validation split creation.  
3. Transfer learning using a pre-trained MobileNetV2 model.  
4. Model compiled and trained with `TensorBoard` and `EarlyStopping` callbacks.  
5. Predictions made on validation, test, and custom image sets.  
6. Kaggle-ready CSV submission prepared using `pandas`.

**Result:**  
A **production-ready classification workflow** — scalable, visual, and reproducible — demonstrating mastery of **TensorFlow pipelines** and **real-world deployment practices**.

> **Core Lesson:**  
> Transfer learning isn’t just efficient — it’s a strategic shortcut for scaling deep learning models with minimal computational waste.

---

### 3. 🌾 **SARIMAX vs LSTM: Comparative Time Series Modeling for Pest Infestation**

A comparative deep-dive into forecasting models — **SARIMAX (statistical)** vs **LSTM (neural)** — applied to **pest infestation time series data** enriched with weather-based exogenous variables.

**Data Preparation:**
- Merging pest reports with multi-source weather data (CSV-based).  
- Handling missing values, reordering columns, and chronological sorting.  
- Ensuring structured, time-based data integrity for fair forecasting.

**Model Comparison:**

**SARIMAX**
- Captures seasonality and weather-related exogenous influences.  
- Forecasts pest infestation for 14–30 days ahead.  
- Visualizes predictions with confidence intervals.

**LSTM**
- Sequence-based deep learning approach.  
- Trained with scaled and windowed time series data.  
- Evaluated using RMSE and R² metrics.  
- Forecasts plotted for visual comparison with SARIMAX results.

**Result:**  
Both models offer unique advantages:
- **SARIMAX** → interpretability and clear seasonality insights.  
- **LSTM** → flexibility in capturing nonlinear dynamics and long-term dependencies.

> **Core Lesson:**  
> Forecasting is both an art and a science — sometimes the interpretability of classical models outweighs the complexity of deep learning, and vice versa.

---

## ⚙️ **Dependencies**

These notebooks use a combination of classical machine learning and deep learning ecosystems:

| Category | Libraries |
|-----------|------------|
| **Core Libraries** | `numpy`, `pandas`, `matplotlib`, `scikit-learn` |
| **Deep Learning** | `tensorflow`, `tensorflow-hub`, `keras` |
| **Statistics & Forecasting** | `statsmodels` (for SARIMAX) |
| **Utility Modules** | `os`, `datetime`, `IPython.display` |

All notebooks are written in **Python 3.x** and structured for **Google Colab** or **Jupyter Notebook** environments.

---

## 🔍 **Core Principles Across All Projects**

| **Principle** | **Description** |
|----------------|-----------------|
| **Reproducibility** | Every model uses fixed random seeds and consistent data splits. |
| **Comparative Fairness** | All comparisons are made under identical conditions and metrics. |
| **Interpretability** | Clear visualizations and performance metrics make results transparent. |
| **Scalability** | Pipelines are designed for easy extension — new models can be integrated seamlessly. |
| **Real-world Application** | Each project mirrors a practical ML challenge — from healthcare to image recognition to agriculture. |

---

## 🧩 **Repository Philosophy**

This isn’t just a machine learning repo — it’s a **manifesto for disciplined experimentation**.

Each notebook showcases a mindset:

> Curiosity guided by structure.  
> Creativity grounded in reproducibility.  
> Technology serving human clarity.

In a world full of models chasing metrics, this work emphasizes **methodology, mindset, and meaning** — because mastery isn’t in getting a high score; it’s in **understanding why**.

---

## 🚀 **Future Enhancements**

- Comparative study of **XGBoost vs LightGBM** for tabular data.  
- Exploration of **hybrid LSTM–SARIMA ensemble forecasting**.  
- Integration of **model explainability tools** (SHAP, LIME).  
- Deployment-ready pipelines using **TensorFlow Serving** and **FastAPI**.

---

## 🧠 **Author**

**Enos**  
_Data Strategist | Web Developer | SaaS & E-commerce Consultant_  

Driven by **precision**, powered by **curiosity**.  
Helping people and brands uncover the hidden levers that move performance.
