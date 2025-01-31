
# Insurance Charges Prediction using Regression

## Overview

This project applies **multiple regression techniques** to predict **insurance charges** based on various factors. It explores **Gradient Descent** and **Newton’s Methods** for optimization and provides **data visualization & preprocessing**. A **Streamlit-based GUI** allows users to interact with the dataset and models.

---

## Features

✅ **Exploratory Data Analysis (EDA):**

- Descriptive statistics, histograms, box plots, correlation heatmaps
- Pairwise plots highlighting smoker impact

✅ **Data Preprocessing:**

- Handling missing values
- Feature engineering (interaction terms)
- Normalization using MinMaxScaler

✅ **Regression Models Implemented:**

- **Gradient Descent Optimization**
- **Newton’s First-Order Method**
- **Newton’s Second-Order Method**

✅ **Performance Evaluation:**

- Mean Squared Error (MSE) & R²
- Weight evolution plots & cost function visualizations
- Contour plots for cost function

✅ **Streamlit GUI:**

- CSV file upload for user-defined datasets
- Interactive data visualization
- Model training & predictions in real-time

---

## Dataset

The dataset contains the following features:

- `age` → Age of the insured
- `sex` → Gender
- `bmi` → Body Mass Index
- `children` → Number of dependents
- `smoker` → Smoking status
- `region` → Geographic region
- `charges` → Insurance cost (target variable)

---

## Model Evaluation

| Model | MSE (Mean Squared Error) | R² Score |
| --- | --- | --- |
| Gradient Descent | **X.XX** | **X.XX** |
| Newton 1st Order | **X.XX** | **X.XX** |
| Newton 2nd Order | **X.XX** | **X.XX** |

📌 *Baseline model predicts the average charges, which serves as a benchmark for improvement.*

---

## Technologies Used

🔹 **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)

🔹 **Machine Learning** (Linear Regression, Gradient Descent, Newton’s Method)

🔹 **Streamlit** (for interactive GUI)

---

To run the GUI :

- install streamlit
- run this command: streamlit run [gui.py](http://gui.py/)
