# 👗 Fit-Predict: Recommender System for Clothing Fit & Ratings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)

## 📖 Project Overview
The **Fit-Predict Recommender System** is a personalized machine learning model designed to predict clothing star ratings and fit. Developed as part of the **CSE 158 (Recommender Systems and Web Mining)** course, this project leverages a massive dataset of approximately **192,000 real-world clothing reviews** from **Rent the Runway** to provide tailored recommendations.

By combining structured user measurements, temporal dynamics, and unstructured textual reviews, the system effectively addresses the nuanced challenge of personalized clothing recommendations online.

## 📊 Dataset
* **Source:** Rent the Runway Clothing Reviews.
* **Size:** ~192,000 instances.
* **Data Types:** 
  * **User Metadata:** Height, weight, age, band size, body type.
  * **Item Metadata:** Size, fit feedback (small, fit, large).
  * **Textual Data:** Written user reviews.
  * **Temporal Data:** Review timestamps.

## 🛠️ Feature Engineering
To capture both broad trends and personalized nuances, we engineered three main categories of features:
1. **User/Item Metadata:** Structural information including the critical `fit_code` which assesses if an item runs small, true to size, or large.
2. **Temporal Features:** Dynamics like `days_since_review`, `item_popularity`, `item_days_since_last_review`, and `item_recent_rating`.
3. **Textual Features (TF-IDF):** Unstructured text from user reviews mapped into TF-IDF vectors to extract semantic meaning and sentiment.

## 🧠 Models Evaluated
We iteratively trained and evaluated multiple models, progressing from simple baselines to complex hybrid approaches:

* **Baselines:** Global Mean, User-Item Bias, and Text-Only (Bag-of-Words, Word2Vec).
* **Linear Regression:** Leveraged purely metadata and temporal features to establish a strong structured baseline.
* **Gradient Boosting Regressor:** Explored non-linear relationships within the structured metadata and temporal features.
* **Ridge Regression (Hybrid):** Combined User/Item Metadata, Temporal Features, and TF-IDF Textual Features into a highly regularized, comprehensive model.

## 🏆 Key Findings & Results
The progression of models demonstrated the importance of blending structured and unstructured data:

* **Fit is King:** In our Gradient Boosting model, `fit_code` was the most dominant feature, accounting for **46.6%** of predictive importance, followed closely by temporal features (e.g., recency of last review).
* **Text Completes the Picture:** While Gradient Boosting handled structured data well, the **Ridge Hybrid Model** incorporating TF-IDF features from text ultimately performed the best, capturing complex sentiment that raw metadata misses.

### Performance Comparison

| Model | Features Used | RMSE | MAE |
|-------|--------------|------|-----|
| **Linear Regression** | Metadata + Temporal | 0.7079 | 0.5722 |
| **Gradient Boosting** | Metadata + Temporal | 0.6744 | 0.5277 |
| **Ridge Regression** | Metadata + Temporal + TF-IDF | **0.5436** | **0.4124** |

> **Conclusion:** The Ridge Regression model achieved the best predictive accuracy, predicting the user's rating within half a star on average. Its robust performance highlights that blending user body measurements, temporal recency, and rich semantic review text provides the most accurate personalized predictions.

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook
* Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Running the Notebook
1. Clone the repository.
2. Ensure you have the dataset downloaded and placed in the appropriate directory (details within the notebook).
3. Open `FitPredict.ipynb` in Jupyter Notebook.
4. Run the cells sequentially to observe data preprocessing, feature engineering, model training, and evaluation.

## 📁 Repository Structure
* `FitPredict.ipynb`: Main Jupyter Notebook containing all data processing, EDA, feature extraction, and model implementations.
* `CSE158 Project.pdf`: Final academic project report detailing the methodologies and results.
* `README.md`: This file.
