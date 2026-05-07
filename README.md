# bengaluru-aqi-prediction
A machine learning web app that predicts the Air Quality Index (AQI) for Bengaluru, India. Built with CatBoost and Streamlit.

![Banner](image/blr_aqi.png)

## Project Overview
Air pollution is a critical public health issue in Bengaluru, one of India's fastest-growing cities. This project uses 5 years of historical pollution data (2015–2020) to build a machine learning model that predicts the daily AQI for different areas of the city.
The goal is to make AQI prediction accessible to everyday users with no technical knowledge required. Users simply select a date and their area, and the app predicts the expected air quality along with health advice.

## What it does

- User picks a date and area in Bengaluru
- App predicts AQI using a trained CatBoost model
- Shows color-coded result, health advice, and an interactive map of monitoring stations

**Note:** Predictions are based on historical monthly averages (2015–2020). Dates within the same month will produce similar AQI values since the model uses that month's average pollution levels as input.

## Tech Stack
- Language: Python 3.10+
- Data Handling: Pandas, NumPy
- Machine Learning: Scikit-learn, CatBoost
- Visualization: Matplotlib, Seaborn, Folium
- Web App: Streamlit, streamlit-folium

## Dataset
[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) - Kaggle
Filtered to Bengaluru → 1,881 rows after cleaning and feature engineering
  
## Model Performance

| Model | MAE | R2 |
| -------- | -------- | -------- |
| Linear Regression | 6.12 | 0.829 |
| Random Forest | 5.59 | 0.864 |
| CatBoost | 5.48 | 0.87 |

## Screenshots

App UI
![App UI](image/app_ui.png)

Prediction 
![Prediction](image/prediction_ui.png)

Interactive Map
![Map](image/map_ui.png)

## Run Locally
- Clone the repo
- install `requirements.txt`
- run notebooks 01→04
- then `streamlit run app.py`
