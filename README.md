# bengaluru-aqi-prediction
End-to-end analysis and classification of Bengaluru's air quality using 5 years of pollutant data. Includes EDA, feature engineering, and a comparison of Logistic Regression, Random Forest, and CatBoost models to predict AQI categories.

![Banner](image/blr_aqi.png)

## Features
- City-level filtering and cleaning of real-world CPCB pollutant data
- Handling of missing values with documented reasoning
- Exploratory data analysis across 5 years of daily AQI readings
- Seasonal and temporal feature engineering (month, season, day of week)
- Correlation analysis to identify key pollutant drivers of AQI
- Multi-model classification with Logistic Regression, Random Forest, and CatBoost
- Model evaluation using accuracy, confusion matrix, and classification report
- Feature importance analysis to interpret model decisions

## Methodology
- Data Collection: Kaggle Air Quality Data in India dataset, filtered to Bengaluru (2015–2020)
- Data Cleaning: Missing value imputation using column medians, dropping rows with no AQI record
- Exploratory Data Analysis: Trend analysis, seasonal breakdown, pollutant correlation heatmaps
- Feature Engineering: Extracted month, year, day of week, and season from date column
- Modeling: Trained and compared three classifiers on an 80/20 train-test split
- Evaluation: Classification report, confusion matrix, and feature importance visualization
- Interpretation: Written analysis of model strengths, failure cases, and real-world implications

## Tech Stack
- Language: Python
- Environment: Jupyter Notebook
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, CatBoost
