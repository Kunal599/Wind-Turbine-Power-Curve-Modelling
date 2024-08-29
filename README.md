# Wind-Turbine-Power-Curve-Modelling
This project analyzes wind turbine power production using data from Data.csv. The goal is to predict wind turbine power production based on various features and compare the performance of different regression models.

Overview
The project involves data cleaning, visualization, and model training to predict the power production of a wind turbine. Key steps include data preprocessing, outlier removal, model training, and visualization of predictions versus actual power production.

Features
Data Loading and Preprocessing: Load the dataset, check for missing values, and perform outlier removal.
Visualization: Generate plots to visualize the relationship between wind speed and power production, theoretical power curves, and predictions.
Model Training: Train and evaluate various regression models including:
K-Nearest Neighbors (KNN)
Decision Trees
Random Forest
Gradient Boosting
Dependencies
numpy
pandas
matplotlib
seaborn
plotly
scipy
scikit-learn
xgboost
Usage
Load Data: Import the dataset and preprocess it by checking for missing values and removing outliers.
Data Visualization:
Plot real vs. theoretical power curves.
Generate pair plots to explore relationships between features.
Model Definition and Training:
Define features and target variables.
Train models using train_test_split and evaluate their performance using metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).
Model Evaluation:
Evaluate different models including KNN, Decision Trees, Random Forest, and Gradient Boosting.
Visualize model predictions against actual data and theoretical power curves.
Functions
outlier_remover(dat, prop, min, max): Removes outliers based on specified quantile thresholds.
Definedata(): Prepares features and target variables for model training.
Models(models): Trains and evaluates the given model, printing performance metrics.
Graph_prediction(y_actual, y_predicted): Plots the actual vs. predicted power production and theoretical power curves.
