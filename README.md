# ML-Flow Smart Berth Planning Documentation

Keywords: Berth Planning, Maritime logistics, SmartPorts, Port optimization, Berth
allocation, Machine Learning


## Table of Contents
1. [Introduction](#introduction)
2. [Libraries and Dependencies](#libraries-and-dependencies)
3. [Data Engineering](#data-engineering)
   - [Dataset Loading](#dataset-loading)
   - [Feature Selection and Treatment](#feature-selection-and-treatment)
   - [Logarithmic Transformation](#logarithmic-transformation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
   - [YData Profiling](#ydata-profiling)
   - [RainCloud Plot](#raincloud-plot)
5. [Model Preparation](#model-preparation)
   - [Train-Test Split](#train-test-split)
   - [K-Fold Cross-Validation](#k-fold-cross-validation)
6. [Model Evaluation Metrics](#model-evaluation-metrics)
7. [Machine Learning Models](#machine-learning-models)
   - [Linear Regression](#linear-regression)
   - [Random Forest Regression](#random-forest-regression)
   - [XGBoost Regression](#xgboost-regression)
   - [Multilayer Perceptron (MLP) Regression](#multilayer-perceptron-mlp-regression)
8. [Model Analysis](#model-analysis)
   - [Residual Analysis](#residual-analysis)
   - [Performance Evaluation](#performance-evaluation)
   - [Feature Importance](#feature-importance)
   - [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
9. [Pipeline](#pipeline)
10. [OLS Regression Results](#ols-regression-results)

## Introduction

This Python script demonstrates a comprehensive machine learning workflow for predicting the time a vessel spends at a berth. It includes data loading, preprocessing, exploratory data analysis, model training, and evaluation using various regression techniques.

## Libraries and Dependencies

The script uses the following main libraries:
- pandas
- numpy
- matplotlib
- mlflow
- scikit-learn
- xgboost
- shap
- statsmodels

Additional libraries used for specific visualizations:
- ptitprince (for RainCloud plots)
- ydata_profiling (for data profiling)

## Data Engineering

### Dataset Loading

The dataset is loaded from a CSV file named 'dataset_modelagem.csv' located in the '../Datasets/' directory.

### Feature Selection and Treatment

Selected features include:
- 'Berth Name'
- 'Terminal Name'
- 'Time At Berth'
- 'Time At Port'
- 'Vessel Type - Generic'
- 'Commercial Market'
- 'Voyage Distance Travelled'
- 'Voyage Speed Average'
- 'Year of build'
- 'Voyage Origin Port'
- 'Flag'
- 'Gross tonnage'
- 'Deadweight'
- 'Length'
- 'Breadth'

Rows with null values are removed. Categorical variables are encoded using LabelEncoder.

### Logarithmic Transformation

The 'Time At Berth' and 'Time At Port' features are log-transformed. Rows with infinite values in 'Time At Berth' are removed.

## Exploratory Data Analysis

### YData Profiling

The script generates a comprehensive data profile using YData Profiling.

### RainCloud Plot

A RainCloud plot is created to visualize the distribution of 'Time At Port' across different berths.

## Model Preparation

### Train-Test Split

The data is split into training and testing sets with a 80-20 ratio.

### K-Fold Cross-Validation

5-fold cross-validation is used for model evaluation.

## Model Evaluation Metrics

Two main metrics are used:
1. Mean Squared Error (MSE)
2. R-squared (R2) Score

## Machine Learning Models

### Linear Regression

A simple linear regression model is implemented using scikit-learn's LinearRegression.

### Random Forest Regression

A Random Forest regressor is implemented with the following parameters:
- n_estimators: 24
- random_state: 30
- oob_score: True
- bootstrap: True

### XGBoost Regression

An XGBoost regressor is implemented with the following parameters:
- objective: 'reg:squarederror'
- random_state: 42

### Multilayer Perceptron (MLP) Regression

An MLP regressor is implemented with the following parameters:
- hidden_layer_sizes: (50, 50)
- activation: 'relu'
- solver: 'adam'
- random_state: 24
- max_iter: 100

Data is normalized using StandardScaler before training the MLP model.

## Model Analysis

### Residual Analysis

Residual analysis is performed using:
- Histogram of residuals
- Scatter plot of predicted vs residuals
- Q-Q plot

### Performance Evaluation

A scatter plot of actual vs predicted values is created to visualize model performance.

### Feature Importance

Feature importance is calculated and displayed for the Random Forest model.

### SHAP (SHapley Additive exPlanations)

SHAP values are calculated and plotted for both Linear Regression and Random Forest models to explain feature impacts.

## Pipeline

A simple pipeline is created combining K-Fold cross-validation with the Linear Regression model.

## OLS Regression Results

Ordinary Least Squares (OLS) regression is performed using statsmodels, and a summary of the results is printed.

This script provides a comprehensive approach to regression modeling, from data preparation to model evaluation and interpretation. It utilizes MLflow for experiment tracking and logging, enabling reproducibility and easy comparison of different models and parameters.
