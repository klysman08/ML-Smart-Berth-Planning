# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: ips_cienciadedados
#     language: python
#     name: python3
# ---

# ## Engenharia de dados

# ### Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


# ### Dataset

#carregar arquivos csv 
dataset = pd.read_csv('../Datasets/dataset_modelagem.csv')
dataset

# ### Tratamento e seleção de fetures

df_modelagem = dataset[['Berth Name', 'Terminal Name', 'Time At Berth', 'Time At Port' , 'Vessel Type - Generic', 'Commercial Market','Voyage Distance Travelled', 'Voyage Speed Average', 'Year of build', 'Voyage Origin Port', 'Flag', 'Gross tonnage', 'Deadweight', 'Length', 'Breadth' ]]

#remover linhas  com valores nulos
df_modelagem = df_modelagem.dropna()
df_modelagem.info()

# +
from sklearn.preprocessing import LabelEncoder

df_modelagem = df_modelagem.copy()

# Encode all columns object type
for column in df_modelagem.columns:
    if df_modelagem[column].dtype == type(object):
        le = LabelEncoder()
        df_modelagem[column] = le.fit_transform(df_modelagem[column])

df_modelagem.info()
# -

# ### Logaritmo para fetures de tempo

# + vscode={"languageId": "javascript"}
df_modelagem[['Time At Berth', 'Time At Port']] = np.log(df_modelagem[['Time At Berth', 'Time At Port']])
# -

# Remover linhas com valores infinitos na coluna 'Time At Berth'
df_modelagem = df_modelagem[np.isfinite(df_modelagem['Time At Berth'])]

# ## EAD Profiling

# + vscode={"languageId": "javascript"}
from ydata_profiling import ProfileReport
profile = ProfileReport(df_modelagem, title="Pandas Profiling Report")
profile
# -

# ### RainCloud

import ptitprince as pt
plt.subplots(figsize=(12, 11))
pt.RainCloud(data = df_modelagem, y = 'Time At Port', x= 'Berth Name', orient = 'h', width_box=0.2, width_viol=2.5, palette='Set2', bw=0.1, move=0.2)


""" import seaborn as sns

sns.pairplot(df_modelagem) """

# ## K-Fold

# ### Split dataset para treino e teste

# +
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold

X = df_modelagem.drop('Time At Berth', axis=1)
y = df_modelagem['Time At Berth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-fold cross-validation object with k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# -

# ## Métricas

# +
from sklearn.metrics import r2_score, mean_squared_error

def metrics(y_test, y_pred):
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# -

# ## Regressão Linear

# +
from sklearn.linear_model import LinearRegression

def model_lr(X_train, y_train, X_test):
    # Criando experimento no MLflow
    my_exp = mlflow.set_experiment('LinearRegression')

    with mlflow.start_run(experiment_id=my_exp.experiment_id):

        model_lr = LinearRegression(n_jobs=-1)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        # Perform cross-validation and get the scores
        scores = cross_val_score(model_lr, X_train, y_train, cv=kfold)

        # Print the scores
        print("Cross-Validation Scores:", scores)
        print("Mean Score:", scores.mean())

        mse, r2 = metrics(y_test, y_pred_lr)

        # Logando os parâmetros do modelo
        mlflow.log_param('n_jobs', -1)

        # Logando as métricas do modelo
        mlflow.log_metrics({'mse': mse, 'r2': r2})
        mlflow.log_metric('Cross-Validation Scores', scores.mean())
        
        # Logando o modelo
        mlflow.sklearn.log_model(model_lr, 'LinearRegression')

        # Logando os dados de treino
        mlflow.log_artifact('../Datasets/dataset_modelagem.csv')

        return model_lr



# -

model_lr = model_lr(X_train, y_train, X_test)

# ## Random Forest

# +
#modelo de reandom forest refressor para prever o tempo de atracação
from sklearn.ensemble import RandomForestRegressor

def model_rf(X_train, y_train, X_test, n_estimators, random_state, oob_score, bootstrap):

    # Criando experimento no MLflow
    my_exp = mlflow.set_experiment('RandomForestRegression')

    with mlflow.start_run(experiment_id=my_exp.experiment_id):

        model_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, oob_score=oob_score, bootstrap=bootstrap)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

        mse, r2 = metrics(y_test, y_pred_rf)

        # Perform cross-validation and get the scores
        scores = cross_val_score(model_rf, X_train, y_train, cv=kfold)

        # Print the scores
        print("Cross-Validation Scores:", scores)
        print("Mean Score:", scores.mean())

        print('R2 Score:', r2)
        print('Mean Squared Error:', mse)
        
        # out-of-bag score
        oob_score = model_rf.oob_score_
        print(f"Out-of-Bag Score: {oob_score}")

        # Logando os parâmetros do modelo
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('oob_score', oob_score)
        mlflow.log_param('bootstrap', bootstrap)
        mlflow.log_metric('Cross-Validation Scores', scores.mean())
        mlflow.log_metric('Out-of-Bag Score', oob_score)
        
        # Logando as métricas do modelo
        mlflow.log_metrics({'mse': mse, 'r2': r2})

        # Logando o modelo
        mlflow.sklearn.log_model(model_rf, 'random_forest_model')

        # Logando os dados de treino
        mlflow.log_artifact('../Datasets/dataset_modelagem.csv')

        return model_rf



# +
n_estimators = 24
random_state = 30
oob_score = True
bootstrap = True

model_rf = model_rf(X_train, y_train, X_test, n_estimators, random_state, oob_score, bootstrap)
# -

# ## XGBoost

# +
from xgboost import XGBRegressor


def model_xg(X_train, y_train, X_test, objective, random_state):

    # Criando experimento no MLflow
    my_exp = mlflow.set_experiment('XGBRegressor')

    with mlflow.start_run(experiment_id=my_exp.experiment_id):

        model_xg = XGBRegressor(objective='reg:squarederror', random_state=42)
        model_xg.fit(X_train, y_train)
        y_pred_xg = model_xg.predict(X_test)

        # Perform cross-validation and get the scores
        scores = cross_val_score(model_xg, X_train, y_train, cv=kfold)

        # Print the scores
        print("Cross-Validation Scores:", scores)
        print("Mean Score:", scores.mean())

        mse, r2 = metrics(y_test, y_pred_xg)
        
        print('R2 Score:', r2)
        print('Mean Squared Error:', mse)

        # Logando os parâmetros do modelo
        mlflow.log_param('objective', objective)
        mlflow.log_param('random_state', random_state)
        mlflow.log_metric('Cross-Validation Scores', scores.mean())

        # Logando as métricas do modelo
        mlflow.log_metrics({'mse': mse, 'r2': r2})

        # Logando o modelo
        mlflow.sklearn.log_model(model_xg, 'XGBRegressor')

        # Logando os dados de treino
        mlflow.log_artifact('../Datasets/dataset_modelagem.csv')

        return model_xg


# +
objective = 'reg:squarederror'
random_state = 42

model_xg = model_xg(X_train, y_train, X_test, objective, random_state)
# -

# ## Multilayer Perceptron

# +
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def model_mlp(X_train, y_train, X_test, hidden_layer_sizes, activation, solver, random_state, max_iter):

    # Copiando os dados
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    y_train_norm = y_train.copy()
    y_test_norm = y_test.copy()

    # Normalizando os dados
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_norm)
    X_test_norm = scaler.transform(X_test_norm)
    y_train_norm = scaler.fit_transform(y_train_norm.values.reshape(-1, 1))
    y_test_norm = scaler.transform(y_test_norm.values.reshape(-1, 1))



    # Criando experimento no MLflow
    my_exp = mlflow.set_experiment('MLP_Regressor')

    with mlflow.start_run(experiment_id=my_exp.experiment_id):

        model_mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, random_state=random_state, max_iter=max_iter)

        model_mlp.fit(X_train_norm, y_train_norm)
        y_pred_mlp = model_mlp.predict(X_test_norm)

        # Perform cross-validation and get the scores
        scores = cross_val_score(model_mlp, X_train_norm, y_train_norm, cv=kfold)

        # Print the scores
        print("Cross-Validation Scores:", scores)
        print("Mean Score:", scores.mean())

        mse, r2 = metrics(y_test_norm, y_pred_mlp)
        
        print('R2 Score:', r2)
        print('Mean Squared Error:', mse)

        # Logando os parâmetros do modelo
        mlflow.log_param('hidden_layer_sizes', hidden_layer_sizes)
        mlflow.log_param('activation', activation)
        mlflow.log_param('solver', solver)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_metric('Cross-Validation Scores', scores.mean())

        # Logando as métricas do modelo
        mlflow.log_metrics({'mse': mse, 'r2': r2})

        # Logando o modelo
        mlflow.sklearn.log_model(model_mlp, 'MLP_Regressor')

        # Logando os dados de treino
        mlflow.log_artifact('../Datasets/dataset_modelagem.csv')

        return model_mlp

# +
hidden_layer_sizes=(50, 50)
activation='relu'
solver='adam'
random_state=24
max_iter=100

model_mlp = model_mlp(X_train, y_train, X_test, hidden_layer_sizes, activation, solver, random_state, max_iter)
# -

# ## Avaliação dos residuos

# +
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def avaliacao_residuos(y_test, y_pred):

    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='blue', palette='viridis', cbar=True )
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    plt.show()

    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Predicted vs Residuals')
    plt.show()

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()



# -

# ### Avaliação residuos Ramdom Forest

y_pred = model_rf.predict(X_test)
avaliacao_residuos(y_test, y_pred)

# ## Avaliação da performance do modelo

# +
# %matplotlib inline
import matplotlib.pyplot as plt

def scatter_plot(y_test, y_pred):
    # Create a scatter plot
    plt.scatter(y_test, y_pred)

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Model Performance - Random Forest')

    # Add a diagonal line for reference
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')

    # Show the plot
    plt.show()


# -

y_pred = model_rf.predict(X_test)
scatter_plot(y_test, y_pred)

# ## Feture importance

# +
importance = model_rf.feature_importances_
feature_names = X_train.columns

# Criar um dataframe com as features e suas importâncias
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Ordenar as features pela importância em ordem decrescente
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Imprimir as features mais importantes
print(feature_importance_df)
# -

# ## Shap 

# +
# lib shap regressor para explicar o modelo
import shap

def shap_lr(model, X_train, X_test):
    # Usar a biblioteca SHAP para explicar o modelo
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Plotar os valores SHAP
    
    shap.summary_plot(shap_values, X_test)

    return shap_values

def shap_rf(model_rf, X_test):
    # explain all the predictions in the test set
    ex = shap.TreeExplainer(model_rf)
    shap_values = ex.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

    return shap_values


# +
shap_lr = shap_lr(model_lr, X_train, X_test)
shap_rf = shap_rf(model_rf, X_test)


# -

# ## Pipeline

# +
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(kfold, model_lr)
pipeline

# -

# ## OLS Regression Results

# +
#https://blog.dailydoseofds.com/p/statsmodel-regression-summary-will

import statsmodels.formula.api as smf
statsmodels = smf.ols('y ~ X', data=df_modelagem)
statsmodels = statsmodels.fit()
print(statsmodels.summary())
