# ML-Flow Smart Berth Planning Documentation

Keywords: Berth Planning, Maritime logistics, SmartPorts, Port optimization, Berth
allocation, Machine Learning

# Índice

1. [Trabalho Realizado](#trabalho-realizado-até-o-momento)
2. [Pipeline dos dados e modelo](#pipeline)
   - [Bibliotecas e Dependências](#bibliotecas-e-dependências)
   - [Data Engineering](#data-engineering)
     - [Dataset Loading](#dataset-loading)
     - [Feature Selection and Treatment](#feature-selection-and-treatment)
     - [Logarithmic Transformation](#logarithmic-transformation)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
     - [YData Profiling](#ydata-profiling)
   - [Model Preparation](#model-preparation)
     - [Train-Test Split](#train-test-split)
     - [K-Fold Cross-Validation](#k-fold-cross-validation)
   - [Model Evaluation Metrics](#model-evaluation-metrics)
   - [Machine Learning Models](#machine-learning-models)
     - [Linear Regression](#linear-regression)
     - [Random Forest Regression](#random-forest-regression)
     - [XGBoost Regression](#xgboost-regression)
     - [Multilayer Perceptron (MLP) Regression](#multilayer-perceptron-mlp-regression)
   - [Model Analysis](#model-analysis)
     - [Residual Analysis](#residual-analysis)
     - [Performance Evaluation](#performance-evaluation)
     - [Feature Importance](#feature-importance)
     - [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
   - [Pipeline](#pipeline)
   - [OLS Regression Results](#ols-regression-results)

3. [Trabalho Futuro](#trabalho-futuro)



# Trabalho realizado até o momento:

## Análise Bibliométrica sobre Planejamento de Berços Inteligentes

- Conduzida uma revisão abrangente da literatura sobre planejamento de berços inteligentes
- Identificados os principais temas, autores influentes e artigos no campo
- Analisada a evolução dos tópicos de pesquisa ao longo do tempo
- Criadas representações visuais dos dados bibliométricos (por exemplo, redes de co-citação, mapas de co-ocorrência de palavras-chave)

## Mapeamento de Plataformas de dados de Navios e Portos

- Identificadas e catalogadas as principais fontes de dados para informações de navios e portos
- Avaliada a qualidade, confiabilidade e acessibilidade de cada plataforma de dados
- Criada uma análise comparativa das diferentes plataformas, destacando seus pontos fortes e limitações

| Plataforma       | Cobertura Global | Precisão dos Dados | Acesso Gratuito | Funcionalidades Avançadas | Dados em Tempo Real | Dados Históricos |
|------------------|------------------|--------------------|-----------------|--------------------------|---------------------|------------------|
| MarineTraffic    | Sim              | Alta               | Sim             | Pago                     | Sim                 | Sim              |
| VesselFinder     | Sim              | Boa                | Sim             | Pago                     | Sim                 | Limitado         |
| FleetMon         | Sim              | Alta               | Sim             | Pago                     | Sim                 | Sim              |
| ShipFix          | Não (foco em frete) | Alta               | Não             | Pago                     | Sim                 | Sim              |
| PortInfo         | Parcial          | Moderada           | Sim             | Variável                 | Não                 | Não              |


## Processo ETL para Histórico dos Navios no porto em um terminato período 

- Projetado e implementado um pipeline de Extração, Transformação e Carga (ETL) para dados históricos do navios
- Dados limpos e padronizados de várias fontes
- Tratados valores ausentes e outliers no conjunto de dados

## Processo ETL para Características de Navios Baseado no IMO Vessels 

- Desenvolvido um processo ETL para extrair características de navios utilizando números da Organização Marítima Internacional (IMO)
- Criada uma base de dados de perfis de navios
- Integração desses dados com os dados históricos dos navios no porto

## Análise Exploratória de Dados (EDA) dos Dados Coletados

- Realizada análise estatística para entender distribuições e relações dos dados
- Criadas visualizações para identificar padrões, tendências e anomalias nos dados
- Conduzida análise de correlação entre diferentes variáveis
- Identificadas possíveis características para modelos de aprendizado de máquina

## Modelagem de Algoritmos de Aprendizado de Máquina

- Implementados e comparados múltiplos algoritmos de aprendizado de máquina:
  - Regressão Logística
  - Random Forest
  - Perceptron Multicamadas (Rede Neural)
- Realizado ajuste de hiperparâmetros para cada modelo
- Avaliados os modelos usando técnicas de validação cruzada

## Explicação e Análise de Importância de Características

- Utilizadas técnicas como valores SHAP (SHapley Additive exPlanations) para interpretar decisões dos modelos
- Características classificadas com base na sua importância nas previsões
- Criadas visualizações para explicar os impactos das características nos resultados dos modelos

## Implementação de Métricas de Avaliação

- Implementado um conjunto abrangente de métricas de avaliação, incluindo:
  - Erro Quadrático Médio (MSE), R-quadrado (R2) para tarefas de regressão

## Integração do MLflow para Rastreamento de Experimentos

- Configurado o MLflow para rastrear e gerenciar experimentos de aprendizado de máquina
- Registrados parâmetros de modelos, métricas e artefatos para cada execução
- Implementado controle de versão para modelos e conjuntos de dados
- Criado um painel para fácil comparação de diferentes experimentos


## Pipeline 

Os notebooks desse repositório demonstra um fluxo de trabalho de machine learning para prever o tempo que um navio passa atracado em um porto.  Inclui carregamento de dados, pré-processamento, análise exploratória de dados, treinamento de modelo e avaliação usando várias técnicas de regressão.

## Bibliotecas e Dependências

O script usa as seguintes bibliotecas principais:
- pandas
- numpy
- matplotlib
- mlflow
- scikit-learn
- xgboost
- shap
- statsmodels

Bibliotecas adicionais usadas para visualizações específicas:
- ptitprince (for RainCloud plots)
- ydata_profiling (for data profiling)

## Data Engineering

### Dataset Loading

O conjunto de dados é carregado a partir de um arquivo CSV denominado 'dataset_modelagem.csv' localizado no diretório '../Datasets/'.

### Feature Selection and Treatment

Os recursos selecionados incluem:
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

Linhas com valores nulos são removidas.  Variáveis ​​categóricas são codificadas usando LabelEncoder.

### Logarithmic Transformation

Os recursos 'Time At Berth' e 'Time At Port' são transformados em log.  Linhas com valores infinitos em 'Time At Berth' são removidas.

## Exploratory Data Analysis

### YData Profiling

O script gera um perfil de dados abrangente usando YData Profiling.


## Model Preparation

### Train-Test Split

Os dados são divididos em conjuntos de treinamento e teste com uma proporção de 80-20.

### K-Fold Cross-Validation

A validação cruzada de 5 vezes é usada para avaliação do modelo.

## Model Evaluation Metrics

Duas métricas principais são usadas:
1. Mean Squared Error (MSE)
2. R-squared (R2) Score

## Machine Learning Models

### Linear Regression

Um modelo de regressão linear simples é implementado usando LinearRegression do scikit-learn.

### Random Forest Regression

Um regressor Random Forest é implementado com os seguintes parâmetros:
- n_estimators: 24
- random_state: 30
- oob_score: True
- bootstrap: True

### XGBoost Regression

Um regressor XGBoost é implementado com os seguintes parâmetros:
- objective: 'reg:squarederror'
- random_state: 42

### Multilayer Perceptron (MLP) Regression

Um regressor MLP é implementado com os seguintes parâmetros:
- hidden_layer_sizes: (50, 50)
- activation: 'relu'
- solver: 'adam'
- random_state: 24
- max_iter: 100

Os dados são normalizados usando StandardScaler antes de treinar o modelo MLP.

## Model Analysis

### Residual Analysis

A análise residual é realizada usando:
- Histogram of residuals
- Scatter plot of predicted vs residuals
- Q-Q plot

### Performance Evaluation

Um gráfico de dispersão de valores reais versus valores previstos é criado para visualizar o desempenho do modelo.

### Feature Importance

A importância do recurso é calculada e exibida para o modelo Random Forest.

### SHAP (SHapley Additive exPlanations)

Os valores SHAP são calculados e plotados para modelos de regressão linear e floresta aleatória para explicar os impactos dos recursos.

## Pipeline

Um pipeline simples é criado combinando a validação cruzada K-Fold com o modelo de regressão linear.

## OLS Regression Results

A regressão de mínimos quadrados ordinários (OLS) é realizada usando modelos estatísticos e um resumo dos resultados é impresso.



**As etapas descritas compreendem desde a preparação de dados até a avaliação e interpretação do modelo.  Foi utilizado a lib MLflow para rastreamento e registro de experimentos, permitindo reprodutibilidade e comparação de diferentes modelos e parâmetros.**



# Trabalho Futuro

## Expansão do Volume e Fontes de Dados

- Negociar acesso a conjuntos de dados maiores e mais abrangentes de autoridades portuárias e empresas de transporte marítimo
- Integrar fluxos de dados em tempo real do AIS (Sistema de Identificação Automática) para rastreamento de navios 
- Explorar possibilidades de incorporação de dados meteorológicos e indicadores econômicos

## Criação e Gestão de Data Lake

- Projetar e implementar uma arquitetura escalável de data lake
- Configurar pipelines de ingestão de dados para atualizações contínuas
- Implementar políticas de governança de dados e controles de acesso
- Integrar monitoramento de qualidade de dados e processos de limpeza automatizados
- Explorar tecnologias como Databricks, Apache Hadoop ou Amazon S3 para armazenamento e recuperação eficiente de dados

## Desenvolvimento de API para Implantação de Modelos

- Projetar uma API RESTful para previsões de modelos
- Implementar autenticação e limitação de taxa para acesso à API
- Configurar monitoramento e registro de uso e desempenho da API

## Interface Web para Visualização de Dados e Previsões

- Projetar uma interface web para analisar os dados das embarcações e do porto
- Criar uma interface de previsão onde os usuários possam inserir parâmetros e receber saídas dos modelos
- Implementar autenticação de usuário e controle de acesso baseado em funções
- Desenvolver recursos para geração de relatórios personalizados e exportação de dados

## Otimização de Modelos e Técnicas Avançadas

- Explorar métodos de ensemble para combinar previsões de múltiplos modelos
- Investigar o uso de modelos de deep learning para reconhecimento de padrões complexos
- Desenvolver um sistema para re-treinamento e implantação automatizada de modelos

## Ferramentas de Simulação e Análise de Cenários

- Desenvolver um motor de simulação para modelar diferentes cenários de alocação de berços
- Criar ferramentas para planejamento de capacidade e identificação de gargalos
- Implementar recursos de análise de sensibilidade para entender o impacto de vários fatores na eficiência dos berços

## Integração com Sistemas de Gestão Portuária

- Desenvolver interfaces para integrar o sistema de planejamento de berços inteligentes com o software de gestão portuária existente
- Implementar protocolos de troca de dados em tempo real para atualizações
- Criar sistemas de alertas e notificações para possíveis problemas ou otimizações

## Análise de Sustentabilidade e Impacto Ambiental

- Integrar dados ambientais no modelo para prever e minimizar o impacto ecológico das operações de berço
- Desenvolver recursos para otimizar a eficiência de combustível e reduzir emissões
- Criar ferramentas de relatórios para conformidade ambiental e métricas de sustentabilidade


