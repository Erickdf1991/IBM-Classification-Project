# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ibm_project (3.12.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer # para aplicar transformações específicas em colunas
from sklearn.metrics import ConfusionMatrixDisplay # para exibir a matriz de confusão
from sklearn.model_selection import StratifiedKFold # para validação cruzada estratificada, visto que nosso alvo é desbalanceado
# Preprocessamentos
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
) # para normalização, padronização e transformação de potência

# classificador referência
from sklearn.dummy import DummyClassifier

# estudo lineares
from sklearn.linear_model import LogisticRegression

# estudo árvores
from sklearn.tree import DecisionTreeClassifier # para estudo de árvores de decisão
from lightgbm import LGBMClassifier # para estudo de gradient boosting com LightGBM
from xgboost import XGBClassifier # para estudo de gradient boosting com XGBoost

# estudo SVM
from sklearn.svm import SVC # para estudo de máquinas de vetor de suporte (SVM)

# estudo kNN
from sklearn.neighbors import KNeighborsClassifier # para estudo de k-vizinhos mais próximos (kNN)

# Exportar o modelo
from joblib import dump

from src.auxiliares import dataframe_coeficientes
from src.config import DADOS_TRATADOS, MODELO_FINAL
from src.graficos import plot_comparar_metricas_modelos, plot_coeficientes
from src.models_rus import RANDOM_STATE
from src.models_rus import (
    grid_search_cv_classificador,
    treinar_e_validar_modelo_classificacao,
    organiza_resultados,
)

# %% [markdown]
# Como a base é desbalanceada, foi utilizado o Under Sampler, é possível ver dentro do models_rus

# %% [markdown]
# # Parameters and Dataframe

# %%
df = pd.read_parquet(DADOS_TRATADOS)

df.head()

# %%
colunas_categoricas_nao_ordenadas = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# %%
colunas_categoricas_ordenadas = [
    "Education",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]

# %%
coluna_alvo = ["Attrition"]

# %%
colunas_numericas = [
    coluna for coluna in df.columns if coluna not in (
        colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas + coluna_alvo
    )
]

print(colunas_numericas)

# %% [markdown]
# # Tratando as colunas numéricas

# %% [markdown]
# Algumas features, conforme visto nos nosso gráficos de histograma, não possuem distribuição normal. E podemos aplicar um Min Max Scaler, onde é feito um ajuste de escala, sem impactar na distribuição delas.

# %%
colunas_numericas_min_max = [
    "DailyRate",
    "HourlyRate",
    "MonthlyRate",
]

# %% [markdown]
# Colunas como Age possuem uma boa assimetria

# %%
colunas_numericas_std = ["Age"]

colunas_numericas_power_transform = [coluna for coluna in colunas_numericas if coluna not in (colunas_numericas_min_max + colunas_numericas_std)]

print(colunas_numericas_power_transform)

# %% [markdown]
# # Separação de X (features) e y (target)

# %%
X = df.drop(columns=coluna_alvo)
y = df[coluna_alvo]

# %% [markdown]
# ## Aplicando Label Encoder na coluna alvo para transformar em numero o "yes" e "no"

# %%
le = LabelEncoder()

y = le.fit_transform(y.values.ravel())

y[:10]

# %%
le.classes_

# %% [markdown]
# ## Instanciando o Kfold

# %%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %% [markdown]
# # Pre Processamento

# %% [markdown]
# preprocessamento_arvores: focado em **modelos baseados em árvore** (não exige escala/padronização numérica).

# %%
preprocessamento_arvores = ColumnTransformer(
    transformers=[
        ("one_hot", OneHotEncoder(drop="first"), colunas_categoricas_nao_ordenadas),
        ("ordinal", OrdinalEncoder(categories="auto"), colunas_categoricas_ordenadas),     
    ]
)

# %% [markdown]
# preprocessamento: focado em **modelos sensíveis à escala** (regressão, SVM, KNN, redes neurais etc.), incluindo transformações numéricas.

# %%
preprocessamento = ColumnTransformer(
    transformers=[
        ("one_hot", OneHotEncoder(drop="first"), colunas_categoricas_nao_ordenadas),
        ("ordinal", OrdinalEncoder(categories="auto"), colunas_categoricas_ordenadas), 
        ("min_max_scaler", MinMaxScaler(), colunas_numericas_min_max),
        ("stdscaler", StandardScaler(), colunas_numericas_std),
        ("power_transformer", PowerTransformer(), colunas_numericas_power_transform),
    ]
)

# %% [markdown]
# ## Dicionário de Classificadores

# %%
classificadores = {
    "DummyClassifier": {
        "preprocessor": None, 
        "classificador": DummyClassifier(strategy="stratified"),
    },
    "LogisticRegression": {
        "preprocessor": preprocessamento, 
        "classificador": LogisticRegression(),
    },
    "DecisionTreeClassifier": {
        "preprocessor": preprocessamento_arvores, 
        "classificador": DecisionTreeClassifier(),
    },
    "LGBMClassifier": {
        "preprocessor": preprocessamento_arvores, 
        "classificador": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            ),
    },
    "XGBClassifier": {
        "preprocessor": preprocessamento_arvores, 
        "classificador": XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            ),
    },
    "SVC": {
        "preprocessor": preprocessamento, 
        "classificador": SVC(),
    },
    "KNeighborsClassifier": {
        "preprocessor": preprocessamento, 
        "classificador": KNeighborsClassifier(),
    },
}

# %% [markdown]
# # Dataframe de Resultados

# %%
resultados = {
    nome_modelo: treinar_e_validar_modelo_classificacao(X, y, kf, **classificador)
    for nome_modelo, classificador in classificadores.items()
}

df_resultados = organiza_resultados(resultados)

df_resultados

# %% [markdown]
# ## Group by por modelos

# %%
df_resultados.groupby("model").mean().sort_values("test_average_precision")

# %% [markdown]
# ### Gráfico

# %%
plot_comparar_metricas_modelos(df_resultados)

# %% [markdown]
# Escolha do modelo (Logistic Regression)
# Após avaliar múltiplos classificadores com validação cruzada e métricas adequadas para classes desbalanceadas (ex.: balanced accuracy, F1, AUROC e principalmente AUPRC), optei pela Regressão Logística como modelo final. Além de apresentar desempenho competitivo e consistente entre os folds, a Regressão Logística oferece alta interpretabilidade (coeficientes permitem entender o impacto das variáveis no risco), treinamento rápido, e uma saída probabilística naturalmente adequada para ajuste de limiar (threshold) conforme o objetivo do problema (priorizar recall ou precision). Para mitigar o desbalanceamento da variável alvo, foi utilizado class_weight="balanced", garantindo maior penalização para erros na classe minoritária.

# %% [markdown]
# # Grid Search - Otimização do modelo escolhido (Logistic Regression)

# %% [markdown]
# ## Grade de Parametros

# %%
param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2", "elasticnet", None],  # Adicionando a possibilidade de usar Elastic Net
    "clf__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],  
}

# %%
clf = LogisticRegression(solver="saga", random_state=RANDOM_STATE)
                         
grid_search = grid_search_cv_classificador(
    clf, param_grid, kf, preprocessamento, refit_metric="average_precision",
)

grid_search

# %%
grid_search.fit(X, y)

# %% [markdown]
# ## Grid Search - Melhores Parâmetros

# %%
grid_search.best_params_

# %%
grid_search.best_score_

# %%
grid_search.cv_results_.keys()

# %%
grid_search.best_estimator_

# %%
grid_search.best_index_

# %%
grid_search.cv_results_["mean_test_average_precision"][grid_search.best_index_]

# %%
grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_]

# %%
colunas_test = [coluna for coluna in df_resultados.columns if coluna.startswith("test_")]

colunas_test

# %%
colunas_test_mean = ["mean_" + coluna for coluna in colunas_test]
colunas_test_mean

# %%
for coluna in colunas_test_mean:
    valor = grid_search.cv_results_[coluna][grid_search.best_index_]
    print(f"{coluna}: {valor:.2f}")


# %% [markdown]
# # Coeficientes

# %%
coefs = dataframe_coeficientes(
    grid_search.best_estimator_["clf"].coef_[0],
    grid_search.best_estimator_["preprocessor"].get_feature_names_out()
)

coefs

# %%
coefs.query("coeficiente == 0")

# %% [markdown]
# Indica que essas features (coeficientes) podem não ter relação com o fato de a pessoa sair ou não da empresa, ou seja, uma pessoa que tem um percentual alto de PerformanceRating, não necessáriamente sairá da empresa, ou o inverso.

# %%
plot_coeficientes(coefs.query("coeficiente != 0"))

# %% [markdown]
# Esse gráfico dá uma direção interessante do que influência positivamente ou negativamente para que uma pessoa saia da empresa. Exemplo: Fazer muita hora extra ou viajar muito a trabalho, são reclamações que nós ouvimos muito em ambientes corporativos.
# Assim como uma pessoa que tem muito envolvimento com o trabalho, um salário alto, tem maior chances de permanecer na empresa.
# Alguns aspectos são mais subgetivos e devem ser colocados e análisados com a lente do contexto ao qual esses funcionários estão inseridos, como Cultura, por exemplo.

# %% [markdown]
# Lembre-se Correlação não implica em Causalidade

# %% [markdown]
# # Facilitando a interpretação dos Coeficientes no resultado do modelo

# %%
coefs_odds = coefs.copy()
coefs_odds["coeficiente"] = np.exp(coefs_odds["coeficiente"])

coefs_odds

# %%
plot_coeficientes(coefs_odds.query("coeficiente != 1"))

# %% [markdown]
# Com a aplicação do exponencial das odds, temos o seguinte cenário onde fazer hora extra aumenta em quase 3 (2,67) vezes as chances de o funcionário sair da empresa.

# %% [markdown]
# ## Matriz de Confusão

# %%
ConfusionMatrixDisplay.from_estimator(
    grid_search.best_estimator_, 
    X, 
    y, 
    display_labels=le.classes_
)

plt.grid(False)
plt.show()

# %%
ConfusionMatrixDisplay.from_estimator(
    grid_search.best_estimator_, 
    X, 
    y, 
    display_labels=le.classes_,
    normalize="true"
)

plt.grid(False)
plt.show()

# %% [markdown]
# **Plano de ação**
#
# - Avaliar os motivos que levam os funcionários a fazerem hora extra.
#   - Mão de obra insuficiente
#   - Falta de organização institucional
#   - Falta de treinamento
#   - Necessidade de investimento em tecnologia
# - Possibilidade de diminuir as viagens de negócios.
#   - Como isso afeta a equipe de vendas?
# - Como a renda mensal de cada setor e nível da empresa se compara com o mercado?

# %% [markdown]
# # Exportando o modelo

# %%
dump(grid_search.best_estimator_, MODELO_FINAL)
