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

from src.config import DADOS_ORIGINAIS, DADOS_TRATADOS

sns.set_theme(palette="bright")

pd.set_option("display.max_info_columns", 100)   
pd.set_option("display.max_columns", 100)        
pd.set_option("display.width", 200)              

# %% [markdown]
# # Dataframe

# %%
df = pd.read_csv(DADOS_ORIGINAIS)

df.head()

# %% [markdown]
# # Exploratory Analysis

# %% [markdown]
# ## Controle de contexto

# %%
with pd.option_context("float_format", "{:.2f}".format, "display.max_columns", 35):
    display(df.describe())

# %%
df.info()

# %%
df.describe(exclude="number")

# %%
df.nunique()[df.nunique() == 1]

# %%
df = df.drop(columns=df.nunique()[df.nunique() == 1].index)
df.info()

# %% [markdown]
# ## Verificando a coluna target do projeto

# %%
df["Attrition"].value_counts()

# %%
df["Attrition"].value_counts(normalize=True)

# %% [markdown]
# ### A coluna target do projeto tem pouca variação

# %% [markdown]
# ## Verificação de valores nulos e duplicidade

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %% [markdown]
# ## Verificar se a coluna de identificação tem valores unicos

# %%
df["EmployeeNumber"].nunique()

# %%
df = df.drop(["EmployeeNumber"], axis=1)
df.info()

# %% [markdown]
# ## Separando as colunas categoricas entre ordenadas e não ordenadas

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

coluna_alvo = ["Attrition"]

# %% [markdown]
# ## Colunas numéricas

# %%
colunas_numericas = [
    coluna for coluna in df.columns if coluna not in (
        colunas_categoricas_nao_ordenadas + colunas_categoricas_ordenadas + coluna_alvo
    )
]

# %%
print(colunas_numericas)

# %%
len(colunas_numericas)

# %% [markdown]
# ## Gráficos de exploração dos dados

# %% [markdown]
# ### Histograma das colunas numéricas

# %%
fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(20, 10))

for ax, coluna in zip(axs.flatten(), colunas_numericas):
    sns.histplot(data=df, x=coluna, ax=ax, kde=True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Boxplots

# %% [markdown]
# #### Idenficação de outliers

# %%
fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(20, 10))

for ax, coluna in zip(axs.flatten(), colunas_numericas):
    sns.boxplot(data=df, x=coluna, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Verificar diferenças entre as duas categorias nas colunas numericas

# %%
fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(20, 10))

for ax, coluna in zip(axs.flatten(), colunas_numericas):
    sns.boxplot(data=df, x=coluna, ax=ax, showmeans=True, hue="Attrition")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Análise de correlação entre as colunas (features)

# %%
corr = df.select_dtypes(include=["number"]).corr()

matriz = np.triu(corr)

fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(
    corr,
    mask=matriz,
    annot=True,
    fmt=".2f",
    ax=ax,
    annot_kws={"size": 9},
    cmap="coolwarm",
)

plt.show()

# %% [markdown]
# ## Melhoria/ Otimização do tamanho do dataframe

# %%
colunas_valores_inteiros = [coluna for coluna in df.select_dtypes("int")]

print(colunas_valores_inteiros)

# %%
df[colunas_valores_inteiros] = df[colunas_valores_inteiros].apply(
    pd.to_numeric, downcast="integer"
)

df.info()

# %% [markdown]
# Redução no número utilizado pela memória de ~400 KB para ~130 KB

# %% [markdown]
# # Salvando em parquet

# %%
df.to_parquet(DADOS_TRATADOS, index=False)
