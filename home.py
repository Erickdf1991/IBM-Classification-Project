import pandas as pd
import streamlit as st

from joblib import load

from src.config import DADOS_TRATADOS, MODELO_FINAL

@st.cache_data
def carregar_dados():
    return pd.read_parquet(DADOS_TRATADOS)

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

df = carregar_dados()
modelo = carregar_modelo()

niveis_educacionais_texto = {
    1: "Abaixo do ensino superior",
    2: "Ensino Superior",
    3: "Bacharel",
    4: "Mestre",
    5: "PhD",
}

niveis_satisfacao_texto = {
    1: "Baixo",
    2: "Médio",
    3: "Alto",
    4: "Muito Alto",
}

niveis_vida_trabalho_texto = {
    1: "Ruim",
    2: "Bom",
    3: "Muito Bom",
    4: "Excelente",
}

#Variaveis com colunas categóricas/texto
generos = sorted(df['Gender'].unique())
niveis_educacionais = sorted(df['Education'].unique())
area_formacao = sorted(df['EducationField'].unique())
departamentos = sorted(df['Department'].unique())
viagem_negocios = sorted(df['BusinessTravel'].unique())
hora_extra = sorted(df['OverTime'].unique())
satisfacao_trabalho = sorted(df['JobSatisfaction'].unique())
satisfacao_colegas = sorted(df['RelationshipSatisfaction'].unique())
satisfacao_ambiente = sorted(df['EnvironmentSatisfaction'].unique())
vida_trabalho = sorted(df['WorkLifeBalance'].unique())
opcao_acoes = sorted(df['StockOptionLevel'].unique())
envolvimento_trabalho = sorted(df['JobInvolvement'].unique())

colunas_slider = [
    "DistanceFromHome",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

colunas_slider_min_max = {
    coluna: {"min_value": df[coluna].min(), "max_value": df[coluna].max()} for coluna in colunas_slider
}

colunas_ignoradas = [
    "Age",
    "DailyRate",
    "JobLevel",
    "HourlyRate",
    "MonthlyRate",
    "PerformanceRating",
]

mediana_colunas_ignoradas = {
    coluna: df[coluna].median() for coluna in colunas_ignoradas
}

st.title('Previsão de Atrito')

with st.container(border=True):
    st.write("### Informações pessoais")

    widget_genero = st.radio("Gênero", options=generos)
    widget_nivel_educacional = st.selectbox(
        "Nível Educacional", 
        options=niveis_educacionais, 
        format_func=lambda numero: niveis_educacionais_texto[numero]
    )
    widget_area_formacao = st.selectbox("Área de Formação", options=area_formacao)
    widget_distancia_casa = st.slider(
        "Distância de casa", **colunas_slider_min_max["DistanceFromHome"]
    )


with st.container(border=True):
    st.write("### Rotina na empresa")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:

        widget_departamento = st.selectbox("Departamento", options=departamentos)
        widget_viagem_negocios = st.selectbox("Viagem a negócios", options=viagem_negocios)
    
    with coluna_direita:
        widget_cargo = st.selectbox(
            "Cargo", 
            options=sorted(df[df["Department"] == widget_departamento]["JobRole"].unique())
        )

        widget_horas_extras = st.radio("Horas Extras", options=hora_extra)

    widget_salario_mensal = st.slider(
        "Salário Mensal", **colunas_slider_min_max["MonthlyIncome"]
    )

with st.container(border=True):
    st.write("### Experiência profissional")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        widget_empresas_trabalhadas = st.slider(
            "Empresas trabalhadas", 
            **colunas_slider_min_max["NumCompaniesWorked"]
        )

        widget_anos_trabalhados = st.slider(
            "Anos trabalhados", 
            **colunas_slider_min_max["TotalWorkingYears"]
        )

        widget_anos_empresa = st.slider(
            "Anos na Empresa", 
            **colunas_slider_min_max["YearsAtCompany"]
        )
    
    with coluna_direita:
        widget_anos_cargo_atual = st.slider(
            "Anos no cargo atual", 
            **colunas_slider_min_max["YearsInCurrentRole"]
        )

        widget_anos_mesmo_gerente = st.slider(
            "Anos com o mesmo gerente", 
            **colunas_slider_min_max["YearsWithCurrManager"]
        )

        widget_anos_ultima_promocao = st.slider(
            "Anos desde a última promoção", 
            **colunas_slider_min_max["YearsSinceLastPromotion"]
        )

with st.container(border=True):
    st.write("### Incentivos e métricas")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        widget_satisfacao_trabalho = st.selectbox(
            "Satisfação com o trabalho", 
            options=satisfacao_trabalho, 
            format_func=lambda numero: niveis_satisfacao_texto[numero]
        )

        widget_satisfacao_colegas = st.selectbox(
            "Satisfação com os colegas", 
            options=satisfacao_colegas, 
            format_func=lambda numero: niveis_satisfacao_texto[numero]
        )

        widget_envolvimento_trabalho = st.selectbox(
            "Envolvimento com o trabalho", 
            options=envolvimento_trabalho,
        )

    
    with coluna_direita:
        widget_satisfacao_ambiente = st.selectbox(
            "Satisfação com o ambiente de trabalho", 
            options=satisfacao_ambiente, 
            format_func=lambda numero: niveis_satisfacao_texto[numero]
        )

        widget_vida_trabalho = st.selectbox(
            "Equilíbrio vida-trabalho", 
            options=vida_trabalho, 
            format_func=lambda numero: niveis_vida_trabalho_texto[numero]
        )

        widget_opcao_acoes = st.radio(
            "Opção de ações", 
            options=opcao_acoes,
        )
    
    widget_aumento_salarial = st.slider(
        "Aumento salarial (%)", 
        **colunas_slider_min_max["PercentSalaryHike"]
    )

    widget_treinamentos_ultimo_ano = st.slider(
        "Treinamentos no último ano", 
        **colunas_slider_min_max["TrainingTimesLastYear"]
    )


entrada_modelo = {
    "Age": mediana_colunas_ignoradas["Age"],
    "BusinessTravel": widget_viagem_negocios,
    "DailyRate": mediana_colunas_ignoradas["DailyRate"],
    "Department": widget_departamento,
    "DistanceFromHome": widget_distancia_casa,
    "Education": widget_nivel_educacional,
    "EducationField": widget_area_formacao,
    "EnvironmentSatisfaction": widget_satisfacao_ambiente,
    "Gender": widget_genero,
    "HourlyRate": mediana_colunas_ignoradas["HourlyRate"],
    "JobInvolvement": widget_envolvimento_trabalho,
    "JobLevel": mediana_colunas_ignoradas["JobLevel"],
    "JobRole": widget_cargo,
    "JobSatisfaction": widget_satisfacao_trabalho,
    "MaritalStatus": "Single",
    "MonthlyIncome": widget_salario_mensal,
    "MonthlyRate": mediana_colunas_ignoradas["MonthlyRate"],
    "NumCompaniesWorked": widget_empresas_trabalhadas,
    "PerformanceRating": mediana_colunas_ignoradas["PerformanceRating"],
    "OverTime": widget_horas_extras,
    "PercentSalaryHike": widget_aumento_salarial,
    "RelationshipSatisfaction": widget_satisfacao_colegas,
    "StockOptionLevel": widget_opcao_acoes,
    "TotalWorkingYears": widget_anos_trabalhados,
    "TrainingTimesLastYear": widget_treinamentos_ultimo_ano,
    "WorkLifeBalance": widget_vida_trabalho,
    "YearsAtCompany": widget_anos_empresa,
    "YearsInCurrentRole": widget_anos_cargo_atual,
    "YearsSinceLastPromotion": widget_anos_ultima_promocao,
    "YearsWithCurrManager": widget_anos_mesmo_gerente,
}

df_entrada_modelo = pd.DataFrame([entrada_modelo])

botao_previsao = st.button("Prever Atrito")

if botao_previsao:
    previsao = modelo.predict(df_entrada_modelo)[0]
    probabilidade_atrito = modelo.predict_proba(df_entrada_modelo)[0][1]

    cor = ":red" if previsao == 1 else ":green"

    texto_probabilidade = (
        f"#### Probabilidade de Atrito: {cor}[{probabilidade_atrito:.2%}]"
    )

    texto_atrito = (
        f"#### Atrito: {cor}[{'Sim' if previsao == 1 else 'Não'}]"
    )

    st.markdown(texto_atrito)
    st.markdown(texto_probabilidade)