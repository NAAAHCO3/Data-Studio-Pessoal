import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Data Studio Pro V2", layout="wide", page_icon="🧪")

# --- ESTILO CSS PERSONALIZADO ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9;}
    .sub-header {font-size: 1.5rem; color: #404040;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧪 Data Science Workbench v2.0</p>', unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith("csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

# --- SIDEBAR: UPLOAD E ESTADO ---
st.sidebar.header("1. Dados")
uploaded_file = st.sidebar.file_uploader("Carregar Dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Carregamento Inicial
    if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
        df = load_data(uploaded_file)
        st.session_state['df_raw'] = df
        st.session_state['df_work'] = df.copy()
        st.session_state['file_name'] = uploaded_file.name
    
    # Botão Reset
    if st.sidebar.button("🔄 Resetar Dados"):
        st.session_state['df_work'] = st.session_state['df_raw'].copy()
        st.rerun()

    df_work = st.session_state['df_work']
    
    st.sidebar.info(f"Dataset: {df_work.shape[0]} linhas, {df_work.shape[1]} colunas")
    
    # --- MENU PRINCIPAL ---
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Módulos:", 
        ["🔍 EDA & Estatística", "🛠️ Feature Engineering", "🤖 Machine Learning (AutoML)", "📊 Visualização", "💾 Exportar"])

    # ====================================================================
    # MÓDULO 1: EDA (Análise Exploratória)
    # ====================================================================
    if menu == "🔍 EDA & Estatística":
        st.header("Análise Exploratória de Dados")
        
        tab1, tab2, tab3 = st.tabs(["Visão Geral", "Correlações", "Distribuições"])
        
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Linhas", df_work.shape[0])
            c2.metric("Colunas", df_work.shape[1])
            c3.metric("Nulos", df_work.isna().sum().sum())
            c4.metric("Duplicatas", df_work.duplicated().sum())
            
            with st.expander("Ver Amostra dos Dados", expanded=True):
                st.dataframe(df_work.head())
            
            with st.expander("Ver Resumo Estatístico"):
                st.write(df_work.describe())

        with tab2:
            st.subheader("Matriz de Correlação (Heatmap)")
            df_num = df_work.select_dtypes(include=['float64', 'int64'])
            
            if not df_num.empty:
                corr = df_num.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Não há colunas numéricas suficientes para correlação.")

        with tab3:
            st.subheader("Distribuição de Variáveis Numéricas")
            col_dist = st.selectbox("Selecione Coluna:", df_work.select_dtypes(include='number').columns)
            if col_dist:
                fig_dist = px.histogram(df_work, x=col_dist, marginal="box", title=f"Distribuição de {col_dist}")
                st.plotly_chart(fig_dist, use_container_width=True)

    # ====================================================================
    # MÓDULO 2: FEATURE ENGINEERING (Engenharia de Atributos)
    # ====================================================================
    elif menu == "🛠️ Feature Engineering":
        st.header("Preparação de Dados")
        st.markdown("Prepare seus dados para modelos de IA.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Limpeza")
            # Tratamento de Nulos
            cols_na = df_work.columns[df_work.isna().any()].tolist()
            if cols_na:
                st.warning(f"Colunas com Nulos: {cols_na}")
                if st.button("Remover Linhas com Nulos"):
                    df_work = df_work.dropna()
                    st.session_state['df_work'] = df_work
                    st.success("Nulos removidos!")
                    st.rerun()
            else:
                st.success("Sem valores nulos.")

            # Remoção de Colunas
            cols_to_drop = st.multiselect("Excluir Colunas Irrelevantes (IDs, Nomes, etc)", df_work.columns)
            if st.button("Dropar Colunas"):
                df_work = df_work.drop(columns=cols_to_drop)
                st.session_state['df_work'] = df_work
                st.rerun()

        with col2:
            st.subheader("2. Transformação")
            
            # One-Hot Encoding (Dummies)
            cols_cat = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
            if cols_cat:
                st.write(f"Colunas Categóricas detectadas: {cols_cat}")
                col_dummy = st.selectbox("Converter coluna texto para número (One-Hot):", [None] + cols_cat)
                if col_dummy and st.button("Aplicar One-Hot Encoding"):
                    df_work = pd.get_dummies(df_work, columns=[col_dummy], drop_first=True)
                    st.session_state['df_work'] = df_work
                    st.success(f"{col_dummy} convertida!")
                    st.rerun()

            # Normalização
            cols_num = df_work.select_dtypes(include=['float64', 'int64']).columns.tolist()
            col_norm = st.multiselect("Normalizar Colunas (StandardScaler):", cols_num)
            if col_norm and st.button("Aplicar Normalização"):
                scaler = StandardScaler()
                df_work[col_norm] = scaler.fit_transform(df_work[col_norm])
                st.session_state['df_work'] = df_work
                st.success("Normalização aplicada!")
                st.rerun()

    # ====================================================================
    # MÓDULO 3: MACHINE LEARNING (AUTO ML)
    # ====================================================================
    elif menu == "🤖 Machine Learning (AutoML)":
        st.header("Treinamento de Modelo Automático")
        
        # Seleção de Variáveis
        target = st.selectbox("Selecione a Variável Alvo (O que você quer prever?):", df_work.columns)
        
        # Identificação do Problema (Regressão ou Classificação)
        is_numeric = pd.api.types.is_numeric_dtype(df_work[target])
        num_unique = df_work[target].nunique()
        
        # Heurística simples para decidir tipo de modelo
        problem_type = "Regressão" if is_numeric and num_unique > 10 else "Classificação"
        st.info(f"Problema detectado: **{problem_type}**")

        features = st.multiselect("Selecione as Features (Variáveis X):", [c for c in df_work.columns if c != target], default=[c for c in df_work.columns if c != target])

        if st.button("🚀 Treinar Modelo"):
            if not features:
                st.error("Selecione pelo menos uma feature.")
            else:
                try:
                    # Preparação
                    X = df_work[features]
                    y = df_work[target]
                    
                    # Trata categóricas restantes (Label Encoding simples para o alvo se for classificação)
                    if problem_type == "Classificação" and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)

                    # Verifica se X tem texto (erro comum)
                    if X.select_dtypes(include=['object']).shape[1] > 0:
                        st.error("Erro: As features selecionadas contêm texto. Use a aba 'Feature Engineering' para converter em números primeiro.")
                    else:
                        # Split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        st.markdown("---")
                        
                        if problem_type == "Regressão":
                            model = RandomForestRegressor(n_estimators=100)
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            
                            # Métricas
                            r2 = r2_score(y_test, preds)
                            mae = mean_absolute_error(y_test, preds)
                            
                            col1, col2 = st.columns(2)
                            col1.metric("R² Score (Precisão)", f"{r2:.2%}")
                            col2.metric("Erro Médio Absoluto", f"{mae:.2f}")
                            
                            # Gráfico Real vs Previsto
                            fig_res = px.scatter(x=y_test, y=preds, labels={'x': 'Real', 'y': 'Previsto'}, title="Real vs Previsto")
                            fig_res.add_shape(type="line", line=dict(dash="dash"), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                            st.plotly_chart(fig_res)
                            
                        else: # Classificação
                            model = RandomForestClassifier(n_estimators=100)
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            
                            acc = accuracy_score(y_test, preds)
                            st.metric("Acurácia", f"{acc:.2%}")
                            st.text("Relatório de Classificação:")
                            st.text(classification_report(y_test, preds))

                        # Feature Importance
                        st.subheader("Importância das Variáveis")
                        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                        imp_df = imp_df.sort_values(by='Importance', ascending=False)
                        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig_imp)

                except Exception as e:
                    st.error(f"Erro no treinamento: {e}")

    # ====================================================================
    # MÓDULO 4: VISUALIZAÇÃO (Igual versão anterior)
    # ====================================================================
    elif menu == "📊 Visualização":
        st.header("Visualização Livre")
        all_cols = df_work.columns.tolist()
        c1, c2, c3 = st.columns(3)
        tipo = c1.selectbox("Tipo", ["Dispersão", "Linha", "Barra", "Histograma", "Boxplot"])
        x = c2.selectbox("X", all_cols)
        y = c3.selectbox("Y", all_cols)
        
        if st.button("Gerar"):
            if tipo == "Dispersão": fig = px.scatter(df_work, x=x, y=y)
            elif tipo == "Linha": fig = px.line(df_work, x=x, y=y)
            elif tipo == "Barra": fig = px.bar(df_work, x=x, y=y)
            elif tipo == "Histograma": fig = px.histogram(df_work, x=x)
            elif tipo == "Boxplot": fig = px.box(df_work, x=x, y=y)
            st.plotly_chart(fig)

    # ====================================================================
    # MÓDULO 5: EXPORTAR
    # ====================================================================
    elif menu == "💾 Exportar":
        st.header("Baixar Dataset Atualizado")
        csv = df_work.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "dados_v2.csv", "text/csv")

else:
    st.info("Por favor, carregue um arquivo CSV ou Excel para iniciar a análise.")