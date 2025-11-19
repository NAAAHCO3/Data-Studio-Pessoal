import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURAÇÃO GERAL ---
st.set_page_config(
    page_title="Data Studio Enterprise", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- CSS DE ALTO CONTRASTE (CORREÇÃO DE CORES) ---
st.markdown("""
<style>
    /* Fonte Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* 1. Fundo e Cor Base */
    .stApp {
        background-color: #0E1117; /* Preto Fundo */
        color: #FAFAFA; /* Branco Quase Puro */
    }
    
    /* 2. Forçar Texto Branco em Parágrafos e Listas (A CORREÇÃO PRINCIPAL) */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li, 
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        color: #FAFAFA !important;
    }

    /* 3. Sidebar mais escura para contraste */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #333;
    }
    
    /* 4. Métricas (Cards) */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #404040;
        padding: 15px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: #CCCCCC !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    
    /* 5. Inputs (Selectbox, Text) - Fundo escuro e texto claro */
    div[data-baseweb="select"] > div {
        background-color: #1E1E1E !important;
        color: white !important;
        border-color: #444 !important;
    }
    div[data-baseweb="base-input"] {
        background-color: #1E1E1E !important;
        border-color: #444 !important;
    }
    input {
        color: white !important;
    }
    
    /* 6. Botões */
    div.stButton > button {
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
        border: none;
    }
    
    /* 7. Área de Upload */
    div[data-testid="stFileUploader"] {
        background-color: #161920;
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #4F8BF9;
    }
    
    /* 8. Tabelas */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES UTILITÁRIAS ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith("csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="model_trained.pkl" style="text-decoration:none; background-color:#2E7d32; color:white; padding:10px 20px; border-radius:5px; font-weight:bold;">📥 Baixar Modelo Treinado (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- SIDEBAR DE NAVEGAÇÃO ---
with st.sidebar:
    st.title("🚀 Data Studio")
    st.caption("v3.2 Enterprise High Contrast")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Carregar Dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy()
            st.session_state['file_name'] = uploaded_file.name
            st.toast("Arquivo carregado!", icon="✅")
        
        df_work = st.session_state['df_work']
        st.success(f"Registros: {df_work.shape[0]} | Colunas: {df_work.shape[1]}")
        
        if st.button("🔄 Restaurar Original"):
            st.session_state['df_work'] = st.session_state['df_raw'].copy()
            st.rerun()
            
        st.markdown("---")
        menu = st.radio("Módulos:", ["🏠 Dashboard", "🔬 Explorador", "🛠️ Engenharia", "🤖 AutoML", "📊 Visualizador"], index=0)
    else:
        menu = "Home"

# --- HOME ---
if not uploaded_file:
    st.title("Bem-vindo ao Data Studio Enterprise")
    st.markdown("""
    ### 🧠 Central de Inteligência de Dados
    
    Esta ferramenta cobre todo o ciclo de vida da ciência de dados. O texto agora está otimizado para leitura em modo escuro.
    
    1. **Ingestão:** Suporte nativo a CSV e Excel.
    2. **Exploração:** Dashboards automáticos.
    3. **Preparação:** Limpeza e transformação (One-Hot, Normalização).
    4. **Modelagem:** AutoML com Random Forest e Gradient Boosting.
    5. **Simulação:** Teste o modelo em tempo real.
    
    👈 **Para começar, carregue um arquivo na barra lateral.**
    """)
    st.stop()

# --- MÓDULO 1: DASHBOARD ---
if menu == "🏠 Dashboard":
    st.header("Visão Geral Executiva")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{df_work.shape[0]:,}")
    c2.metric("Colunas", df_work.shape[1])
    c3.metric("Nulos", df_work.isna().sum().sum())
    c4.metric("Memória", f"{df_work.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    
    st.markdown("---")
    
    col_L, col_R = st.columns([2, 1])
    with col_L:
        st.subheader("Visualização da Tabela")
        st.dataframe(df_work.head(100), use_container_width=True, height=400)
    
    with col_R:
        st.subheader("Tipos de Dados")
        dtypes = df_work.dtypes.value_counts().reset_index()
        dtypes.columns = ['Tipo', 'Qtd']
        dtypes['Tipo'] = dtypes['Tipo'].astype(str)
        
        # Template Dark automático
        fig = px.pie(dtypes, names='Tipo', values='Qtd', hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)") 
        st.plotly_chart(fig, use_container_width=True)

# --- MÓDULO 2: EXPLORADOR ---
elif menu == "🔬 Explorador":
    st.header("Análise Exploratória")
    
    tabs = st.tabs(["📊 Distribuição", "🔥 Correlação"])
    
    with tabs[0]:
        col_sel = st.selectbox("Variável:", df_work.columns)
        if pd.api.types.is_numeric_dtype(df_work[col_sel]):
            fig = px.histogram(df_work, x=col_sel, marginal="box", template="plotly_dark", color_discrete_sequence=['#4F8BF9'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            top = df_work[col_sel].value_counts().head(20).reset_index()
            fig = px.bar(top, x='index', y=col_sel, template="plotly_dark", labels={'index': col_sel, col_sel: 'Contagem'})
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            corr = df_num.corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", template="plotly_dark", color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.warning("Precisa de colunas numéricas para correlação.")

# --- MÓDULO 3: ENGENHARIA ---
elif menu == "🛠️ Engenharia":
    st.header("Preparação de Dados")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Ferramentas")
        acao = st.selectbox("Ação:", ["Tratar Nulos", "Remover Colunas", "One-Hot Encoding", "Normalizar"])
        
        if acao == "Tratar Nulos":
            cols = df_work.columns[df_work.isna().any()].tolist()
            if cols:
                c_target = st.selectbox("Coluna:", cols)
                metodo = st.radio("Preencher com:", ["Zero", "Média", "Moda"])
                if st.button("Aplicar"):
                    if metodo=="Zero": df_work[c_target].fillna(0, inplace=True)
                    elif metodo=="Média" and pd.api.types.is_numeric_dtype(df_work[c_target]): 
                        df_work[c_target].fillna(df_work[c_target].mean(), inplace=True)
                    else: df_work[c_target].fillna(df_work[c_target].mode()[0], inplace=True)
                    st.session_state['df_work'] = df_work
                    st.rerun()
            else:
                st.info("Sem nulos.")

        elif acao == "Remover Colunas":
            cols_drop = st.multiselect("Escolha:", df_work.columns)
            if st.button("Excluir") and cols_drop:
                df_work.drop(columns=cols_drop, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
        
        elif acao == "One-Hot Encoding":
            cols_cat = df_work.select_dtypes(include=['object']).columns
            c_target = st.selectbox("Coluna Categórica:", cols_cat)
            if st.button("Converter") and c_target:
                # CORREÇÃO IMPORTANTE: dtype=int para gerar 0 e 1
                df_work = pd.get_dummies(df_work, columns=[c_target], drop_first=True, dtype=int)
                st.session_state['df_work'] = df_work
                st.success("Convertido!")
                st.rerun()

        elif acao == "Normalizar":
            cols_num = st.multiselect("Colunas:", df_work.select_dtypes(include='number').columns)
            if st.button("Normalizar") and cols_num:
                scaler = StandardScaler()
                df_work[cols_num] = scaler.fit_transform(df_work[cols_num])
                st.session_state['df_work'] = df_work
                st.rerun()

    with c2:
        st.subheader("Visualização")
        st.dataframe(df_work.head(15), use_container_width=True)

# --- MÓDULO 4: AUTO ML ---
elif menu == "🤖 AutoML":
    st.header("Machine Learning Automático")
    
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        target = st.selectbox("Alvo (Previsão):", df_work.columns)
        features = st.multiselect("Features (Variáveis):", [c for c in df_work.columns if c != target])
        
        is_reg = pd.api.types.is_numeric_dtype(df_work[target]) and df_work[target].nunique() > 20
        tipo = "Regressão" if is_reg else "Classificação"
        st.caption(f"Modo: {tipo}")
        
        algoritmo = st.selectbox("Modelo:", ["Random Forest", "Gradient Boosting"])
        btn_treinar = st.button("🚀 Iniciar Treino", type="primary")
    
    with c_right:
        if btn_treinar and features:
            with st.spinner("Treinando..."):
                try:
                    X = df_work[features]
                    y = df_work[target]
                    
                    # Validação rápida
                    if X.select_dtypes(include=['object']).shape[1] > 0:
                        st.error("Erro: Converta colunas de texto em números na aba Engenharia antes de treinar.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    
                    if tipo == "Regressão":
                        model = RandomForestRegressor() if algoritmo == "Random Forest" else GradientBoostingRegressor()
                    else:
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            y_train, y_test = train_test_split(y, test_size=0.2)
                        model = RandomForestClassifier() if algoritmo == "Random Forest" else GradientBoostingClassifier()
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    # Salva na sessão
                    st.session_state['model'] = model
                    st.session_state['features'] = features
                    
                    # Métricas
                    if tipo == "Regressão":
                        r2 = r2_score(y_test, preds)
                        st.metric("R² (Precisão)", f"{r2:.2%}")
                        fig = px.scatter(x=y_test, y=preds, labels={'x':'Real', 'y':'Previsto'}, template="plotly_dark", title="Real vs Previsto")
                        fig.add_shape(type="line", line=dict(dash="dash", color="white"), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        acc = accuracy_score(y_test, preds)
                        st.metric("Acurácia", f"{acc:.2%}")
                        cm = confusion_matrix(y_test, preds)
                        fig = px.imshow(cm, text_auto=True, template="plotly_dark", title="Matriz de Confusão")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Erro: {e}")

    # Simulador
    if 'model' in st.session_state:
        st.markdown("---")
        st.subheader("🔮 Simulador")
        cols = st.columns(4)
        inputs = {}
        for i, col in enumerate(st.session_state['features']):
            with cols[i%4]:
                if pd.api.types.is_numeric_dtype(df_work[col]):
                    val = float(df_work[col].mean())
                    inputs[col] = st.number_input(col, value=val)
                else:
                    inputs[col] = st.selectbox(col, [0, 1])
        
        if st.button("Prever Resultado"):
            res = st.session_state['model'].predict(pd.DataFrame([inputs]))[0]
            st.success(f"Previsão: {res:.2f}")
            download_model(st.session_state['model'])

# --- MÓDULO 5: VISUALIZADOR ---
elif menu == "📊 Visualizador":
    st.header("Gráficos Personalizados")
    c1, c2, c3 = st.columns(3)
    tipo = c1.selectbox("Tipo:", ["Scatter", "Barra", "Linha", "Boxplot"])
    x = c2.selectbox("Eixo X:", df_work.columns)
    y = c3.selectbox("Eixo Y:", df_work.columns)
    cor = st.selectbox("Cor (Legenda):", [None] + list(df_work.columns))
    
    if st.button("Gerar Gráfico"):
        if tipo == "Scatter": fig = px.scatter(df_work, x=x, y=y, color=cor, template="plotly_dark")
        elif tipo == "Barra": fig = px.bar(df_work, x=x, y=y, color=cor, template="plotly_dark")
        elif tipo == "Linha": fig = px.line(df_work, x=x, y=y, color=cor, template="plotly_dark")
        elif tipo == "Boxplot": fig = px.box(df_work, x=x, y=y, color=cor, template="plotly_dark")
        
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)