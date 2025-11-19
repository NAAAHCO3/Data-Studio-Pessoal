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
    page_title="Data Studio Pessoal", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- CSS PROFISSIONAL (DARK THEME TWEAKS) ---
st.markdown("""
<style>
    /* Importando fonte moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Estilização de Cards para Métricas */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Títulos e Cabeçalhos */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(90deg, #4F8BF9 0%, #9B5DE5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }

    /* Ajuste da Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #333;
    }
    
    /* Botões Personalizados */
    div.stButton > button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 139, 249, 0.4);
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
    href = f'<a href="data:file/output_model;base64,{b64}" download="model_trained.pkl" class="css-button">📥 Download Modelo (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- SIDEBAR DE NAVEGAÇÃO ---
with st.sidebar:
    st.title("🚀 Data Studio")
    st.markdown("v3.0")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Carregar Dataset (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Gestão de Estado do Dataset
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy()
            st.session_state['file_name'] = uploaded_file.name
            st.toast("Dataset carregado com sucesso!", icon="✅")
        
        df_work = st.session_state['df_work']
        
        st.info(f"📊 **Status:**\n\nLinhas: {df_work.shape[0]}\nColunas: {df_work.shape[1]}")
        
        if st.button("🔄 Restaurar Original", type="secondary"):
            st.session_state['df_work'] = st.session_state['df_raw'].copy()
            st.rerun()
            
        st.markdown("---")
        
        # Menu de Ícones
        menu = st.radio(
            "Navegação:",
            ["🏠 Dashboard", "🔬 Explorador de Dados", "🛠️ Engenharia de Features", "🤖 AutoML Studio", "📊 Visualizador Pro"],
            index=0
        )
    else:
        menu = "Home"

# --- PÁGINA INICIAL (SEM DADOS) ---
if not uploaded_file:
    st.title("Bem-vindo ao Data Studio Enterprise")
    st.markdown("""
    ### Sua Central de Inteligência de Dados
    
    Esta ferramenta foi desenhada para cobrir todo o ciclo de vida da ciência de dados:
    
    1.  **Ingestão:** Suporte a CSV e Excel.
    2.  **Exploração:** Dashboards automáticos e edição de dados.
    3.  **Preparação:** Limpeza e transformação (One-Hot, Normalização).
    4.  **Modelagem:** AutoML com Random Forest, Gradient Boosting e Regressão Linear.
    5.  **Simulação:** Teste o modelo com novos dados em tempo real.
    
    👈 **Comece carregando um arquivo na barra lateral.**
    """)
    st.stop()

# ====================================================================
# MÓDULO 1: DASHBOARD
# ====================================================================
if menu == "🏠 Dashboard":
    st.header("Visão Geral Executiva")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Registros", f"{df_work.shape[0]:,}")
    col2.metric("Variáveis (Colunas)", df_work.shape[1])
    col3.metric("Dados Faltantes", f"{df_work.isna().sum().sum()} células", delta_color="inverse")
    col4.metric("Memória Usada", f"{df_work.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.markdown("---")
    
    # Layout de Grid
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.subheader("Amostra de Dados Interativa")
        st.data_editor(df_work.head(1000), height=400, use_container_width=True)
    
    with c2:
        st.subheader("Tipos de Dados")
        dtypes = df_work.dtypes.value_counts().reset_index()
        dtypes.columns = ['Tipo', 'Contagem']
        dtypes['Tipo'] = dtypes['Tipo'].astype(str)
        fig = px.pie(dtypes, names='Tipo', values='Contagem', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# MÓDULO 2: EXPLORADOR
# ====================================================================
elif menu == "🔬 Explorador de Dados":
    st.header("Análise Exploratória Profunda")
    
    tabs = st.tabs(["📊 Distribuições", "🔥 Correlações (Heatmap)", "📑 Pandas Profiling Light"])
    
    with tabs[0]:
        st.markdown("#### Analisar Distribuição Univariada")
        col_sel = st.selectbox("Selecione uma variável:", df_work.columns)
        
        c1, c2 = st.columns(2)
        with c1:
            if pd.api.types.is_numeric_dtype(df_work[col_sel]):
                fig = px.histogram(df_work, x=col_sel, marginal="box", nbins=30, title=f"Histograma de {col_sel}")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                top_n = df_work[col_sel].value_counts().nlargest(15).reset_index()
                top_n.columns = [col_sel, 'Contagem']
                fig = px.bar(top_n, x=col_sel, y='Contagem', title=f"Top 15 Categorias: {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.write("### Estatísticas Descritivas")
            if pd.api.types.is_numeric_dtype(df_work[col_sel]):
                desc = df_work[col_sel].describe().to_frame()
                st.dataframe(desc, use_container_width=True)
            else:
                st.write(f"**Valores Únicos:** {df_work[col_sel].nunique()}")
                st.write(f"**Moda:** {df_work[col_sel].mode()[0]}")
                st.dataframe(df_work[col_sel].value_counts().head(10), use_container_width=True)

    with tabs[1]:
        st.markdown("#### Relacionamento entre Variáveis")
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            corr = df_num.corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.warning("Precisa de pelo menos 2 colunas numéricas.")

    with tabs[2]:
        st.markdown("#### Diagnóstico de Qualidade")
        nulls = df_work.isna().sum().reset_index()
        nulls.columns = ['Coluna', 'Nulos']
        nulls['%'] = (nulls['Nulos'] / len(df_work)) * 100
        
        st.dataframe(
            nulls.style.background_gradient(subset=['%'], cmap='Reds'),
            use_container_width=True
        )

# ====================================================================
# MÓDULO 3: ENGENHARIA DE FEATURES
# ====================================================================
elif menu == "🛠️ Engenharia de Features":
    st.header("Laboratório de Dados")
    st.markdown("Transforme dados brutos em inputs prontos para IA.")
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("Ações")
        action = st.selectbox("Escolha uma operação:", 
            ["Tratar Nulos (Numérico)", "Tratar Nulos (Texto)", "Remover Colunas", "One-Hot Encoding (Dummy)", "Normalização (StandardScaler)"])
        
        st.markdown("---")
        
        if action == "Tratar Nulos (Numérico)":
            cols = df_work.select_dtypes(include='number').columns[df_work.select_dtypes(include='number').isna().any()].tolist()
            if cols:
                c_sel = st.selectbox("Coluna:", cols)
                method = st.radio("Método:", ["Média", "Mediana", "Zero"])
                if st.button("Aplicar"):
                    if method == "Média": val = df_work[c_sel].mean()
                    elif method == "Mediana": val = df_work[c_sel].median()
                    else: val = 0
                    df_work[c_sel] = df_work[c_sel].fillna(val)
                    st.session_state['df_work'] = df_work
                    st.success("Aplicado!")
                    st.rerun()
            else:
                st.success("Sem nulos numéricos pendentes.")

        elif action == "Tratar Nulos (Texto)":
            cols = df_work.select_dtypes(exclude='number').columns[df_work.select_dtypes(exclude='number').isna().any()].tolist()
            if cols:
                c_sel = st.selectbox("Coluna:", cols)
                fill_val = st.text_input("Preencher com:", "Não Informado")
                if st.button("Aplicar"):
                    df_work[c_sel] = df_work[c_sel].fillna(fill_val)
                    st.session_state['df_work'] = df_work
                    st.success("Aplicado!")
                    st.rerun()
            else:
                st.success("Sem nulos de texto pendentes.")

        elif action == "Remover Colunas":
            cols = st.multiselect("Selecione para excluir:", df_work.columns)
            if cols and st.button("Excluir Selecionadas"):
                df_work = df_work.drop(columns=cols)
                st.session_state['df_work'] = df_work
                st.rerun()

        elif action == "One-Hot Encoding (Dummy)":
            cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
            if cols:
                c_sel = st.selectbox("Coluna Categórica:", cols)
                if st.button("Converter em Números"):
                    df_work = pd.get_dummies(df_work, columns=[c_sel], drop_first=True)
                    st.session_state['df_work'] = df_work
                    st.success("Colunas dummy criadas!")
                    st.rerun()
            else:
                st.warning("Sem colunas categóricas.")
        
        elif action == "Normalização (StandardScaler)":
            cols = st.multiselect("Colunas para Normalizar:", df_work.select_dtypes(include='number').columns)
            if cols and st.button("Normalizar"):
                scaler = StandardScaler()
                df_work[cols] = scaler.fit_transform(df_work[cols])
                st.session_state['df_work'] = df_work
                st.success("Dados normalizados!")
                st.rerun()

    with col_r:
        st.subheader("Pré-visualização em Tempo Real")
        st.dataframe(df_work.head(15), use_container_width=True)
        st.info(f"Dimensões Atuais: {df_work.shape}")

# ====================================================================
# MÓDULO 4: AUTO ML STUDIO
# ====================================================================
elif menu == "🤖 AutoML Studio":
    st.header("Inteligência Artificial Automatizada")
    
    c_config, c_result = st.columns([1, 2])
    
    with c_config:
        st.subheader("1. Configuração")
        target = st.selectbox("🎯 Variável Alvo (Target):", df_work.columns)
        
        # Heurística de Tipo de Problema
        is_numeric = pd.api.types.is_numeric_dtype(df_work[target])
        n_unique = df_work[target].nunique()
        problem_type = "Regressão" if is_numeric and n_unique > 20 else "Classificação"
        
        st.caption(f"Tipo Detectado: **{problem_type}**")
        
        features = st.multiselect("Variáveis Preditoras (Features):", [c for c in df_work.columns if c != target])
        
        model_type = st.selectbox("Escolha o Algoritmo:", 
                                  ["Random Forest", "Gradient Boosting", "Linear/Logistic Regression"] if problem_type == "Regressão" 
                                  else ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        
        split_size = st.slider("Tamanho do Teste (%)", 10, 50, 20) / 100
        
        train_btn = st.button("🚀 Treinar Modelo", type="primary")

    with c_result:
        if train_btn and features:
            try:
                with st.spinner("Treinando modelo inteligente..."):
                    # Preparação
                    X = df_work[features]
                    y = df_work[target]
                    
                    # Validação de Texto nas Features
                    if X.select_dtypes(include=['object']).shape[1] > 0:
                        st.error("⚠️ Features contêm texto! Vá para 'Engenharia de Features' e use One-Hot Encoding.")
                        st.stop()
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
                    
                    # Seleção do Modelo
                    if problem_type == "Regressão":
                        if model_type == "Random Forest": model = RandomForestRegressor()
                        elif model_type == "Gradient Boosting": model = GradientBoostingRegressor()
                        else: model = LinearRegression()
                    else:
                        # Encoding do target se for texto
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y_train = le.fit_transform(y_train)
                            y_test = le.transform(y_test)
                        
                        if model_type == "Random Forest": model = RandomForestClassifier()
                        elif model_type == "Gradient Boosting": model = GradientBoostingClassifier()
                        else: model = LogisticRegression()
                    
                    # Treino
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    # Persistência em Sessão
                    st.session_state['model'] = model
                    st.session_state['features'] = features
                    st.session_state['model_type'] = problem_type
                    
                    # --- RESULTADOS ---
                    st.subheader("🏆 Performance do Modelo")
                    
                    m1, m2, m3 = st.columns(3)
                    if problem_type == "Regressão":
                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        m1.metric("R² (Explicação)", f"{r2:.1%}")
                        m2.metric("Erro Médio (MAE)", f"{mae:.2f}")
                        
                        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Real', 'y': 'Previsto'}, title="Real vs Previsto")
                        fig.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        acc = accuracy_score(y_test, preds)
                        m1.metric("Acurácia", f"{acc:.1%}")
                        
                        # Matriz de Confusão
                        cm = confusion_matrix(y_test, preds)
                        fig = px.imshow(cm, text_auto=True, title="Matriz de Confusão", color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)

                    # Feature Importance (Se disponível)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("O que mais impacta o resultado?")
                        imp = pd.DataFrame({'Feature': features, 'Importância': model.feature_importances_}).sort_values('Importância', ascending=False)
                        fig_imp = px.bar(imp, x='Importância', y='Feature', orientation='h')
                        st.plotly_chart(fig_imp, use_container_width=True)

            except Exception as e:
                st.error(f"Erro no treino: {e}")

    # --- ÁREA DE SIMULAÇÃO (PÓS-TREINO) ---
    if 'model' in st.session_state:
        st.markdown("---")
        st.header("🔮 Simulador de Previsões")
        st.markdown("Use o modelo treinado para prever novos casos.")
        
        sim_cols = st.columns(4)
        user_input = {}
        
        # Gera inputs dinâmicos baseados nas features
        features_model = st.session_state['features']
        for i, col in enumerate(features_model):
            with sim_cols[i % 4]:
                # Tenta inferir valores min/max para criar sliders inteligentes
                if pd.api.types.is_numeric_dtype(df_work[col]):
                    min_v = float(df_work[col].min())
                    max_v = float(df_work[col].max())
                    mean_v = float(df_work[col].mean())
                    user_input[col] = st.number_input(f"{col}", value=mean_v)
                else:
                    # Se for categórica dummy, é 0 ou 1
                    user_input[col] = st.selectbox(f"{col}", [0, 1])
        
        if st.button("🔮 Realizar Previsão"):
            input_df = pd.DataFrame([user_input])
            prediction = st.session_state['model'].predict(input_df)[0]
            
            st.success(f"### Resultado da Previsão: {prediction:.2f}")
            
        # Botão de Download
        st.markdown("### Exportar Modelo")
        download_model(st.session_state['model'])

# ====================================================================
# MÓDULO 5: VISUALIZADOR PRO
# ====================================================================
elif menu == "📊 Visualizador Pro":
    st.header("Analytics Studio")
    
    c1, c2, c3 = st.columns(3)
    chart_type = c1.selectbox("Tipo de Gráfico", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap Density"])
    x_axis = c2.selectbox("Eixo X", df_work.columns)
    y_axis = c3.selectbox("Eixo Y", df_work.columns, index=1 if len(df_work.columns) > 1 else 0)
    
    col_opt1, col_opt2 = st.columns(2)
    color_dim = col_opt1.selectbox("Legenda (Cor)", [None] + list(df_work.columns))
    size_dim = col_opt2.selectbox("Tamanho da Bolha (Scatter apenas)", [None] + list(df_work.select_dtypes(include='number').columns))
    
    if st.button("Gerar Visualização", type="primary"):
        try:
            if chart_type == "Scatter Plot":
                fig = px.scatter(df_work, x=x_axis, y=y_axis, color=color_dim, size=size_dim, title=f"{y_axis} vs {x_axis}")
            elif chart_type == "Line Chart":
                fig = px.line(df_work, x=x_axis, y=y_axis, color=color_dim)
            elif chart_type == "Bar Chart":
                fig = px.bar(df_work, x=x_axis, y=y_axis, color=color_dim)
            elif chart_type == "Histogram":
                fig = px.histogram(df_work, x=x_axis, color=color_dim)
            elif chart_type == "Box Plot":
                fig = px.box(df_work, x=x_axis, y=y_axis, color=color_dim)
            elif chart_type == "Violin Plot":
                fig = px.violin(df_work, x=x_axis, y=y_axis, color=color_dim)
            elif chart_type == "Heatmap Density":
                fig = px.density_heatmap(df_work, x=x_axis, y=y_axis)
                
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")