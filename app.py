"""
Enterprise Analytics — Code-First Edition (v13.0)
Author: Gemini Advanced
Version: 13.0 (Python Studio, Seaborn, Advanced SQL, Massive Academy)

Destaques v13.0:
- CORE: Substituição de ETL/Viz visuais por "Python Studio" (Terminal com Snippets).
- LIB: Adição de Seaborn e Matplotlib nativos.
- EDU: Academy expandido para formato "E-book Completo".
- DATA: Gerador de Dados Avançado (Sazonalidade, Tendência, Categorias).
- SQL: Cheat Sheet Definitiva (Window Functions, CTEs).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import logging
import re
import time
import pickle
import yaml
import hashlib
import json
import random
import unicodedata
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

# --- Scientific Stack ---
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, classification_report

# Optional Libs
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

# PDF Support
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ---------------------------
# CONFIG & STYLES
# ---------------------------
st.set_page_config(
    page_title="Data Studio Code-First", 
    layout="wide", 
    page_icon="🐍", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    code { font-family: 'Fira Code', monospace; }
    
    /* Tutorial Highlight */
    .tutorial-box {
        background-color: #fffbeb; border-left: 5px solid #f59e0b; 
        padding: 15px; border-radius: 8px; margin-bottom: 20px; 
        font-size: 1rem; color: #78350f; box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    
    /* Python Editor Style */
    .stTextArea textarea {
        font-family: 'Fira Code', monospace !important;
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .tutorial-box { background-color: #451a03; border-color: #d97706; color: #fef3c7; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SESSION INIT
# ---------------------------
def init_session():
    defaults = {
        'df': pd.DataFrame(), 
        'df_raw': pd.DataFrame(),
        'report_charts': [], 
        'model_registry': [], 
        'last_file_uid': None,
        'tutorial_mode': False,
        'code_snippet': "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\n# O DataFrame está disponível como 'df'\nst.write(df.head())",
        'generated_data_params': {}
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()

# ---------------------------
# UTILITIES
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file, encoding_errors='ignore')
        return pd.read_excel(file)
    except Exception as e: st.error(f"Erro leitura: {str(e)}"); return pd.DataFrame()

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True).str.replace(r"[^0-9a-zA-Z_]", "", regex=True).str.lower())
    return df

def render_tutorial(text: str):
    if st.session_state.get('tutorial_mode'):
        st.markdown(f"<div class='tutorial-box'><b>🎓 TUTORIAL:</b> {text}</div>", unsafe_allow_html=True)

# ---------------------------
# DATA GENERATOR ENGINE
# ---------------------------
class DataGenerator:
    @staticmethod
    def generate(n_rows, trend_type, noise_level, has_text, has_cats):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_rows)
        data = {'data': dates}
        
        # 1. Numérico com Tendência
        base = np.linspace(0, 100, n_rows)
        noise = np.random.normal(0, noise_level, n_rows)
        
        if trend_type == "Linear Crescente": y = base * 2 + 50 + noise
        elif trend_type == "Sazonal (Senoide)": y = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, n_rows)) + noise
        elif trend_type == "Exponencial": y = 10 * np.exp(np.linspace(0, 2, n_rows)) + noise
        else: y = np.random.normal(100, 30, n_rows) # Aleatório
        
        data['valor_principal'] = np.round(y, 2)
        data['valor_secundario'] = np.round(y * 0.5 + np.random.normal(0, 10, n_rows), 2) # Correlacionado
        
        # 2. Categorias
        if has_cats:
            cats = ['A', 'B', 'C']
            weights = [0.6, 0.3, 0.1] # Desbalanceado
            data['categoria'] = np.random.choice(cats, n_rows, p=weights)
            data['status'] = np.random.choice(['Ativo', 'Inativo', 'Pendente'], n_rows)

        # 3. Texto (NLP)
        if has_text:
            reviews_pos = ["Adorei o produto", "Muito bom", "Excelente qualidade", "Recomendo"]
            reviews_neg = ["Péssimo", "Não gostei", "Chegou quebrado", "Devolvi"]
            # Texto correlacionado com o valor
            txts = []
            labels = []
            for val in data['valor_principal']:
                if val > data['valor_principal'].mean():
                    txts.append(np.random.choice(reviews_pos))
                    labels.append("Positivo")
                else:
                    txts.append(np.random.choice(reviews_neg))
                    labels.append("Negativo")
            data['comentario_cliente'] = txts
            data['sentimento_real'] = labels

        # 4. Dirty Data Injection
        df = pd.DataFrame(data)
        
        # Inject Nulls
        mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95])
        df = df.mask(mask)
        
        return df

# ---------------------------
# PDF ENGINE (FIXED)
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Enterprise Analytics Report v13.0', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v13(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = EnterprisePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Capa
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Relatório Técnico", 0, 1, 'C')
    pdf.ln(10)
    
    # KPIs
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Resumo dos Dados", 1, 1)
    pdf.set_font("Helvetica", "", 12)
    w = 45
    pdf.cell(w, 10, f"Linhas: {kpis['rows']}", 1)
    pdf.cell(w, 10, f"Colunas: {kpis['cols']}", 1)
    pdf.cell(w, 10, f"Nulos: {kpis['nulls']}", 1)
    pdf.ln(15)
    
    # Charts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Análises Geradas", 0, 1)
    
    import tempfile, os
    for i, ch in enumerate(charts):
        if i > 0 and i % 2 == 0: pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, ch['title'], 0, 1)
        
        # Image Logic
        try:
            # Tentar exportar usando kaleido
            img = ch['fig'].to_image(format="png", scale=1.0, engine="kaleido")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img)
                path = tmp.name
            pdf.image(path, x=20, w=170)
            os.unlink(path)
        except Exception as e:
            pdf.set_font("Courier", "I", 9)
            pdf.cell(0, 10, f"[Imagem não disponível: {str(e)}]", 0, 1)
        
        pdf.ln(5)
        if ch.get('note'):
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 5, f"Obs: {ch['note']}")
            pdf.ln(5)
            
    return bytes(pdf.output())

# ---------------------------
# PAGES
# ---------------------------

def page_home():
    st.title("🏠 Home")
    
    if st.session_state.get('tutorial_mode'):
        st.info("🎓 MODO TUTORIAL ATIVO: Siga as caixas amarelas para aprender.")

    df = st.session_state['df']
    
    if df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.info("Nenhum dado carregado. Use o Gerador ou carregue um arquivo.")
            if st.button("Ir para Gerador de Dados 🎲"):
                st.session_state['page_selection'] = '🎲 Gerador de Dados' # Hacky nav
                st.rerun()
        return

    # Overview
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Linhas", f"{df.shape[0]:,}")
    k2.metric("Colunas", df.shape[1])
    k3.metric("Duplicatas", df.duplicated().sum())
    k4.metric("Nulos", df.isna().sum().sum())

    st.markdown("### 📋 Amostra dos Dados")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### 🔢 Describe")
    st.dataframe(df.describe(include='all'), use_container_width=True)

def page_generator():
    st.title("🎲 Gerador de Dados")
    render_tutorial("Use esta tela para criar datasets controlados e entender como os algoritmos reagem a diferentes padrões (ex: criar uma tendência linear perfeita e ver a Regressão Linear acertar 100%).")
    
    c1, c2 = st.columns(2)
    with c1:
        n_rows = st.slider("Número de Linhas", 100, 5000, 500)
        trend = st.selectbox("Padrão Numérico", ["Aleatório", "Linear Crescente", "Sazonal (Senoide)", "Exponencial"])
        noise = st.slider("Nível de Ruído (Dificuldade)", 0, 100, 20)
    with c2:
        has_text = st.checkbox("Incluir Texto (NLP)", value=True)
        has_cats = st.checkbox("Incluir Categorias Desbalanceadas", value=True)
        
    if st.button("🚀 Gerar Dataset"):
        df = DataGenerator.generate(n_rows, trend, noise, has_text, has_cats)
        st.session_state['df'] = df
        st.session_state['df_raw'] = df.copy()
        st.success("Dados gerados e carregados na memória!")
        st.dataframe(df.head(), use_container_width=True)

def page_python_studio():
    st.title("🐍 Python Studio")
    render_tutorial("""
    Esta é sua área de trabalho profissional. Escreva código Python real para manipular o dataframe `df`.
    Use os botões ao lado para colar 'Snippets' (códigos prontos) e aprenda lendo o código.
    """)
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    c_code, c_snip = st.columns([2, 1])
    
    with c_snip:
        st.subheader("📚 Snippets (Colas)")
        
        with st.expander("Manipulação (Pandas)"):
            if st.button("Filtrar Dados"):
                st.session_state['code_snippet'] = "df = df[df['valor_principal'] > 50]\nst.write(df.head())"
            if st.button("Agrupar (GroupBy)"):
                st.session_state['code_snippet'] = "res = df.groupby('categoria')['valor_principal'].sum().reset_index()\nst.write(res)"
            if st.button("Tratar Nulos"):
                st.session_state['code_snippet'] = "df = df.fillna(0)\nst.write('Nulos removidos!')\nst.write(df.isna().sum())"
            if st.button("Criar Dummies"):
                st.session_state['code_snippet'] = "df = pd.get_dummies(df, columns=['categoria'], drop_first=True)\nst.write(df.head())"

        with st.expander("Gráficos (Seaborn/Matplotlib)"):
            if st.button("Histograma (Dist)"):
                st.session_state['code_snippet'] = "fig, ax = plt.subplots()\nsns.histplot(df['valor_principal'], kde=True, ax=ax)\nst.pyplot(fig)"
            if st.button("Boxplot (Outliers)"):
                st.session_state['code_snippet'] = "fig, ax = plt.subplots()\nsns.boxplot(x='categoria', y='valor_principal', data=df, ax=ax)\nst.pyplot(fig)"
            if st.button("Heatmap (Correlação)"):
                st.session_state['code_snippet'] = "fig, ax = plt.subplots()\nsns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)\nst.pyplot(fig)"
            if st.button("Pairplot (Geral)"):
                st.session_state['code_snippet'] = "fig = sns.pairplot(df.select_dtypes(include='number'))\nst.pyplot(fig)"

        with st.expander("Gráficos Interativos (Plotly)"):
            if st.button("Scatter Interativo"):
                st.session_state['code_snippet'] = "fig = px.scatter(df, x='data', y='valor_principal', color='categoria', title='Análise Temporal')\nst.plotly_chart(fig)"

    with c_code:
        st.markdown("Editor de Código (Variável `df` disponível)")
        code = st.text_area("Python Code", value=st.session_state.get('code_snippet', ''), height=300)
        
        if st.button("▶️ Executar Código"):
            try:
                # Sandbox execution
                local_vars = {
                    'df': df.copy(), 
                    'pd': pd, 'np': np, 
                    'plt': plt, 'sns': sns, 'px': px, 
                    'st': st
                }
                exec(code, {}, local_vars)
                
                # Update df if changed
                if 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
                    # Check if user wanted to update global df? For safety in this mode, 
                    # we might require explicit button or keep it local. 
                    # Let's allow update to session if variable is modified.
                    if not local_vars['df'].equals(df):
                        st.session_state['df'] = local_vars['df']
                        st.toast("DataFrame atualizado na memória!", icon="💾")
                        
            except Exception as e:
                st.error(f"Erro no código: {e}")

def page_sql_studio():
    st.title("💻 SQL Studio & Cheat Sheet")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB necessário."); return

    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.subheader("📖 Cheat Sheet")
        with st.expander("Básico"):
            st.markdown("**Selecionar:** `SELECT col1, col2 FROM df`")
            st.markdown("**Filtrar:** `SELECT * FROM df WHERE col > 10`")
            st.markdown("**Ordenar:** `ORDER BY col DESC`")
            st.markdown("**Limitar:** `LIMIT 100`")
        
        with st.expander("Agregações"):
            st.markdown("**Contar:** `COUNT(*)`")
            st.markdown("**Soma:** `SUM(col)`")
            st.markdown("**Média:** `AVG(col)`")
            st.markdown("**Agrupar:** `GROUP BY col`")
            st.markdown("**Filtro Agregado:** `HAVING COUNT(*) > 1`")
            
        with st.expander("Avançado (DuckDB)"):
            st.markdown("**Window (Rank):** `RANK() OVER (ORDER BY col)`")
            st.markdown("**Window (Media Movel):** `AVG(col) OVER (ORDER BY data ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`")
            st.markdown("**Texto:** `UPPER(col)`, `LOWER(col)`")
            st.markdown("**Data:** `DATE_TRUNC('month', data)`")
            
    with c2:
        render_tutorial("Use SQL para consultas rápidas. O DuckDB permite funções analíticas avançadas (Window Functions) que seriam complexas no Excel.")
        query = st.text_area("Query SQL (Tabela = 'df')", "SELECT * FROM df LIMIT 10", height=200)
        if st.button("Executar Query"):
            try:
                res = duckdb.query(query).to_df()
                st.dataframe(res, use_container_width=True)
            except Exception as e:
                st.error(f"Erro SQL: {e}")

def page_ml_studio():
    st.title("🤖 ML Studio (Glass Box)")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    t1, t2 = st.tabs(["Configurar & Treinar", "Simulador & Teste"])
    
    with t1:
        render_tutorial("Ajuste os Hiperparâmetros antes de treinar. Isso define como o modelo 'aprende'.")
        
        mode = st.radio("Modo", ["NLP (Texto)", "Regressão (Numérico)", "Classificação (Categoria)"])
        
        # Model Configuration
        model = None
        params_log = {}
        
        c1, c2 = st.columns(2)
        with c1:
            if mode == "NLP (Texto)":
                txt_col = st.selectbox("Texto", df.select_dtypes(include='object').columns)
                target = st.selectbox("Target", df.columns)
                
                st.markdown("#### Hiperparâmetros")
                ngram = st.slider("N-Grams (1=palavras, 2=frases curtas)", 1, 3, 1)
                max_feat = st.slider("Max Features (Vocabulário)", 100, 5000, 1000)
                
                params_log = {"ngram_range": (1, ngram), "max_features": max_feat}
                
            else:
                target = st.selectbox("Target", df.columns)
                feats = st.multiselect("Features", [c for c in df.columns if c!=target])
                
                st.markdown("#### Hiperparâmetros (Random Forest)")
                n_est = st.slider("Nº Árvores (n_estimators)", 10, 300, 100, help="Mais árvores = Mais estável, mas mais lento.")
                max_d = st.slider("Profundidade (max_depth)", 2, 30, 10, help="Controla complexidade. Alto = Risco de Overfitting.")
                
                params_log = {"n_estimators": n_est, "max_depth": max_d}

        with c2:
            st.info("Configuração Atual:")
            st.json(params_log)
            
            if st.button("🚀 Treinar Modelo"):
                with st.spinner("Treinando..."):
                    try:
                        if mode == "NLP (Texto)":
                            X = df[txt_col].astype(str)
                            y = df[target].astype(str)
                            pipe = Pipeline([
                                ('tfidf', TfidfVectorizer(**params_log)),
                                ('clf', LogisticRegression())
                            ])
                            metric = "Acurácia"
                        else:
                            X = df[feats]
                            y = df[target]
                            
                            # Prep pipeline
                            nums = X.select_dtypes(include=np.number).columns
                            cats = X.select_dtypes(include=['object']).columns
                            pre = ColumnTransformer([
                                ('n', SimpleImputer(strategy='median'), nums),
                                ('c', OneHotEncoder(handle_unknown='ignore'), cats)
                            ])
                            
                            if mode == "Regressão (Numérico)":
                                y = y.fillna(y.mean())
                                model_obj = RandomForestRegressor(**params_log)
                                metric = "R²"
                            else:
                                y = y.fillna(y.mode()[0]).astype(str)
                                model_obj = RandomForestClassifier(**params_log)
                                metric = "Acurácia"
                            
                            pipe = Pipeline([('pre', pre), ('model', model_obj)])

                        # Train/Test
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        pipe.fit(X_tr, y_tr)
                        
                        score = pipe.score(X_te, y_te)
                        train_score = pipe.score(X_tr, y_tr)
                        
                        st.success(f"Treino Finalizado!")
                        k1, k2, k3 = st.columns(3)
                        k1.metric(f"{metric} (Teste)", f"{score:.2f}")
                        k2.metric(f"{metric} (Treino)", f"{train_score:.2f}")
                        
                        # Overfitting check
                        if train_score - score > 0.15:
                            k3.error("⚠️ Alto Overfitting")
                            st.warning("O modelo decorou os dados. Reduza a 'Profundidade' ou 'Max Features'.")
                        else:
                            k3.success("✅ Modelo Saudável")

                        # Confusion Matrix (Classif/NLP)
                        if mode != "Regressão (Numérico)":
                            cm = confusion_matrix(y_te, pipe.predict(X_te))
                            st.markdown("#### Matriz de Confusão")
                            st.caption("Eixo Y = Realidade | Eixo X = Previsão. Diagonal = Acertos.")
                            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
                            st.plotly_chart(fig_cm, use_container_width=True)

                        # Save
                        st.session_state['ml_pipeline'] = pipe
                        st.session_state['ml_type'] = mode
                        if mode != "NLP (Texto)": st.session_state['ml_feats'] = feats

                    except Exception as e: st.error(f"Erro: {e}")

    with t2:
        if st.session_state.get('ml_pipeline'):
            st.subheader("Testar Modelo")
            model = st.session_state['ml_pipeline']
            mtype = st.session_state['ml_type']
            
            if mtype == "NLP (Texto)":
                txt = st.text_area("Digite um texto para classificar:")
                if st.button("Classificar"):
                    pred = model.predict([txt])[0]
                    proba = model.predict_proba([txt]).max()
                    st.info(f"Predição: **{pred}** (Confiança: {proba:.1%})")
            
            elif mtype == "Regressão (Numérico)":
                inputs = {}
                for f in st.session_state.get('ml_feats', []):
                    inputs[f] = st.text_input(f"Valor para {f}", "0")
                if st.button("Prever Valor"):
                    df_in = pd.DataFrame([inputs])
                    for c in df_in.columns: 
                        try: df_in[c] = pd.to_numeric(df_in[c])
                        except: pass
                    pred = model.predict(df_in)[0]
                    st.success(f"Valor Previsto: {pred:.2f}")
        else:
            st.info("Treine um modelo primeiro.")

def page_academy():
    st.title("🎓 Data Academy (O Livro)")
    
    tabs = st.tabs(["📊 Estatística", "💻 SQL Avançado", "🐍 Python", "🤖 Machine Learning"])
    
    with tabs[0]:
        st.header("Estatística para Dados")
        st.markdown("""
        ### 1. Distribuições
        - **Normal (Gaussiana):** O famoso "sino". A maioria dos modelos assume que os dados seguem isso (ex: altura das pessoas).
        - **Skewed (Assimétrica):** Cauda longa. Ex: Salários (muita gente ganha pouco, poucos ganham milhões). Use Log Transformation aqui.
        
        ### 2. Correlação não é Causalidade
        Só porque "Vendas de Sorvete" e "Ataques de Tubarão" sobem juntos (no verão), não significa que um causa o outro.
        
        ### 3. Testes de Hipótese (P-Value)
        - **H0 (Nula):** Nada aconteceu.
        - **H1 (Alternativa):** Algo mudou.
        - **P-Value < 0.05:** Rejeita a Nula. (Ou seja, "Estatisticamente Significativo").
        """)
        
    with tabs[1]:
        st.header("SQL Avançado")
        st.markdown("""
        ### Window Functions (O Poder do DuckDB)
        Fazem cálculos "olhando ao redor" da linha atual.
        
        **1. Média Móvel (Moving Average)**
        ```sql
        SELECT data, valor,
               AVG(valor) OVER (ORDER BY data ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as media_movel
        FROM vendas;
        ```
        
        **2. Ranking**
        ```sql
        SELECT produto, vendas,
               RANK() OVER (ORDER BY vendas DESC) as ranking
        FROM vendas;
        ```
        
        **3. CTEs (Common Table Expressions)**
        Deixam o código limpo.
        ```sql
        WITH VendasAltas AS (
            SELECT * FROM vendas WHERE valor > 1000
        )
        SELECT * FROM VendasAltas JOIN clientes ON ...
        ```
        """)

    with tabs[2]:
        st.header("Python & Pandas")
        st.markdown("""
        ### Lambda Functions
        Funções rápidas de uma linha.
        ```python
        df['dobro'] = df['valor'].apply(lambda x: x * 2)
        ```
        
        ### List Comprehension
        Loop rápido em uma linha.
        ```python
        novos_valores = [x * 2 for x in lista_antiga]
        ```
        
        ### Merge (VLOOKUP do Python)
        ```python
        # Inner: Só o que tem nos dois
        # Left: Tudo da esquerda + o que der match na direita
        pd.merge(tabela_a, tabela_b, on='id', how='left')
        ```
        """)

    with tabs[3]:
        st.header("Teoria de ML")
        st.markdown("""
        ### Bias-Variance Tradeoff
        - **Bias (Viés):** Modelo simplório demais. Erra treino e teste. (Underfitting)
        - **Variance (Variância):** Modelo complexo demais. Decora treino, erra teste. (Overfitting)
        
        ### Precision vs Recall
        Imagine um teste de COVID.
        - **Precision:** Dos que eu disse que tem COVID, quantos realmente tem? (Evita falso positivo).
        - **Recall:** De todos que tem COVID, quantos eu consegui achar? (Evita falso negativo - geralmente mais importante em medicina).
        """)

def page_report():
    st.title("📑 Relatório & PDF")
    charts = st.session_state['report_charts']
    df = st.session_state['df']
    
    if not charts:
        st.info("Vazio. Adicione gráficos via Python Studio (use st.session_state['report_charts'].append({'fig': fig, ...}) ou similar se customizar). Nota: Na versão Code-First, a adição de gráficos ao relatório é feita via código ou usando a função helper na sidebar se implementada.")
        # In Code-First, integrating auto-add from exec is hard. Providing manual adder ui here.
        st.write("Para adicionar gráficos aqui, use a aba Python Studio e salve as figuras no session_state['report_charts'].")
    else:
        for i, ch in enumerate(charts):
            st.markdown(f"**{i+1}. {ch.get('title','Gráfico')}**")
            try:
                st.plotly_chart(ch['fig'], use_container_width=True)
            except:
                st.pyplot(ch['fig'])
            
            st.caption(ch.get('note', ''))
            if st.button("Remover", key=f"del_{i}"):
                st.session_state['report_charts'].pop(i)
                st.rerun()
    
    st.markdown("---")
    if st.button("Gerar PDF Completo"):
        try:
            kpis = {"rows": len(df), "cols": df.shape[1], "nulls": df.isna().sum().sum(), "dups": df.duplicated().sum()}
            # Filter chart dicts that have plotly figs mostly, matplotlib support in PDF engine requires savefig
            # Simplified for plotly here as per prev versions
            plotly_charts = [c for c in charts if isinstance(c['fig'], go.Figure)]
            pdf_bytes = generate_report_v13(df, plotly_charts, kpis)
            st.download_button("Baixar PDF", pdf_bytes, "relatorio.pdf", "application/pdf")
            st.success("Gerado!")
        except Exception as e: st.error(f"Erro PDF: {e}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("🐍 Code-First v13")
        
        if st.checkbox("Modo Tutorial", value=st.session_state.get('tutorial_mode', False)):
            st.session_state['tutorial_mode'] = True
        else:
            st.session_state['tutorial_mode'] = False

        uploaded = st.file_uploader("Arquivo", type=['csv','xlsx'])
        if uploaded:
            uid = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get('last_uid') != uid:
                try:
                    if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded)
                    else: df = pd.read_excel(uploaded)
                    st.session_state['df'] = clean_colnames(df)
                    st.session_state['df_raw'] = st.session_state['df'].copy()
                    st.session_state['last_uid'] = uid
                    st.rerun()
                except Exception as e: st.error(e)

        st.markdown("---")
        menu = st.radio("Menu", ["🏠 Home", "🎲 Gerador Dados", "🐍 Python Studio", "💻 SQL Studio", "🎓 Academy", "🏆 ML Studio", "📑 Relatório"])
        if st.button("Reset"): st.session_state.clear(); st.rerun()

    if menu == "🏠 Home": page_home()
    elif menu == "🎲 Gerador Dados": page_generator()
    elif menu == "🐍 Python Studio": page_python_studio()
    elif menu == "💻 SQL Studio": page_sql_studio()
    elif menu == "🎓 Academy": page_academy()
    elif menu == "🏆 ML Studio": page_ml_studio()
    elif menu == "📑 Relatório": page_report()

if __name__ == "__main__":
    main()