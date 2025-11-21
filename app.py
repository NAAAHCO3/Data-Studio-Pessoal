"""
Enterprise Analytics — Code-First Edition (v13.0)
Author: Gemini Advanced
Version: 13.0 (Python Studio, Seaborn, Advanced SQL, Massive Academy)

Destaques v13.0:
- CORE: Substituição de ETL/Viz visuais por "Python Studio" (Terminal com Snippets).
- LIB: Adição de Seaborn e Matplotlib nativos.
- EDU: Academy expandido para formato "E-book Completo".
- DATA: Gerador de Dados Avançado (Customizável coluna a coluna).
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
    
    /* Academy Styles */
    .academy-header { color: #2563eb; font-size: 1.5rem; font-weight: 800; margin-top: 20px; }
    .academy-text { font-size: 1rem; line-height: 1.6; color: #334155; text-align: justify; }
    
    /* Python Editor Style */
    .stTextArea textarea {
        font-family: 'Fira Code', monospace !important;
        background-color: #0e1117 !important;
        color: #e6edf3 !important;
    }
    
    /* Cheat Sheet Box */
    .cheat-box {
        background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin-bottom: 10px;
    }
    .cheat-title { font-weight: bold; color: #0f172a; font-size: 0.9rem; }
    .cheat-desc { font-size: 0.8rem; color: #64748b; }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .academy-text { color: #cbd5e1; }
        .cheat-box { background: #1e293b; border-color: #334155; }
        .cheat-title { color: #f8fafc; }
        .cheat-desc { color: #94a3b8; }
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
        'code_snippet': "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\n# O DataFrame está disponível como 'df'\n# Exemplo: Ver as primeiras linhas\nst.write(df.head())",
        'gen_config': [] # List of columns to generate
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

# ---------------------------
# DATA GENERATOR ENGINE (CUSTOMIZABLE)
# ---------------------------
class DataGenerator:
    @staticmethod
    def generate_column(n_rows, config):
        typ = config['type']
        name = config['name']
        
        if typ == "Linear Trend":
            noise = np.random.normal(0, config.get('noise', 10), n_rows)
            base = np.linspace(0, 100, n_rows)
            return base * config.get('slope', 1) + config.get('intercept', 0) + noise
        
        elif typ == "Sazonal (Senoide)":
            x = np.linspace(0, 4 * np.pi, n_rows)
            return config.get('amplitude', 10) * np.sin(x) + config.get('base', 50) + np.random.normal(0, 5, n_rows)
        
        elif typ == "Categorico":
            cats = config.get('categories', ['A', 'B']).split(',')
            return np.random.choice(cats, n_rows)
        
        elif typ == "Texto (NLP)":
            pos = ["Excelente", "Bom", "Adorei", "Recomendo"]
            neg = ["Ruim", "Péssimo", "Odiei", "Não recomendo"]
            return np.random.choice(pos + neg, n_rows)
        
        elif typ == "Data":
            return pd.date_range(start='2023-01-01', periods=n_rows, freq='D')
            
        return np.zeros(n_rows)

# ---------------------------
# PDF ENGINE
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Code-First Analytics Report', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v13(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = EnterprisePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Relatório Técnico", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Resumo", 1, 1)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Linhas: {kpis['rows']} | Colunas: {kpis['cols']}", 0, 1)
    pdf.cell(0, 10, f"Nulos: {kpis['nulls']} | Duplicatas: {kpis['dups']}", 0, 1)
    pdf.ln(15)
    
    # Charts
    for i, ch in enumerate(charts):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, ch['title'], 0, 1)
        
        if ch.get('type') == 'image_bytes':
            # Handling raw bytes from matplotlib/seaborn
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(ch['data'])
                path = tmp.name
            try:
                pdf.image(path, x=15, w=180)
                os.unlink(path)
            except:
                pdf.cell(0, 10, "[Erro imagem]", 0, 1)
        
        pdf.ln(5)
        if ch.get('note'):
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 5, f"Nota: {ch['note']}")
            
    return bytes(pdf.output())

# ---------------------------
# PAGES
# ---------------------------

def page_home():
    st.title("🏠 Home")
    df = st.session_state['df']
    
    if df.empty:
        st.info("Nenhum dado carregado. Use a aba '🎲 Gerador' ou carregue um arquivo na lateral.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Linhas", f"{df.shape[0]:,}")
    k2.metric("Colunas", df.shape[1])
    k3.metric("Duplicatas", df.duplicated().sum())
    k4.metric("Nulos", df.isna().sum().sum())

    st.markdown("### 📋 Amostra & Estrutura")
    t1, t2 = st.tabs(["Head", "Info/Types"])
    with t1: st.dataframe(df.head(), use_container_width=True)
    with t2:
        dtypes = df.dtypes.astype(str).reset_index()
        dtypes.columns = ["Coluna", "Tipo"]
        st.dataframe(dtypes, use_container_width=True)

def page_generator():
    st.title("🎲 Gerador de Dados Pro")
    st.markdown("Construa seu dataset coluna por coluna para testar hipóteses.")
    
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.subheader("Adicionar Coluna")
        col_name = st.text_input("Nome da Coluna")
        col_type = st.selectbox("Tipo", ["Data", "Linear Trend", "Sazonal (Senoide)", "Categorico", "Texto (NLP)"])
        
        params = {}
        if col_type == "Linear Trend":
            params['slope'] = st.number_input("Inclinação (Slope)", value=1.0)
            params['noise'] = st.number_input("Ruído", value=10.0)
        elif col_type == "Categorico":
            params['categories'] = st.text_input("Categorias (sep. vírgula)", "A,B,C")
        elif col_type == "Sazonal (Senoide)":
            params['amplitude'] = st.number_input("Amplitude", value=10.0)
        
        if st.button("➕ Adicionar"):
            if col_name:
                st.session_state['gen_config'].append({"name": col_name, "type": col_type, **params})
                st.success(f"Coluna {col_name} agendada.")

    with c2:
        st.subheader("Configuração Atual")
        config = st.session_state['gen_config']
        if config:
            st.table(pd.DataFrame(config))
            if st.button("Limpar Configuração"):
                st.session_state['gen_config'] = []
                st.rerun()
            
            n_rows = st.number_input("Número de Linhas", 10, 10000, 500)
            if st.button("🚀 Gerar DataFrame"):
                data = {}
                for conf in config:
                    data[conf['name']] = DataGenerator.generate_column(n_rows, conf)
                
                df = pd.DataFrame(data)
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.success("Dados gerados e carregados!")
                st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("Adicione colunas à esquerda.")

def page_python_studio():
    st.title("🐍 Python Studio (IDE)")
    st.markdown("Escreva código real. `df` é seu dataframe. `plt` e `sns` estão disponíveis.")
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    col_snip, col_edit = st.columns([1, 3])
    
    with col_snip:
        st.subheader("📚 Snippets")
        st.caption("Clique para colar o código")
        
        with st.expander("Manipulação (Pandas)"):
            if st.button("Ver Nulos"): st.session_state['code_snippet'] = "st.write(df.isna().sum())"
            if st.button("Filtrar Dados"): st.session_state['code_snippet'] = "filtered = df[df['coluna'] > 100]\nst.write(filtered.head())"
            if st.button("Agrupar (GroupBy)"): st.session_state['code_snippet'] = "res = df.groupby('coluna')['valor'].sum().reset_index()\nst.write(res)"
            if st.button("Pivot Table"): st.session_state['code_snippet'] = "piv = df.pivot_table(index='data', columns='cat', values='val')\nst.write(piv)"

        with st.expander("Visualização (Seaborn)"):
            if st.button("Histograma"): st.session_state['code_snippet'] = "fig, ax = plt.subplots()\nsns.histplot(data=df, x='coluna', kde=True, ax=ax)\nst.pyplot(fig)"
            if st.button("Boxplot"): st.session_state['code_snippet'] = "fig, ax = plt.subplots()\nsns.boxplot(data=df, x='cat', y='val', ax=ax)\nst.pyplot(fig)"
            if st.button("Heatmap Corr"): st.session_state['code_snippet'] = "fig, ax = plt.subplots(figsize=(10,8))\nsns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)\nst.pyplot(fig)"
            if st.button("Pairplot"): st.session_state['code_snippet'] = "fig = sns.pairplot(df.select_dtypes(include='number'))\nst.pyplot(fig)"

    with col_edit:
        code = st.text_area("Editor", value=st.session_state.get('code_snippet', ''), height=400)
        c1, c2 = st.columns([1, 5])
        if c1.button("▶️ Executar"):
            try:
                local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'px': px, 'go': go, 'st': st}
                exec(code, {}, local_vars)
                
                # Capture logic (naive)
                # If user creates a figure named 'fig', we can save it
                if 'fig' in local_vars:
                    st.session_state['temp_fig'] = local_vars['fig']
                    st.success("Figura detectada na memória.")
            except Exception as e:
                st.error(f"Erro: {e}")
        
        if c2.button("💾 Salvar 'fig' no Relatório"):
            if 'temp_fig' in st.session_state:
                fig = st.session_state['temp_fig']
                # Check type
                img_data = None
                if isinstance(fig, plt.Figure):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    img_data = buf.getvalue()
                elif isinstance(fig, go.Figure):
                    try: img_data = fig.to_image(format="png")
                    except: pass
                
                if img_data:
                    st.session_state['report_charts'].append({"title": "Python Plot", "type": "image_bytes", "data": img_data, "note": "Gerado via código"})
                    st.toast("Salvo!")
            else:
                st.warning("Nenhuma variável 'fig' encontrada na última execução.")

def page_sql_studio():
    st.title("💻 SQL Studio & Cheat Sheet")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB ausente."); return

    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.markdown("### 📖 Cheat Sheet Definitiva")
        
        with st.expander("🔍 SELECT Básico"):
            st.markdown("""
            **Tudo:** `SELECT * FROM df`
            **Colunas:** `SELECT col1, col2 FROM df`
            **Alias:** `SELECT col1 AS nome_novo FROM df`
            **Distintos:** `SELECT DISTINCT categoria FROM df`
            """)
            
        with st.expander("⚖️ Filtros (WHERE)"):
            st.markdown("""
            **Maior/Menor:** `WHERE valor > 100`
            **Texto:** `WHERE nome = 'João'`
            **Parcial:** `WHERE nome LIKE '%Silva%'`
            **Lista:** `WHERE uf IN ('SP', 'RJ')`
            **Nulos:** `WHERE email IS NULL`
            **Lógica:** `WHERE (A > 10 OR B < 5) AND C = 1`
            """)
            
        with st.expander("∑ Agregações"):
            st.markdown("""
            **Contar:** `COUNT(*)`
            **Soma:** `SUM(vendas)`
            **Média:** `AVG(idade)`
            **Max/Min:** `MAX(data)`
            **Estrutura:**
            ```sql
            SELECT cat, SUM(val)
            FROM df
            GROUP BY cat
            HAVING SUM(val) > 1000
            ```
            """)
            
        with st.expander("🪟 Window Functions (Pro)"):
            st.markdown("""
            **Rank:** `RANK() OVER (ORDER BY val DESC)`
            **Acumulado:** `SUM(val) OVER (ORDER BY data)`
            **Anterior (Lag):** `LAG(val) OVER (ORDER BY data)`
            **Média Móvel:**
            ```sql
            AVG(val) OVER (
              ORDER BY data
              ROWS BETWEEN 2 PRECEDING
              AND CURRENT ROW
            )
            ```
            """)
            
        with st.expander("📅 Datas & Texto"):
            st.markdown("""
            **Parte Data:** `EXTRACT(month FROM data)`
            **Truncar:** `DATE_TRUNC('month', data)`
            **Diferença:** `DATEDIFF('day', data1, data2)`
            **Maiúsc:** `UPPER(nome)`
            **Tamanho:** `LENGTH(nome)`
            """)

    with c2:
        st.info("Query Editor (Tabela = 'df')")
        q = st.text_area("SQL", "SELECT * FROM df LIMIT 10", height=250)
        if st.button("Executar (Ctrl+Enter)"):
            try:
                res = duckdb.query(q).to_df()
                st.dataframe(res, use_container_width=True)
            except Exception as e: st.error(f"Erro: {e}")

def page_academy():
    st.title("🎓 Academy: O Livro Aberto de Dados")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Estatística Fundamental", "SQL Avançado", "Python Eficiente", "Teoria de ML"])
    
    with tab1:
        st.markdown("### 📊 Estatística para Analistas")
        st.markdown("""
        **1. Medidas de Tendência Central**
        * **Média:** Soma tudo e divide. Sensível a outliers (salário do Neymar distorce a média do Brasil).
        * **Mediana:** O valor do meio. Robusta a outliers (se o Neymar entrar na sala, a mediana mal muda).
        * **Moda:** O valor que mais aparece.
        
        **2. Medidas de Dispersão**
        * **Desvio Padrão:** O quanto os dados fogem da média. Desvio baixo = dados consistentes. Alto = dados voláteis.
        * **Intervalo Interquartil (IQR):** A distância entre os 25% menores e os 25% maiores. Usado para achar outliers no Boxplot.
        
        **3. Distribuições**
        * **Normal (Gaussiana):** Formato de sino. Muita coisa na natureza segue isso (altura, peso).
        * **Log-Normal:** Cauda longa (Salários, Preços de Imóveis).
        
        **4. Teste de Hipótese (P-Value)**
        * Imagine que você mudou a cor do botão de "Comprar". As vendas subiram. Foi sorte ou foi o botão?
        * **P-Value < 0.05:** A chance de ter sido sorte é menor que 5%. Aceitamos que o botão funcionou.
        """)

    with tab2:
        st.markdown("### 💻 SQL Avançado")
        st.markdown("""
        **CTEs (Common Table Expressions)**
        Em vez de subqueries aninhadas impossíveis de ler, use `WITH`.
        ```sql
        WITH VendasMensais AS (
            SELECT DATE_TRUNC('month', data) as mes, SUM(valor) as total
            FROM vendas GROUP BY 1
        )
        SELECT * FROM VendasMensais WHERE total > 10000;
        ```
        
        **Joins Explicados**
        * **INNER JOIN:** Só traz o que tem match nos dois lados (Interseção).
        * **LEFT JOIN:** Traz TUDO da esquerda, e o que der match da direita (se não tiver, vem NULL). Fundamental para enriquecer dados sem perder linhas.
        * **FULL JOIN:** Traz tudo de todo mundo.
        
        **Window Functions (O Superpoder)**
        Permitem calcular coisas comparando a linha atual com outras, sem agrupar (sumir) com as linhas.
        * `LEAD()`: Olha o valor da próxima linha.
        * `LAG()`: Olha o valor da linha anterior (ótimo para calcular Growth MoM).
        """)
        
    with tab3:
        st.markdown("### 🐍 Python Eficiente")
        st.markdown("""
        **Evite Loops (for) no Pandas!**
        O Pandas é otimizado para operar vetores (colunas inteiras de uma vez).
        * ❌ `for i in df: ...` (Lento)
        * ✅ `df['col'] * 2` (Rápido)
        * ✅ `df.apply(funcao)` (Médio - use se não der vetorizado)
        
        **Loc vs Iloc**
        * `loc`: Busca por RÓTULO (Label). `df.loc['2023-01-01']`
        * `iloc`: Busca por POSIÇÃO (Index). `df.iloc[0]` (primeira linha)
        
        **Merge vs Concat**
        * `merge`: Junta lado a lado baseado em uma chave (ID). Igual SQL Join.
        * `concat`: Cola um embaixo do outro (empilhar meses de vendas) ou lado a lado (sem chave).
        """)

    with tab4:
        st.markdown("### 🤖 Machine Learning Desmistificado")
        st.markdown("""
        **Classificação vs Regressão**
        * O alvo é uma categoria (Gato/Cachorro, Churn/Não Churn)? **Classificação**.
        * O alvo é um número infinito (Preço, Temperatura)? **Regressão**.
        
        **Métricas de Erro**
        * **MAE (Erro Médio Absoluto):** "Em média, eu erro R$ 50,00". Fácil de explicar.
        * **RMSE (Raiz do Erro Quadrático):** "Em média eu erro... mas penalizo muito erros grandes". Se errar feio é inaceitável, use esse.
        
        **Bias-Variance Tradeoff**
        * **Underfitting (Viés):** O modelo é burro. Não aprendeu nem o treino. (Linha reta em dados curvos).
        * **Overfitting (Variância):** O modelo é "decorba". Ligou os pontos do treino, mas erra qualquer dado novo.
        """)

def page_ml_studio():
    st.title("🤖 ML Studio Transparente")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    c1, c2 = st.columns(2)
    target = c1.selectbox("Target (O que prever?)", df.columns)
    feats = c2.multiselect("Features (Variáveis)", [c for c in df.columns if c!=target])
    
    st.markdown("### ⚙️ Configuração de Hiperparâmetros")
    st.info("Hiperparâmetros são os 'botões de ajuste' do algoritmo. Eles controlam como ele aprende.")
    
    c_param1, c_param2 = st.columns(2)
    n_est = c_param1.slider("n_estimators (Random Forest)", 10, 300, 100)
    c_param1.caption("Quantas árvores de decisão criar. Mais árvores = mais estável, mas mais lento e pesado.")
    
    max_d = c_param2.slider("max_depth (Profundidade)", 2, 50, 10)
    c_param2.caption("O quão complexa cada árvore pode ser. Profundidade alta captura detalhes, mas causa Overfitting (decora os dados).")

    if st.button("Treinar e Analisar"):
        if not feats: st.error("Selecione features."); return
        try:
            X = df[feats]
            y = df[target]
            
            # Pipeline setup
            nums = X.select_dtypes(include=np.number).columns
            cats = X.select_dtypes(include=['object']).columns
            pre = ColumnTransformer([
                ('num', SimpleImputer(strategy='median'), nums),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cats)
            ])
            
            is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            
            if is_reg:
                y = y.fillna(y.mean())
                model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
                metric = "R²"
            else:
                y = y.fillna(y.mode()[0]).astype(str)
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                metric = "Acurácia"
                
            pipe = Pipeline([('pre', pre), ('model', model)])
            
            # Split
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_tr, y_tr)
            
            score_tr = pipe.score(X_tr, y_tr)
            score_te = pipe.score(X_te, y_te)
            
            st.divider()
            st.subheader("📊 Resultados")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Treino (Decorado)", f"{score_tr:.2f}")
            k2.metric("Teste (Realidade)", f"{score_te:.2f}")
            
            diff = score_tr - score_te
            if diff > 0.15:
                k3.error("⚠️ Overfitting Grave")
                st.warning(f"O modelo performou {diff:.0%} melhor no treino. Ele decorou os dados. Sugestão: Reduza o 'max_depth'.")
            elif diff < 0.05:
                k3.success("✅ Modelo Robusto")
            else:
                k3.warning("⚠️ Atenção Moderada")
                
            # Params JSON
            with st.expander("Ver Configuração Técnica (JSON)"):
                st.json(model.get_params())

        except Exception as e: st.error(f"Erro: {e}")

def page_report():
    st.title("📑 Relatório")
    charts = st.session_state['report_charts']
    df = st.session_state['df']
    
    if not charts:
        st.info("Nenhum gráfico salvo via Python Studio.")
    else:
        for i, ch in enumerate(charts):
            st.markdown(f"**{i+1}. {ch.get('title','Gráfico')}**")
            if ch['type'] == 'image_bytes':
                st.image(ch['data'])
            st.caption(ch.get('note', ''))
            if st.button(f"Remover {i}", key=f"del_{i}"):
                st.session_state['report_charts'].pop(i)
                st.rerun()

    if st.button("Gerar PDF"):
        try:
            kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
            pdf = generate_report_v13(df, charts, kpis)
            st.download_button("Baixar PDF", pdf, "relatorio_codefirst.pdf", "application/pdf")
        except Exception as e: st.error(f"Erro PDF: {e}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("🐍 Code-First v13")
        
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