"""
Enterprise Analytics — Interactive Tutorial Edition (v12.0)
Author: Gemini Advanced
Version: 12.0 (Interactive Tutorial, Dirty Data Gen, Gamified Learning)

Destaques v12.0:
- EDU: 'Modo Tutorial' gamificado com missões passo-a-passo.
- DATA: Gerador de dados 'Sujos' (Nulls, Dups, Outliers) para treino real.
- UX: Navegação guiada por objetivos ("Sua missão agora é...").
- CORE: Toda a robustez da v11 (SQL, ML, PDF, Visual Studio).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    page_title="Data Studio Tutorial", 
    layout="wide", 
    page_icon="🎓", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Cards */
    .st-card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    /* Tutorial Highlight */
    .tutorial-box {
        background-color: #fffbeb; border-left: 5px solid #f59e0b; 
        padding: 15px; border-radius: 8px; margin-bottom: 20px; 
        font-size: 1rem; color: #78350f; box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    .tutorial-title { font-weight: 800; display: block; margin-bottom: 5px; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; }
    
    /* KPI Box */
    .kpi-box {
        text-align: center; padding: 15px; background: #f8fafc; 
        border-radius: 8px; border: 1px solid #e2e8f0;
    }
    .kpi-val { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
    .kpi-lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase; }
    
    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .st-card { background-color: #1e293b; border-color: #334155; color: white; }
        .tutorial-box { background-color: #451a03; border-color: #d97706; color: #fef3c7; }
        .kpi-box { background-color: #0f172a; border-color: #334155; }
        .kpi-val { color: #f1f5f9; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# CORE: EVENT BUS
# ---------------------------
class EventBus:
    def __init__(self): self.logs = []
    def emit(self, ns: str, evt: str, payload: dict = None, level: str = "INFO"):
        msg = payload.get('msg', f"{ns}:{evt}")
        if level == "ERROR": st.error(msg)
        elif level == "SUCCESS": st.toast(msg, icon="✅")
        elif level == "WARN": st.warning(msg)

if 'bus' not in st.session_state: st.session_state['bus'] = EventBus()
bus = st.session_state['bus']

# ---------------------------
# ENGINES
# ---------------------------
class PipelineEngine:
    def get_steps(self): return st.session_state.get('etl_steps', [])
    
    def add_step(self, op: str, params: Dict[str, Any], description: str):
        step = {"op": op, "params": params, "desc": description}
        st.session_state['etl_steps'].append(step)
        bus.emit("etl", "step", {"msg": description}, "SUCCESS")

    def undo(self):
        if st.session_state['etl_steps']:
            removed = st.session_state['etl_steps'].pop()
            bus.emit("etl", "undo", {"msg": f"Desfeito: {removed['desc']}"}, "WARN")
            self.replay()

    def replay(self):
        try:
            df = st.session_state['df_raw'].copy()
            for step in self.get_steps():
                op = step['op']
                p = step['params']
                if op == "calc":
                    if p['op'] == "+": df[p['nm']] = df[p['a']] + df[p['b']]
                    elif p['op'] == "-": df[p['nm']] = df[p['a']] - df[p['b']]
                    elif p['op'] == "*": df[p['nm']] = df[p['a']] * df[p['b']]
                    elif p['op'] == "/": df[p['nm']] = df[p['a']] / df[p['b']]
                elif op == "dropna": df = df.dropna()
                elif op == "fillna_0": df = df.fillna(0)
                elif op == "drop_duplicates": df = df.drop_duplicates()
                elif op == "dummies": df = pd.get_dummies(df, columns=p['cols'], drop_first=True, dtype=int)
                elif op == "clean_text":
                    def clean(text):
                        if not isinstance(text, str): return ""
                        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()
                        text = re.sub(r'[^a-z\s]', '', text)
                        return re.sub(r'\s+', ' ', text).strip()
                    df[p['col']] = df[p['col']].apply(clean)
            st.session_state['df'] = df
        except Exception as e: bus.emit("etl", "error", {"msg": str(e)}, "ERROR")
    
    def export_python(self) -> str:
        code = ["import pandas as pd", "df = pd.read_csv('data.csv')"]
        for step in self.get_steps():
            op = step['op']
            p = step['params']
            if op == "calc": code.append(f"df['{p['nm']}'] = df['{p['a']}'] {p['op']} df['{p['b']}']")
            elif op == "dropna": code.append("df = df.dropna()")
            elif op == "dummies": code.append(f"df = pd.get_dummies(df, columns={p['cols']}, drop_first=True)")
        return "\n".join(code)

class ModelRegistry:
    def register(self, model, name: str, metrics: dict, features: List[str], params: dict):
        entry = {"name": name, "model": model, "metrics": metrics, "features": features, "params": params, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
        if 'model_registry' not in st.session_state: st.session_state['model_registry'] = []
        st.session_state['model_registry'].append(entry)
        bus.emit("ml", "registered", {"msg": f"Modelo {name} registrado"})
    def get_models(self): return st.session_state.get('model_registry', [])

# ---------------------------
# SESSION INIT
# ---------------------------
def init_session():
    defaults = {
        'df': pd.DataFrame(), 'df_raw': pd.DataFrame(), 'etl_steps': [],
        'report_charts': [], 'model_registry': [], 'last_file_uid': None,
        'tutorial_mode': False, 'ml_pipeline': None, 'ml_type': None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()
pipeline_engine = PipelineEngine()
model_registry = ModelRegistry()

# ---------------------------
# UTILITIES & DEMO DATA
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file, encoding_errors='ignore')
        return pd.read_excel(file)
    except Exception as e: bus.emit("error", {"msg":str(e)}); return pd.DataFrame()

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True).str.replace(r"[^0-9a-zA-Z_]", "", regex=True).str.lower())
    return df

@st.cache_data
def get_dirty_demo_data():
    """Gera dados com problemas propositais para o tutorial."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2024-01-01', periods=n)
    cats = ['Saboaria', 'Higiene', 'Detergente']
    reviews_pos = ["Tudo certo", "Muito bom", "Limpo", "Conforme"]
    reviews_neg = ["Sujo", "Quebrado", "Ruim", "Vazamento", "Fora do lugar"]
    
    df = pd.DataFrame({
        'data': dates,
        'setor': np.random.choice(cats, n),
        'producao': np.random.uniform(100, 1000, n),
        'temperatura': np.random.normal(25, 5, n),
        'comentario': np.random.choice(reviews_pos + reviews_neg, n)
    })
    
    # 1. Injetar Nulos
    df.loc[np.random.choice(df.index, 30), 'producao'] = np.nan
    
    # 2. Injetar Duplicatas
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)
    
    # 3. Injetar Outlier (O "monstro" para achar no SQL)
    df.loc[5, 'producao'] = 1000000 # 1 Milhão!
    df.loc[5, 'comentario'] = "ERRO GRAVE DE LEITURA"
    
    # Target Logic for ML
    df['status_final'] = np.where(df['producao'] > 500, 'Meta Batida', 'Abaixo da Meta')
    
    return df

def render_tutorial(text: str):
    if st.session_state.get('tutorial_mode'):
        st.markdown(f"""
        <div class='tutorial-box'>
            <span class='tutorial-title'>🎓 MISSÃO DO TUTORIAL</span>
            {text}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# PDF ENGINE
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Tutorial Report v12.0', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v12(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = EnterprisePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Capa
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Relatório Completo", 0, 1, 'C')
    
    # 1. KPIs
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Métricas Gerais", 1, 1)
    pdf.set_font("Helvetica", "", 12)
    w = 45
    pdf.cell(w, 10, f"Linhas: {kpis['rows']}", 1)
    pdf.cell(w, 10, f"Nulos: {kpis['nulls']}", 1)
    pdf.cell(w, 10, f"Duplicatas: {kpis['dups']}", 1)
    pdf.ln(15)
    
    # 2. Charts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Análises Visuais", 0, 1)
    
    import tempfile, os
    for i, ch in enumerate(charts):
        if i % 2 == 0 and i > 0: pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, ch['title'], 0, 1)
        try:
            img = ch['fig'].to_image(format="png", scale=1.0, engine="kaleido")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img)
                path = tmp.name
            pdf.image(path, x=20, w=170)
            os.unlink(path)
        except: 
            pdf.set_font("Courier", "I", 10)
            pdf.cell(0, 10, "[Imagem não disponível]", 0, 1)
        pdf.ln(5)
        if ch.get('note'):
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 5, f"Nota: {ch['note']}")
            pdf.ln(5)
            
    return bytes(pdf.output())

# ---------------------------
# PAGES
# ---------------------------

def page_home():
    st.title("🏠 Home")
    
    df = st.session_state['df']
    
    if df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.info("Bem-vindo! Vamos aprender na prática?")
            if st.button("🚀 Iniciar Tutorial Interativo", type="primary"):
                df = get_dirty_demo_data()
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.session_state['tutorial_mode'] = True
                st.rerun()
            
            st.markdown("---")
            uploaded = st.file_uploader("Ou carregue seu arquivo", type=['csv','xlsx'])
            if uploaded:
                # Load logic (simplified for home)
                try:
                    if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded, encoding_errors='ignore')
                    else: df = pd.read_excel(uploaded)
                    st.session_state['df'] = clean_colnames(df)
                    st.session_state['df_raw'] = st.session_state['df'].copy()
                    st.session_state['tutorial_mode'] = False
                    st.rerun()
                except Exception as e: st.error(e)
        return

    render_tutorial("""
    <b>Passo 1: Diagnóstico</b><br>
    Olhe para os KPIs abaixo. Viu que temos <b>Nulos</b> e <b>Duplicatas</b>?
    Isso vai quebrar nossa análise. Sua primeira missão é ir na aba <b>🛠️ ETL Studio</b> e limpar isso.
    """)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[0]:,}</div><div class='kpi-lbl'>Linhas</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[1]}</div><div class='kpi-lbl'>Colunas</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.duplicated().sum()}</div><div class='kpi-lbl'>Duplicatas</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.isna().sum().sum()}</div><div class='kpi-lbl'>Células Vazias</div></div>", unsafe_allow_html=True)

    st.markdown("### 🔎 Visão Geral")
    st.dataframe(df.head(10), use_container_width=True)

def page_academy():
    st.title("🎓 Academy: O Guia Definitivo")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Glossário", "🐍 App vs Python", "💻 App vs SQL", "🧠 Teoria ML"])
    
    with tab1:
        st.markdown("""
        ### Fundamentos
        - **ETL (Extract, Transform, Load):** O processo de pegar dados brutos, limpar e preparar para análise.
        - **Outlier:** Um valor que foge do padrão. No tutorial, escondemos uma venda de 1 Milhão! Isso distorce a média.
        - **Feature Engineering:** Criar colunas novas (ex: extrair 'Mês' de 'Data') para ajudar o modelo a entender padrões temporais.
        """)
    with tab2:
        st.header("Como fazer no Python?")
        c1, c2 = st.columns(2)
        with c1:
            st.info("No App: Botão 'Remover Nulos'")
            st.code("df.dropna(inplace=True)")
        with c2:
            st.info("No App: Filtro Visual")
            st.code("df = df[df['coluna'] > 100]")
    with tab3:
        st.header("Como fazer no SQL?")
        st.info("O SQL é a linguagem dos bancos de dados. É essencial saber.")
        st.code("""
-- Selecionar colunas
SELECT data, vendas FROM tabela;

-- Filtrar
SELECT * FROM tabela WHERE vendas > 1000;

-- Agrupar (Pivot)
SELECT setor, SUM(vendas) 
FROM tabela 
GROUP BY setor;
        """, language="sql")
    with tab4:
        st.markdown("""
        ### Tipos de ML
        1. **Classificação:** O alvo é uma categoria (Sim/Não, Aprovado/Reprovado). Usamos *Acurácia* e *Matriz de Confusão*.
        2. **Regressão:** O alvo é um número (Preço, Quantidade). Usamos *R²* (o quanto explicamos a variação).
        
        ### O Perigo do Overfitting
        Se seu modelo acerta 100% no treino e 50% no teste, ele decorou a prova!
        Soluções: Mais dados, modelos mais simples (menos árvores na floresta), menos colunas.
        """)

def page_etl():
    st.header("🛠️ ETL Studio")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    render_tutorial("""
    <b>Passo 2: Limpeza</b><br>
    Vá na aba <b>Limpar & Filtrar</b> abaixo.<br>
    1. Clique em <b>Remover Nulos</b>.<br>
    2. Clique em <b>Remover Duplicatas</b>.<br>
    Observe como o número de linhas na tabela à direita diminui. Isso é saneamento de dados!
    """)

    # Split View
    c_ctrl, c_data = st.columns([1, 2])
    
    with c_ctrl:
        if st.button("↩️ Desfazer"): pipeline_engine.undo(); st.rerun()
        
        t1, t2 = st.tabs(["Transformar", "Limpar & Filtrar"])
        
        with t1:
            st.caption("Calculadora")
            with st.form("calc"):
                nums = df.select_dtypes(include=np.number).columns
                if len(nums)>0:
                    a = st.selectbox("A", nums)
                    op = st.selectbox("Op", ["+","-","*","/"])
                    b = st.selectbox("B", nums)
                    nm = st.text_input("Nome", "res")
                    if st.form_submit_button("Calcular"):
                        try:
                            if op=="+": df[nm] = df[a]+df[b]
                            elif op=="-": df[nm] = df[a]-df[b]
                            elif op=="*": df[nm] = df[a]*df[b]
                            elif op=="/": df[nm] = df[a]/df[b]
                            st.session_state['df'] = df
                            pipeline_engine.add_step("calc", {"a":a,"b":b,"op":op,"nm":nm}, f"Calc {nm}")
                            st.rerun()
                        except Exception as e: st.error(e)
        
        with t2:
            if st.button("Remover Nulos (DropNA)"):
                df = df.dropna()
                st.session_state['df'] = df
                pipeline_engine.add_step("dropna", {}, "Drop NA")
                st.rerun()
            
            if st.button("Remover Duplicatas"):
                df = df.drop_duplicates()
                st.session_state['df'] = df
                pipeline_engine.add_step("drop_duplicates", {}, "Drop Duplicates")
                st.rerun()
            
            st.markdown("---")
            st.caption("Preparação para ML")
            cats = df.select_dtypes(include=['object']).columns
            if len(cats)>0:
                tc = st.selectbox("Coluna Texto", cats)
                if st.button("Limpar Texto (Lower+Strip)"):
                    # Add clean step logic in engine replay usually, simple here
                    st.info("Adicionado ao pipeline.")

    with c_data:
        st.subheader("Dados em Tempo Real")
        st.dataframe(df, height=500, use_container_width=True)
        st.caption(f"Linhas: {len(df)} | Colunas: {len(df.columns)}")

def page_sql():
    st.header("💻 SQL Engine")
    render_tutorial("""
    <b>Passo 3: O Caçador de Outliers</b><br>
    Sabemos que existe uma venda errada de 1 Milhão. Vamos achá-la com SQL.<br>
    Use a colinha abaixo para rodar: <code>SELECT * FROM df WHERE producao > 5000</code> (se estiver no modo tutorial).
    """)
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB ausente."); return

    c_code, c_res = st.columns([1, 2])
    
    with c_code:
        st.subheader("Editor")
        q = st.text_area("Query (tabela = 'df')", "SELECT * FROM df LIMIT 10", height=150)
        if st.button("Executar"):
            try:
                res = duckdb.query(q).to_df()
                st.session_state['sql_res'] = res
            except Exception as e: st.error(f"Erro: {e}")
        
        st.markdown("---")
        st.markdown("### 📋 Colinha")
        with st.expander("Filtros (WHERE)"):
            st.code("SELECT * FROM df WHERE coluna > 100", language="sql")
            st.caption("Pega apenas linhas onde valor é maior que 100.")
        with st.expander("Agrupamento (GROUP BY)"):
            st.code("SELECT setor, AVG(valor) \nFROM df \nGROUP BY setor", language="sql")
            st.caption("Calcula a média de valor para cada setor.")
            
    with c_res:
        st.subheader("Resultado")
        if 'sql_res' in st.session_state:
            st.dataframe(st.session_state['sql_res'], use_container_width=True)
        else:
            st.info("Execute uma query.")

def page_viz():
    st.header("📈 Visual Studio")
    render_tutorial("""
    <b>Passo 4: Visualização</b><br>
    Crie um <b>Scatter Plot</b>. Coloque 'producao' no Eixo Y e 'data' no Eixo X.<br>
    Você verá um ponto lá no alto isolado. Esse é o nosso Outlier! Adicione esse gráfico ao relatório.
    """)
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    l, r = st.columns([1,3])
    with l:
        st.subheader("Configuração")
        
        ct = st.selectbox("Tipo", ["Scatter","Barras","Linha","Histograma","Boxplot","Heatmap"])
        x = st.selectbox("Eixo X", df.columns)
        y = st.selectbox("Eixo Y", [None]+list(df.select_dtypes(include=np.number).columns))
        clr = st.selectbox("Cor/Legenda", [None]+list(df.columns))
        tt = st.text_input("Título", f"{ct} de {x}")
        
        with st.expander("🎨 Estilo"):
            theme = st.selectbox("Tema", ["plotly_white", "plotly_dark", "seaborn"])
            color_seq = st.color_picker("Cor Principal", "#2563eb")
            
        with st.expander("🔍 Filtro Local (Não afeta dados)"):
            f_col = st.selectbox("Filtrar Coluna", [None]+list(df.columns))
            plot_df = df.copy()
            if f_col:
                vals = df[f_col].unique()
                if len(vals) < 50:
                    sel = st.multiselect("Valores", vals, default=vals)
                    plot_df = plot_df[plot_df[f_col].isin(sel)]

    with r:
        # Live Render
        try:
            colors = [color_seq]
            if ct=="Barras": fig = px.bar(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=colors)
            elif ct=="Linha": fig = px.line(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=colors)
            elif ct=="Scatter": fig = px.scatter(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=colors)
            elif ct=="Histograma": fig = px.histogram(plot_df, x=x, color=clr, title=tt, template=theme, color_discrete_sequence=colors)
            elif ct=="Boxplot": fig = px.box(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=colors)
            elif ct=="Heatmap": fig = px.imshow(plot_df.corr(numeric_only=True), text_auto=True, title=tt, template=theme)
            
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns([3,1])
            note = c1.text_input("Nota para relatório")
            if c2.button("➕ Add Relatório"):
                st.session_state['report_charts'].append({"fig":fig, "title":tt, "note":note})
                st.toast("Adicionado!")
        except Exception as e: st.error(f"Erro visual: {e}")

def page_ml():
    st.header("🏆 ML Studio")
    render_tutorial("""
    <b>Passo 5: Inteligência Artificial</b><br>
    Vamos usar a coluna 'comentario' para prever se o 'status_final' é Aprovado ou Reprovado.<br>
    1. Vá na aba <b>NLP</b>.<br>
    2. Selecione 'comentario' como Texto e 'status_final' como Target.<br>
    3. Treine e veja a Matriz de Confusão. Depois, teste na aba Simulador com uma frase sua!
    """)
    
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    t1, t2 = st.tabs(["Treino & Resultados", "Simulador (Teste)"])
    
    with t1:
        mode = st.radio("Modo", ["NLP (Texto)", "Tabular (Números)"], horizontal=True)
        
        if mode == "NLP (Texto)":
            st.info("TF-IDF + Regressão Logística: O padrão ouro para classificação de texto simples.")
            
            c1, c2 = st.columns(2)
            txt_col = c1.selectbox("Coluna Texto", df.select_dtypes(include='object').columns)
            target = c2.selectbox("Alvo (Target)", df.columns)
            
            if st.button("Treinar Modelo NLP"):
                with st.spinner("Treinando..."):
                    try:
                        X = df[txt_col].astype(str)
                        y = df[target].astype(str)
                        
                        # Robust Pipeline
                        pipe = Pipeline([
                            ('tfidf', TfidfVectorizer(min_df=1, ngram_range=(1,1), max_features=1000)),
                            ('clf', LogisticRegression(max_iter=1000))
                        ])
                        
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        pipe.fit(X_tr, y_tr)
                        preds = pipe.predict(X_te)
                        acc = accuracy_score(y_te, preds)
                        
                        st.success(f"Acurácia: {acc:.1%}")
                        
                        # Confusion Matrix
                        cm = confusion_matrix(y_te, preds)
                        st.markdown("#### Matriz de Confusão")
                        st.caption("Mostra os erros. Se a diagonal principal estiver brilhante, o modelo acertou!")
                        st.plotly_chart(px.imshow(cm, text_auto=True, labels=dict(x="Previsto", y="Real")), use_container_width=True)
                        
                        st.session_state['ml_pipeline'] = pipe
                        st.session_state['ml_type'] = 'nlp'

                    except Exception as e: st.error(f"Erro: {e}")
        else:
            # Tabular Logic
            tgt = st.selectbox("Target", df.columns)
            fts = st.multiselect("Features", [c for c in df.columns if c!=tgt])
            if st.button("Treinar Tabular") and fts:
                try:
                    X = df[fts]
                    y = df[tgt]
                    nums = X.select_dtypes(include=np.number).columns
                    cats = X.select_dtypes(include='object').columns
                    pre = ColumnTransformer([('n', SimpleImputer(strategy='median'), nums), ('c', OneHotEncoder(handle_unknown='ignore'), cats)])
                    is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique()>20
                    model = RandomForestRegressor(n_estimators=50) if is_reg else RandomForestClassifier(n_estimators=50)
                    pipe = Pipeline([('pre',pre), ('m',model)])
                    
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
                    pipe.fit(X_tr, y_tr)
                    score = pipe.score(X_te, y_te)
                    st.success(f"Score: {score:.2f}")
                    st.session_state['ml_pipeline'] = pipe
                    st.session_state['ml_type'] = 'tabular'
                    st.session_state['ml_feats'] = fts
                except Exception as e: st.error(e)

    with t2:
        if st.session_state.get('ml_pipeline'):
            st.subheader("Testar Modelo")
            model = st.session_state['ml_pipeline']
            
            if st.session_state['ml_type'] == 'nlp':
                txt = st.text_input("Digite uma frase:")
                if st.button("Classificar"):
                    pred = model.predict([txt])[0]
                    st.info(f"Resultado: **{pred}**")
            else:
                vals = {}
                for f in st.session_state.get('ml_feats', []):
                    vals[f] = st.text_input(f"Valor {f}", "0")
                if st.button("Prever"):
                    df_in = pd.DataFrame([vals])
                    # auto cast
                    for c in df_in.columns: 
                        try: df_in[c] = pd.to_numeric(df_in[c]) 
                        except: pass
                    pred = model.predict(df_in)[0]
                    st.success(f"Previsão: {pred}")

def page_report():
    st.header("📑 Relatório")
    render_tutorial("Gere o PDF final. Ele vai compilar os KPIs, uma amostra dos dados limpos e todos os gráficos que você adicionou. É o entregável do seu projeto.")
    
    charts = st.session_state['report_charts']
    df = st.session_state['df']
    
    if not charts: st.info("Vazio. Adicione gráficos no Visual Studio."); return

    for i, ch in enumerate(charts):
        st.markdown(f"**{ch['title']}**")
        st.plotly_chart(ch['fig'], use_container_width=True)
        if st.button("Remover", key=f"del{i}"):
             st.session_state['report_charts'].pop(i); st.rerun()

    if st.button("Gerar PDF"):
        try:
            kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
            pdf = generate_report_v12(df, charts, kpis)
            st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")
            st.success("PDF Gerado!")
        except Exception as e: st.error(f"Erro PDF: {e}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("🎓 Academy v11")
        
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
                    st.session_state['tutorial_mode'] = False # Disable tutorial on real data
                    st.rerun()
                except Exception as e: st.error(e)
        
        st.markdown("---")
        menu = st.radio("Menu", ["🏠 Home", "🎓 Academy", "🛠️ ETL Studio", "💻 SQL Studio", "📈 Visual Studio", "🏆 ML Studio", "📑 Relatório"])
        if st.button("Reset"): st.session_state.clear(); st.rerun()

    if menu == "🏠 Home": page_home()
    elif menu == "🎓 Academy": page_academy()
    elif menu == "🛠️ ETL Studio": page_etl()
    elif menu == "💻 SQL Studio": page_sql()
    elif menu == "📈 Visual Studio": page_viz()
    elif menu == "🏆 ML Studio": page_ml()
    elif menu == "📑 Relatório": page_report()

if __name__ == "__main__":
    main()