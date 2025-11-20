"""
Enterprise Analytics — Masterclass Edition (v10.0)
Author: Gemini Advanced
Version: 10.0 (Tutorial Mode, Live Viz, ML Diagnostics, Advanced ETL)

Destaques v10.0:
- EDU: Modo Tutorial Interativo (Guias passo-a-passo).
- ETL: Agrupamento (GroupBy) e Ordenação.
- VIZ: Live Preview (sem botão gerar) + Cores customizadas.
- ML: Explicação de Hiperparâmetros, Detecção de Overfitting e Matriz de Confusão.
- SQL: Cheat Sheet (Colinha).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import logging
import re
import time
import pickle
import yaml
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

# --- Scientific Stack ---
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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
    page_title="Data Studio v10", 
    layout="wide", 
    page_icon="🎓", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Cards & Containers */
    .st-card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .tutorial-box {
        background-color: #fffbeb; border-left: 5px solid #f59e0b; 
        padding: 15px; border-radius: 8px; margin-bottom: 15px; font-size: 0.95rem;
    }
    .kpi-box {
        text-align: center; padding: 15px; background: #f8fafc; 
        border-radius: 8px; border: 1px solid #e2e8f0;
    }
    .kpi-val { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
    .kpi-lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    
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
# EVENT BUS
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
                elif op == "sort": df = df.sort_values(by=p['col'], ascending=p['asc'])
                elif op == "groupby":
                    df = df.groupby(p['gb_col'])[p['val_col']].agg(p['agg']).reset_index()
                    df.rename(columns={p['val_col']: f"{p['val_col']}_{p['agg']}"}, inplace=True)
                elif op == "dummies": df = pd.get_dummies(df, columns=p['cols'], drop_first=True, dtype=int)
            st.session_state['df'] = df
        except Exception as e: bus.emit("etl", "error", {"msg": str(e)}, "ERROR")

    def export_python(self) -> str:
        code = ["import pandas as pd", "df = pd.read_csv('data.csv')"]
        for step in self.get_steps():
            op = step['op']
            p = step['params']
            if op == "calc": code.append(f"df['{p['nm']}'] = df['{p['a']}'] {p['op']} df['{p['b']}']")
            elif op == "dropna": code.append("df = df.dropna()")
            elif op == "groupby": code.append(f"df = df.groupby('{p['gb_col']}')['{p['val_col']}'].agg('{p['agg']}').reset_index()")
        return "\n".join(code)

class SmartViz:
    @staticmethod
    def analyze(df: pd.DataFrame, x: str, y: str = None) -> Dict[str, Any]:
        analysis = {"recommended": "Scatter", "reason": "Padrão"}
        x_series = df[x]
        is_num_x = pd.api.types.is_numeric_dtype(x_series)
        is_date_x = pd.api.types.is_datetime64_any_dtype(x_series)
        
        if y is None:
            if is_num_x: return {"recommended": "Histograma", "reason": "Distribuição numérica."}
            return {"recommended": "Barras", "reason": "Contagem de categorias."}
        
        y_series = df[y]
        is_num_y = pd.api.types.is_numeric_dtype(y_series)
        
        if is_date_x and is_num_y: return {"recommended": "Linha", "reason": "Série temporal."}
        if not is_num_x and is_num_y: return {"recommended": "Barras", "reason": "Comparação categórica."}
        if is_num_x and is_num_y: return {"recommended": "Scatter", "reason": "Correlação numérica."}
        
        return analysis

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
        'tutorial_mode': False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()
pipeline_engine = PipelineEngine()
model_registry = ModelRegistry()

# ---------------------------
# UTILITIES
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
def get_demo_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2024-01-01', periods=n)
    cats = ['Eletrônicos', 'Casa', 'Roupas', 'Livros']
    reviews = ["Ótimo", "Ruim", "Excelente", "Péssimo", "Regular", "Não gostei", "Maravilhoso", "Demorou"]
    df = pd.DataFrame({
        'data': dates,
        'categoria': np.random.choice(cats, n),
        'vendas': np.random.uniform(50, 500, n),
        'quantidade': np.random.randint(1, 20, n),
        'nota': np.random.randint(1, 6, n),
        'comentario': np.random.choice(reviews, n)
    })
    df['lucro'] = df['vendas'] * 0.3
    # Intro de ruído para correlação não ser perfeita
    df['vendas'] = df['vendas'] + np.random.normal(0, 20, n)
    return df

# ---------------------------
# TUTORIAL HELPER
# ---------------------------
def render_tutorial(text: str):
    if st.session_state.get('tutorial_mode'):
        st.markdown(f"<div class='tutorial-box'>💡 <b>Tutorial:</b> {text}</div>", unsafe_allow_html=True)

# ---------------------------
# PDF ENGINE
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Masterclass Analytics Report v10.0', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v10(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = EnterprisePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Relatório Analítico", 0, 1, 'C')
    
    # KPIs
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Métricas Chave", 1, 1)
    pdf.set_font("Helvetica", "", 12)
    w = 45
    pdf.cell(w, 10, f"Linhas: {kpis['rows']}", 1)
    pdf.cell(w, 10, f"Colunas: {kpis['cols']}", 1)
    pdf.cell(w, 10, f"Nulos: {kpis['nulls']}", 1)
    pdf.ln(15)
    
    # Table Snapshot
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Amostra de Dados", 1, 1)
    pdf.set_font("Helvetica", "B", 8)
    head = df.head(10)
    cols = head.columns[:6]
    cw = 190 / len(cols)
    for c in cols: pdf.cell(cw, 8, str(c)[:12], 1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, r in head.iterrows():
        for c in cols: pdf.cell(cw, 8, str(r[c])[:12], 1)
        pdf.ln()
    pdf.ln(10)

    # Charts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Gráficos", 0, 1)
    import tempfile, os
    for i, ch in enumerate(charts):
        if i > 0 and i % 2 == 0: pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, ch['title'], 0, 1)
        try:
            img = ch['fig'].to_image(format="png", scale=1.0, engine="kaleido")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img)
                path = tmp.name
            pdf.image(path, x=20, w=170)
            os.unlink(path)
        except: pdf.cell(0, 10, "[Imagem indisponível - Instale 'kaleido']", 0, 1)
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------
# PAGES
# ---------------------------

def page_home():
    st.title("🏠 Home & Overview")
    render_tutorial("Aqui você tem uma visão 'de pássaro' dos seus dados. Verifique métricas básicas e distribuições para entender a qualidade antes de analisar.")
    
    df = st.session_state['df']
    
    if df.empty:
        st.markdown("### Comece aqui 👇")
        c1, c2 = st.columns(2)
        with c1:
            st.info("Para aprender, recomendamos usar os dados de demonstração que preparamos.")
            if st.button("🚀 Carregar Dados Demo (Tutorial)", type="primary"):
                df = get_demo_data()
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.session_state['tutorial_mode'] = True
                st.rerun()
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[0]:,}</div><div class='kpi-lbl'>Linhas</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[1]}</div><div class='kpi-lbl'>Colunas</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.duplicated().sum()}</div><div class='kpi-lbl'>Duplicatas</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.isna().mean().mean()*100:.1f}%</div><div class='kpi-lbl'>Nulos (Média)</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Detailed Describe
    t1, t2, t3 = st.tabs(["📋 Amostra", "🔢 Estatísticas (Numérico)", "🔤 Texto/Categorias"])
    
    with t1:
        st.dataframe(df.head(20), use_container_width=True)
    
    with t2:
        st.markdown("Resumo estatístico (`df.describe()`): Média, Desvio Padrão, Mínimo, Máximo, Quartis.")
        nums = df.select_dtypes(include=np.number)
        if not nums.empty:
            st.dataframe(nums.describe().T, use_container_width=True)
        else:
            st.info("Sem colunas numéricas.")

    with t3:
        st.markdown("Resumo categórico (`df.describe(include='object')`): Contagem, Únicos, Top valor.")
        cats = df.select_dtypes(include='object')
        if not cats.empty:
            st.dataframe(cats.describe().T, use_container_width=True)
        else:
            st.info("Sem colunas de texto.")

def page_academy():
    st.title("🎓 Data Academy")
    render_tutorial("Use esta aba como sua enciclopédia. Consulte sempre que tiver dúvida sobre um termo.")
    
    tab1, tab2 = st.tabs(["Dicionário Técnico", "Conceitos Intermediários"])
    
    with tab1:
        st.header("O Básico")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            - **DataFrame:** Tabela de dados (Excel bombado).
            - **Features (X):** As colunas que você usa para prever algo (ex: Idade, Renda).
            - **Target (Y):** A coluna que você quer prever (ex: Comprou?).
            """)
        with cols[1]:
            st.markdown("""
            - **Overfitting:** O modelo "decorou" os dados de treino e erra tudo em dados novos.
            - **Acurácia:** Taxa de acerto global (ex: 90% de acerto). Cuidado se os dados forem desbalanceados!
            """)

    with tab2:
        st.header("Nível Próximo")
        st.markdown("""
        ### Viés vs Variância
        - **Viés Alto (Underfitting):** O modelo é "burro demais", não aprendeu o padrão.
        - **Variância Alta (Overfitting):** O modelo é "sensível demais", aprendeu até o ruído.
        
        ### Matriz de Confusão
        Uma tabela que mostra onde o modelo errou:
        - Falso Positivo: Disse que era Câncer, mas não era (Susto).
        - Falso Negativo: Disse que não era Câncer, mas era (Perigo).
        """)

def page_etl():
    st.header("🛠️ ETL Studio")
    render_tutorial("Aqui transformamos os dados. Tente usar o 'Agrupar' para criar um resumo por Categoria.")
    
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    c1, c2 = st.columns([1, 4])
    if c1.button("↩️ Desfazer"): pipeline_engine.undo(); st.rerun()
    if c2.button("▶️ Replay Pipeline"): pipeline_engine.replay(); st.rerun()
    
    t1, t2, t3 = st.tabs(["Transformar", "Limpar/Filtrar", "Avançado (Group/Sort)"])
    
    with t1:
        st.subheader("Calculadora de Colunas")
        nums = df.select_dtypes(include=np.number).columns
        if len(nums)>0:
            c_a, c_op, c_b = st.columns([2,1,2])
            a = c_a.selectbox("A", nums)
            op = c_op.selectbox("Op", ["+","-","*","/"])
            b = c_b.selectbox("B", nums)
            nm = st.text_input("Nome", "res")
            if st.button("Calcular"):
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
        if st.button("Remover Nulos"):
            df = df.dropna()
            st.session_state['df'] = df
            pipeline_engine.add_step("dropna", {}, "Drop NA")
            st.rerun()
        
        cats = df.select_dtypes(include=['object']).columns
        if len(cats)>0:
            cols = st.multiselect("Dummies (Prep ML)", cats)
            if st.button("Aplicar Dummies") and cols:
                df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
                st.session_state['df'] = df
                pipeline_engine.add_step("dummies", {"cols":cols}, "Dummies")
                st.rerun()

    with t3:
        st.subheader("Agrupamento & Ordenação")
        mode = st.radio("Ação", ["Agrupar (GroupBy)", "Ordenar (Sort)"])
        
        if mode == "Agrupar (GroupBy)":
            gb_col = st.selectbox("Agrupar por (Categoria)", df.columns)
            val_col = st.selectbox("Coluna de Valor", nums)
            agg = st.selectbox("Função", ["sum", "mean", "count", "max", "min"])
            if st.button("Criar Agrupamento"):
                df = df.groupby(gb_col)[val_col].agg(agg).reset_index()
                df.rename(columns={val_col: f"{val_col}_{agg}"}, inplace=True)
                st.session_state['df'] = df
                pipeline_engine.add_step("groupby", {"gb_col":gb_col, "val_col":val_col, "agg":agg}, f"Group {gb_col}")
                st.rerun()
        else:
            sort_col = st.selectbox("Ordenar por", df.columns)
            asc = st.checkbox("Crescente?", value=True)
            if st.button("Ordenar"):
                df = df.sort_values(by=sort_col, ascending=asc)
                st.session_state['df'] = df
                pipeline_engine.add_step("sort", {"col":sort_col, "asc":asc}, f"Sort {sort_col}")
                st.rerun()

def page_viz():
    st.header("📈 Visual Studio")
    render_tutorial("Mude os eixos e veja o gráfico atualizar na hora. Use o seletor de cores para personalizar.")
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    l, r = st.columns([1,3])
    with l:
        st.subheader("Configuração")
        x = st.selectbox("Eixo X", df.columns)
        y = st.selectbox("Eixo Y", [None]+list(df.select_dtypes(include=np.number).columns))
        
        smart = SmartViz.analyze(df, x, y)
        st.info(f"💡 Dica: {smart['recommended']}")
        
        ct = st.selectbox("Tipo", ["Barras","Linha","Scatter","Histograma","Pizza","Boxplot","Heatmap"], index=["Barras","Linha","Scatter","Histograma","Pizza","Boxplot","Heatmap"].index(smart['recommended']))
        clr = st.selectbox("Legenda/Grupo", [None]+list(df.columns))
        
        color_discrete = st.color_picker("Cor Principal", "#2563eb")
        tt = st.text_input("Título", f"Análise de {x}")

    with r:
        # Live Render (No Button)
        try:
            if ct=="Barras": fig = px.bar(df, x=x, y=y, color=clr, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Linha": fig = px.line(df, x=x, y=y, color=clr, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Scatter": fig = px.scatter(df, x=x, y=y, color=clr, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Histograma": fig = px.histogram(df, x=x, color=clr, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Boxplot": fig = px.box(df, x=x, y=y, color=clr, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Pizza": fig = px.pie(df, names=x, values=y, title=tt, color_discrete_sequence=[color_discrete])
            elif ct=="Heatmap": fig = px.imshow(df.corr(numeric_only=True), text_auto=True, title=tt)
            
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns([3,1])
            note = c1.text_input("Nota para relatório")
            if c2.button("➕ Add Relatório"):
                st.session_state['report_charts'].append({"fig":fig, "title":tt, "note":note})
                st.toast("Adicionado!")
                
        except Exception as e: st.error(f"Aguardando configuração válida... ({e})")

def page_ml():
    st.header("🏆 ML Studio Pro")
    render_tutorial("Aqui você cria o cérebro da IA. Note como mostramos os 'Hiperparâmetros' (configurações) do modelo antes de treinar.")
    
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    t1, t2 = st.tabs(["Treino & Avaliação", "Model Registry"])
    
    with t1:
        c1, c2 = st.columns(2)
        target = c1.selectbox("Target (Alvo)", df.columns)
        feats = c2.multiselect("Features", [c for c in df.columns if c!=target])
        
        # Configuração Glass Box
        st.subheader("⚙️ Configuração do Modelo")
        model_type = st.selectbox("Algoritmo", ["RandomForest", "Logistic/Linear", "DecisionTree"])
        
        params = {}
        if "RandomForest" in model_type:
            n_est = st.slider("Nº de Árvores (n_estimators)", 10, 200, 50, help="Quantas árvores de decisão a floresta terá. Mais árvores = mais estável, mas mais lento.")
            depth = st.slider("Profundidade Máx (max_depth)", 2, 20, 10, help="O quão complexa cada árvore pode ficar. Muito profundo = risco de decorar (overfitting).")
            params = {"n_estimators": n_est, "max_depth": depth}
        
        if st.button("🚀 Treinar e Avaliar"):
            if not feats: st.error("Escolha features."); return
            
            with st.spinner("Treinando..."):
                try:
                    X = df[feats]
                    y = df[target]
                    
                    # Prep
                    nums = X.select_dtypes(include=np.number).columns
                    cats = X.select_dtypes(include=['object']).columns
                    pre = ColumnTransformer([
                        ('num', SimpleImputer(strategy='median'), nums),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), cats)
                    ])

                    is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
                    
                    # Model Selection
                    if is_reg:
                        y = y.fillna(y.mean())
                        if "Random" in model_type: model = RandomForestRegressor(**params)
                        elif "Linear" in model_type: model = LinearRegression()
                        else: model = DecisionTreeRegressor()
                        metric_name = "R²"
                    else:
                        y = y.fillna(y.mode()[0]).astype(str)
                        if "Random" in model_type: model = RandomForestClassifier(**params)
                        elif "Logistic" in model_type: model = LogisticRegression()
                        else: model = DecisionTreeClassifier()
                        metric_name = "Acurácia"

                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    pipe = Pipeline([('pre', pre), ('clf', model)])
                    pipe.fit(X_train, y_train)
                    
                    # Metrics
                    train_score = pipe.score(X_train, y_train)
                    test_score = pipe.score(X_test, y_test)
                    
                    # Display Diagnostics
                    st.markdown("### 📊 Diagnóstico do Modelo")
                    k1, k2, k3 = st.columns(3)
                    k1.metric(f"{metric_name} (Teste)", f"{test_score:.2f}")
                    k2.metric(f"{metric_name} (Treino)", f"{train_score:.2f}")
                    
                    # Overfitting Check
                    diff = train_score - test_score
                    if diff > 0.15:
                        k3.error(f"⚠️ Overfitting Alto (+{diff:.2f})")
                        st.warning("O modelo está muito melhor no treino do que no teste. Tente: 1) Reduzir Profundidade, 2) Mais dados, 3) Menos features.")
                    elif diff < 0.05:
                        k3.success("✅ Modelo Robusto")
                    else:
                        k3.warning("⚠️ Atenção Moderada")

                    # Advanced Plots
                    c_plot1, c_plot2 = st.columns(2)
                    with c_plot1:
                        if not is_reg:
                            st.markdown("**Matriz de Confusão**")
                            preds = pipe.predict(X_test)
                            cm = confusion_matrix(y_test, preds)
                            fig_cm = px.imshow(cm, text_auto=True, title="Erros vs Acertos")
                            st.plotly_chart(fig_cm, use_container_width=True)
                        else:
                            st.markdown("**Real vs Previsto**")
                            preds = pipe.predict(X_test)
                            fig_reg = px.scatter(x=y_test, y=preds, labels={'x':'Real', 'y':'Previsto'})
                            fig_reg.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
                            st.plotly_chart(fig_reg, use_container_width=True)

                    # Register
                    model_registry.register(pipe, f"{model_type}_{target}", {metric_name: test_score}, feats, params)
                    
                except Exception as e: st.error(f"Erro: {e}")

    with t2:
        models = model_registry.get_models()
        if models:
            st.dataframe(pd.DataFrame(models).drop(columns=['model']))
        else:
            st.info("Nenhum modelo salvo.")

def page_sql():
    st.header("💻 SQL & Colinha")
    render_tutorial("Use os botões abaixo para colar códigos comuns. O SQL é ótimo para filtragens complexas.")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB ausente."); return

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("### 📋 Colinha")
        if st.button("Top 10 Linhas"): st.session_state['sql_q'] = "SELECT * FROM df LIMIT 10"
        if st.button("Contagem por Categoria"): st.session_state['sql_q'] = "SELECT categoria, COUNT(*) FROM df GROUP BY categoria"
        if st.button("Filtro Simples"): st.session_state['sql_q'] = "SELECT * FROM df WHERE vendas > 100"
        if st.button("Média e Soma"): st.session_state['sql_q'] = "SELECT SUM(vendas), AVG(custo) FROM df"

    with c1:
        q = st.text_area("Query (tabela = 'df')", value=st.session_state.get('sql_q', "SELECT * FROM df LIMIT 10"), height=200)
        if st.button("Executar"):
            try:
                res = duckdb.query(q).to_df()
                st.dataframe(res, use_container_width=True)
            except Exception as e: st.error(e)

def page_report():
    st.header("📑 Relatório")
    charts = st.session_state['report_charts']
    df = st.session_state['df']
    
    if not charts: st.info("Vazio."); return

    for i, ch in enumerate(charts):
        st.markdown(f"**{ch['title']}**")
        st.plotly_chart(ch['fig'], use_container_width=True)
        if st.button("Remover", key=f"del{i}"):
             st.session_state['report_charts'].pop(i); st.rerun()

    if st.button("Gerar PDF"):
        kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
        pdf = generate_report_v10(df, charts, kpis)
        st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("💎 Masterclass v10")
        st.caption("Aprenda Data Science na Prática")
        
        if st.checkbox("🎓 Modo Tutorial", value=st.session_state.get('tutorial_mode', False)):
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
                    df = clean_colnames(df)
                    st.session_state['df'] = df
                    st.session_state['df_raw'] = df.copy()
                    st.session_state['last_uid'] = uid
                    bus.emit("sys", "load", {"msg": "Dados carregados"}, "SUCCESS")
                    st.rerun()
                except Exception as e: st.error(e)
        
        st.markdown("---")
        menu = st.radio("Menu", ["🏠 Home", "🎓 Academy", "🛠️ ETL Studio", "💻 SQL & Colinha", "📈 Visual Studio", "🏆 ML Studio", "📑 Relatório"])
        st.markdown("---")
        if st.button("Hard Reset"):
            st.session_state.clear()
            st.rerun()

    if menu == "🏠 Home": page_home()
    elif menu == "🎓 Academy": page_academy()
    elif menu == "🛠️ ETL Studio": page_etl()
    elif menu == "💻 SQL & Colinha": page_sql()
    elif menu == "📈 Visual Studio": page_viz()
    elif menu == "🏆 ML Studio": page_ml()
    elif menu == "📑 Relatório": page_report()

if __name__ == "__main__":
    main()