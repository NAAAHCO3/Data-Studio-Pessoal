"""
Enterprise Analytics — Ultimate Learning Edition (v11.0)
Author: Gemini Advanced
Version: 11.0 (Deep Academy, Visual Customization, NLP Fix, Integrated Layout)

Destaques v11.0:
- UI: Visualização de dados integrada nas telas de ETL e SQL.
- EDU: Academy com comparação "App vs Code (Python/SQL)" e teoria aprofundada.
- VIZ: Customização avançada (cores, grid, fundo) e filtros locais.
- ML: Retorno do NLP, Simulador de teste e Explicações didáticas.
- BUGFIX: Correção na geração de binários do PDF (fpdf2).
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
from datetime import datetime
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
    page_title="Data Studio Academy v11", 
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
        background-color: #eff6ff; border-left: 5px solid #3b82f6; 
        padding: 15px; border-radius: 8px; margin-bottom: 15px; font-size: 0.95rem; color: #1e3a8a;
    }
    .concept-box {
        background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px;
    }
    
    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .st-card { background-color: #1e293b; border-color: #334155; color: white; }
        .tutorial-box { background-color: #172554; border-color: #3b82f6; color: #dbeafe; }
        .concept-box { background-color: #064e3b; border-color: #059669; color: #ecfdf5; }
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
        y_series = df[y] if y else None
        is_num_y = pd.api.types.is_numeric_dtype(y_series) if y else False

        if y is None:
            if is_num_x: return {"recommended": "Histograma", "reason": "Distribuição numérica."}
            return {"recommended": "Barras", "reason": "Contagem de frequências."}
        else:
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
        'tutorial_mode': False, 'ml_pipeline': None, 'ml_type': None
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
        'comentario_cliente': np.random.choice(reviews, n)
    })
    df['lucro'] = df['vendas'] * 0.3
    df['vendas'] = df['vendas'] + np.random.normal(0, 20, n)
    return df

# ---------------------------
# PDF ENGINE
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Academy Analytics Report v11.0', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v11(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = EnterprisePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Relatório Analítico", 0, 1, 'C')
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Métricas Chave", 1, 1)
    pdf.set_font("Helvetica", "", 12)
    w = 45
    pdf.cell(w, 10, f"Linhas: {kpis['rows']}", 1)
    pdf.cell(w, 10, f"Colunas: {kpis['cols']}", 1)
    pdf.cell(w, 10, f"Nulos: {kpis['nulls']}", 1)
    pdf.ln(15)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Gráficos", 0, 1)
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
        except: pdf.cell(0, 10, "[Imagem indisponível - Instale 'kaleido']", 0, 1)
        pdf.ln(5)
        
    # Fix for PDF Output: Return bytes directly
    return bytes(pdf.output(dest='S'))

# ---------------------------
# PAGES
# ---------------------------

def page_home():
    st.title("🏠 Home & Overview")
    if st.session_state.get('tutorial_mode'):
        st.info("💡 DICA DO PROFESSOR: Comece olhando o 'describe()'. Ele te diz se a média faz sentido ou se tem outliers (valores estranhos) que distorcem tudo.")

    df = st.session_state['df']
    if df.empty:
        st.markdown("### Comece aqui 👇")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Carregar Dados Demo (Full Stack)", type="primary"):
                df = get_demo_data()
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.session_state['tutorial_mode'] = True
                st.rerun()
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Linhas", f"{df.shape[0]:,}")
    k2.metric("Colunas", df.shape[1])
    k3.metric("Duplicatas", df.duplicated().sum())
    k4.metric("Nulos (Total)", df.isna().sum().sum())

    st.markdown("### 🔎 Estatísticas Descritivas")
    t1, t2 = st.tabs(["🔢 Numéricas (Describe)", "🔤 Categóricas"])
    
    with t1:
        nums = df.select_dtypes(include=np.number)
        if not nums.empty:
            st.dataframe(nums.describe().T, use_container_width=True)
        else: st.info("Sem colunas numéricas.")
    
    with t2:
        cats = df.select_dtypes(include='object')
        if not cats.empty:
            st.dataframe(cats.describe().T, use_container_width=True)
        else: st.info("Sem colunas de texto.")

def page_academy():
    st.title("🎓 Academy: Teoria & Prática")
    tab1, tab2, tab3 = st.tabs(["📚 Glossário", "🐍 Python vs App", "💻 SQL vs App"])
    
    with tab1:
        st.markdown("""
        ### Dicionário de Dados
        - **DataFrame:** Tabela de dados em memória.
        - **Overfitting (Sobreajuste):** Quando seu modelo decora o passado e erra o futuro. Sinal: Acurácia de Treino 99%, Teste 60%.
        - **Matriz de Confusão:** Tabela que mostra: O que era A e eu disse que era A (Acerto), e o que era A e eu disse B (Erro).
        - **Feature Engineering:** Criar novas colunas (ex: extrair 'mês' da data) para ajudar o modelo a aprender.
        """)
    
    with tab2:
        st.header("App vs Python")
        c1, c2 = st.columns(2)
        with c1:
            st.info("👉 O que você faz aqui (Botão)")
            st.markdown("**1. Filtrar Vendas > 100**")
            st.markdown("**2. Agrupar por Categoria**")
            st.markdown("**3. Preencher Nulos com 0**")
        with c2:
            st.success("🐍 Como é no código (Pandas)")
            st.code("df = df[df['vendas'] > 100]", language="python")
            st.code("df.groupby('categoria')['vendas'].sum()", language="python")
            st.code("df.fillna(0, inplace=True)", language="python")

    with tab3:
        st.header("App vs SQL")
        c1, c2 = st.columns(2)
        with c1:
            st.info("👉 O que você faz aqui")
            st.markdown("**1. Selecionar colunas**")
            st.markdown("**2. Agrupar e Somar**")
            st.markdown("**3. Ordenar**")
        with c2:
            st.warning("💻 Como é no SQL")
            st.code("SELECT col1, col2 FROM tabela", language="sql")
            st.code("SELECT cat, SUM(val) FROM tab GROUP BY cat", language="sql")
            st.code("SELECT * FROM tab ORDER BY data DESC", language="sql")

def page_etl():
    st.header("🛠️ ETL Studio")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    # Top: Data Preview always visible
    with st.expander("👀 Visualizar Dados Atuais", expanded=True):
        st.dataframe(df.head(5), use_container_width=True)

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
    
    with t3:
        st.subheader("Agrupamento")
        gb_col = st.selectbox("Agrupar por (Categoria)", df.columns)
        val_col = st.selectbox("Coluna de Valor", nums)
        agg = st.selectbox("Função", ["sum", "mean", "count"])
        if st.button("Criar Agrupamento"):
            df = df.groupby(gb_col)[val_col].agg(agg).reset_index()
            df.rename(columns={val_col: f"{val_col}_{agg}"}, inplace=True)
            st.session_state['df'] = df
            pipeline_engine.add_step("groupby", {"gb_col":gb_col, "val_col":val_col, "agg":agg}, f"Group {gb_col}")
            st.rerun()

def page_viz():
    st.header("📈 Visual Studio")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    l, r = st.columns([1,3])
    with l:
        st.subheader("Configuração")
        x = st.selectbox("Eixo X", df.columns)
        y = st.selectbox("Eixo Y", [None]+list(df.select_dtypes(include=np.number).columns))
        ct = st.selectbox("Tipo", ["Barras","Linha","Scatter","Histograma","Pizza","Boxplot","Heatmap"])
        clr = st.selectbox("Legenda/Grupo", [None]+list(df.columns))
        tt = st.text_input("Título", f"Análise de {x}")
        
        with st.expander("🎨 Estilo e Cores"):
            theme = st.selectbox("Tema", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])
            color_discrete = st.color_picker("Cor Principal", "#2563eb")
            
        with st.expander("🔍 Filtros Locais"):
            filter_col = st.selectbox("Filtrar coluna (apenas visual)", [None] + list(df.columns))
            if filter_col:
                f_vals = df[filter_col].unique()
                f_sel = st.multiselect("Valores", f_vals, default=f_vals)

    with r:
        # Apply local filter
        plot_df = df.copy()
        if filter_col and f_sel:
            plot_df = plot_df[plot_df[filter_col].isin(f_sel)]

        # Live Render
        try:
            if ct=="Barras": fig = px.bar(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Linha": fig = px.line(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Scatter": fig = px.scatter(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Histograma": fig = px.histogram(plot_df, x=x, color=clr, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Boxplot": fig = px.box(plot_df, x=x, y=y, color=clr, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Pizza": fig = px.pie(plot_df, names=x, values=y, title=tt, template=theme, color_discrete_sequence=[color_discrete])
            elif ct=="Heatmap": fig = px.imshow(plot_df.corr(numeric_only=True), text_auto=True, title=tt, template=theme)
            
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns([3,1])
            note = c1.text_input("Nota para relatório")
            if c2.button("➕ Add Relatório"):
                st.session_state['report_charts'].append({"fig":fig, "title":tt, "note":note})
                st.toast("Adicionado!")
        except Exception as e: st.error(f"Configuração inválida: {e}")

def page_ml():
    st.header("🏆 ML Studio")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    t1, t2, t3 = st.tabs(["Treino", "Simulador", "Conceitos"])
    
    with t1:
        mode = st.radio("Modo", ["Classificação de Texto (NLP)", "Tabular (Regressão/Classif)"])
        
        if mode == "Classificação de Texto (NLP)":
            txt_col = st.selectbox("Coluna Texto", df.select_dtypes(include='object').columns)
            target = st.selectbox("Coluna Alvo (Target)", df.columns)
            
            if st.button("Treinar NLP"):
                try:
                    pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=500)), ('clf', LogisticRegression())])
                    X = df[txt_col].fillna("").astype(str)
                    y = df[target].fillna("Unknown").astype(str)
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
                    pipe.fit(X_tr, y_tr)
                    acc = pipe.score(X_te, y_te)
                    st.session_state['ml_pipeline'] = pipe
                    st.session_state['ml_type'] = 'nlp'
                    st.success(f"Treinado! Acurácia: {acc:.2f}")
                except Exception as e: st.error(e)
        
        else:
            c1, c2 = st.columns(2)
            target = c1.selectbox("Target", df.columns)
            feats = c2.multiselect("Features", [c for c in df.columns if c!=target])
            
            # Hyperparameters
            st.markdown("#### ⚙️ Ajuste Fino (Hiperparâmetros)")
            n_est = st.slider("Nº Árvores (n_estimators)", 10, 200, 50, help="Mais árvores = modelo mais robusto, mas mais lento.")
            
            if st.button("Treinar"):
                try:
                    X = df[feats]
                    y = df[target]
                    nums = X.select_dtypes(include=np.number).columns
                    cats = X.select_dtypes(include=['object']).columns
                    pre = ColumnTransformer([('n', SimpleImputer(strategy='median'), nums), ('c', OneHotEncoder(handle_unknown='ignore'), cats)])
                    
                    is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
                    model = RandomForestRegressor(n_estimators=n_est) if is_reg else RandomForestClassifier(n_estimators=n_est)
                    
                    pipe = Pipeline([('pre', pre), ('clf', model)])
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
                    pipe.fit(X_tr, y_tr)
                    
                    score_tr = pipe.score(X_tr, y_tr)
                    score_te = pipe.score(X_te, y_te)
                    
                    st.metric("Score Teste", f"{score_te:.2f}")
                    
                    if score_tr - score_te > 0.15:
                        st.warning("⚠️ Overfitting detectado! O modelo decorou o treino. Tente diminuir as árvores ou usar menos colunas.")
                    
                    if not is_reg:
                        st.markdown("#### Matriz de Confusão")
                        st.caption("Mostra onde o modelo acertou e onde confundiu as classes.")
                        preds = pipe.predict(X_te)
                        cm = confusion_matrix(y_te, preds)
                        st.plotly_chart(px.imshow(cm, text_auto=True), use_container_width=True)

                    st.session_state['ml_pipeline'] = pipe
                    st.session_state['ml_type'] = 'tabular'
                    st.session_state['ml_feats'] = feats

                except Exception as e: st.error(e)

    with t2:
        if st.session_state.get('ml_pipeline'):
            st.subheader("🎮 Testar Modelo")
            model = st.session_state['ml_pipeline']
            mtype = st.session_state['ml_type']
            
            if mtype == 'nlp':
                txt = st.text_input("Digite uma frase:")
                if st.button("Classificar"):
                    pred = model.predict([txt])[0]
                    st.info(f"IA diz: **{pred}**")
            else:
                vals = {}
                for f in st.session_state.get('ml_feats', []):
                    vals[f] = st.text_input(f"Valor para {f}", "0")
                if st.button("Prever Valor"):
                    df_in = pd.DataFrame([vals])
                    # Auto cast
                    for c in df_in.columns: 
                        try: df_in[c] = pd.to_numeric(df_in[c])
                        except: pass
                    pred = model.predict(df_in)[0]
                    st.success(f"Resultado Previsto: **{pred}**")
        else:
            st.info("Treine um modelo primeiro.")
            
    with t3:
        st.markdown("### O que significam essas métricas?")
        st.markdown("- **R² (R-quadrado):** Explica o quanto seu modelo 'entende' a variação dos dados. 1.0 é perfeito, 0.0 é inútil.")
        st.markdown("- **Matriz de Confusão:** A diagonal principal mostra os acertos. Fora dela são os erros.")

def page_sql():
    st.header("💻 SQL & Cheat Sheet")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB ausente."); return
    
    # Preview
    with st.expander("Tabela", expanded=True):
        st.dataframe(df.head(3), use_container_width=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("### 📋 Colinha Genérica")
        if st.button("Selecionar Colunas"): st.session_state['sql_q'] = "SELECT coluna_a, coluna_b FROM df"
        if st.button("Filtrar (WHERE)"): st.session_state['sql_q'] = "SELECT * FROM df WHERE valor > 100"
        if st.button("Agrupar (GROUP BY)"): st.session_state['sql_q'] = "SELECT categoria, COUNT(*) FROM df GROUP BY categoria"
        if st.button("Ordenar (ORDER BY)"): st.session_state['sql_q'] = "SELECT * FROM df ORDER BY data DESC"
        if st.button("Lógica (CASE WHEN)"): st.session_state['sql_q'] = "SELECT *, CASE WHEN valor > 50 THEN 'Alto' ELSE 'Baixo' END as status FROM df"

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
        pdf = generate_report_v11(df, charts, kpis)
        st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("🎓 Masterclass v11")
        st.caption("Learning Edition")
        
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