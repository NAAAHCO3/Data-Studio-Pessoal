"""
Enterprise Analytics — Academy Edition (v9.0)
Author: Gemini Advanced
Version: 9.0 (Academy Module, Educational Tooltips, Full-Stack Demo Data)

Destaques v9.0:
- EDU: Módulo 'Academy' com teoria, glossário e snippets de código (Python & SQL).
- UX: 'Modo Professor' com explicações de código nas telas de operação.
- DATA: Gerador de dados demo aprimorado (inclui texto para NLP).
- CORE: Todas as funcionalidades da v8.0 (EventBus, SQL Sandbox, etc).
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
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Callable, Union

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
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix

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
# CORE: DESIGN SYSTEM & CONFIG
# ---------------------------
st.set_page_config(
    page_title="Data Studio Academy", 
    layout="wide", 
    page_icon="🎓", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Academy Card */
    .academy-card {
        background-color: #f0f9ff; border-left: 5px solid #0ea5e9; 
        padding: 20px; border-radius: 8px; margin-bottom: 20px;
    }
    
    /* Enterprise Cards */
    .st-card {
        background-color: #ffffff; padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12); border-top: 4px solid #2563eb; margin-bottom: 20px;
    }
    
    /* KPI Box */
    .kpi-box {
        text-align: center; padding: 15px; background: #f8fafc; 
        border-radius: 8px; border: 1px solid #e2e8f0;
    }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
    .kpi-lbl { font-size: 0.85rem; color: #64748b; text-transform: uppercase; }
    
    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .st-card { background-color: #1e293b; border-color: #334155; color: white; }
        .kpi-box { background-color: #0f172a; border-color: #334155; }
        .kpi-val { color: #f1f5f9; }
        .academy-card { background-color: #0c4a6e; border-color: #38bdf8; color: #e0f2fe; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# CORE: EVENT BUS v2
# ---------------------------
class EventBus:
    def __init__(self):
        self.logs = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EA_Core")

    def emit(self, namespace: str, event: str, payload: Dict[str, Any] = None, level: str = "INFO"):
        ts = datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "namespace": namespace,
            "event": event,
            "level": level,
            "payload": str(payload)
        }
        if 'audit_log' not in st.session_state: st.session_state['audit_log'] = []
        st.session_state['audit_log'].append(entry)
        
        msg = payload.get('msg', f"{namespace}:{event}")
        if level == "ERROR": st.error(f"[{namespace}] {msg}")
        elif level == "SUCCESS": st.toast(msg, icon="✅")
        elif level == "WARN": st.warning(msg)

if 'bus' not in st.session_state: st.session_state['bus'] = EventBus()
bus = st.session_state['bus']

# ---------------------------
# ENGINE: PIPELINE & LOGIC
# ---------------------------
class PipelineEngine:
    def get_steps(self): return st.session_state.get('etl_steps', [])
    
    def add_step(self, op: str, params: Dict[str, Any], description: str):
        step = {"op": op, "params": params, "desc": description}
        st.session_state['etl_steps'].append(step)
        bus.emit("etl", "step_added", {"msg": description}, "SUCCESS")

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
                elif op == "dummies": df = pd.get_dummies(df, columns=p['cols'], drop_first=True, dtype=int)
            st.session_state['df'] = df
        except Exception as e: bus.emit("etl", "replay_error", {"msg": str(e)}, "ERROR")
    
    def export_python(self) -> str:
        code = ["import pandas as pd", "df = pd.read_csv('data.csv')"]
        for step in self.get_steps():
            op = step['op']
            p = step['params']
            if op == "calc": code.append(f"df['{p['nm']}'] = df['{p['a']}'] {p['op']} df['{p['b']}']")
            elif op == "dropna": code.append("df = df.dropna()")
            elif op == "dummies": code.append(f"df = pd.get_dummies(df, columns={p['cols']}, drop_first=True)")
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
            if is_num_x:
                analysis["recommended"] = "Histograma"
                analysis["reason"] = "Distribuição de variável numérica única."
            else:
                analysis["recommended"] = "Barras"
                analysis["reason"] = "Contagem de frequências por categoria."
        else:
            if is_date_x and is_num_y:
                analysis["recommended"] = "Linha"
                analysis["reason"] = "Série temporal (evolução no tempo)."
            elif not is_num_x and is_num_y:
                analysis["recommended"] = "Barras"
                analysis["reason"] = "Comparação de valores entre categorias."
            elif is_num_x and is_num_y:
                analysis["recommended"] = "Scatter"
                analysis["reason"] = "Correlação entre duas variáveis numéricas."
        return analysis

class SQLSanitizer:
    @staticmethod
    def validate(query: str) -> Tuple[bool, str]:
        forbidden = ["DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE", "GRANT", "INSERT"]
        if any(cmd in query.upper() for cmd in forbidden): return False, "Comando proibido (Modo Leitura)."
        return True, "OK"

class ModelRegistry:
    def register(self, model, name: str, metrics: dict, features: List[str]):
        entry = {"name": name, "model": model, "metrics": metrics, "features": features, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
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
        'report_charts': [], 'model_registry': [], 'last_file_uid': None
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
def get_demo_data():
    """Gera dados ricos para teste de todas as abas."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2024-01-01', periods=n)
    cats = ['Eletrônicos', 'Casa', 'Vestuário', 'Livros']
    reviews = [
        "Ótimo produto, recomendo muito", "Chegou atrasado mas a qualidade é boa", 
        "Péssimo, quebrou no primeiro dia", "Atendimento excelente", 
        "Não gostei da cor", "Maravilhoso, compraria novamente",
        "Muito caro pelo que oferece", "Custo benefício incrível"
    ]
    
    df = pd.DataFrame({
        'data_venda': dates,
        'categoria': np.random.choice(cats, n),
        'custo': np.random.uniform(50, 300, n),
        'margem_lucro': np.random.uniform(0.1, 0.5, n),
        'nota_cliente': np.random.randint(1, 6, n),
        'comentario_cliente': np.random.choice(reviews, n)
    })
    # Criar correlações
    df['preco_venda'] = df['custo'] * (1 + df['margem_lucro'])
    df['vendas_total'] = df['preco_venda'] * np.random.randint(1, 5, n)
    return df

# ---------------------------
# PDF ENGINE
# ---------------------------
class EnterprisePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Academy Analytics Report v9.0', 0, 1, 'R')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pag {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report_v9(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
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
    
    # Charts
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Gráficos", 0, 1)
    import tempfile, os
    for i, ch in enumerate(charts):
        if i % 2 == 0 and i > 0: pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, ch['title'], 0, 1)
        try:
            img = ch['fig'].to_image(format="png", scale=1, engine="kaleido")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img)
                path = tmp.name
            pdf.image(path, x=20, w=170)
            os.unlink(path)
        except: pdf.cell(0, 10, "[Imagem indisponível]", 0, 1)
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------
# PAGES: ACADEMY (NOVO)
# ---------------------------
def page_academy():
    st.title("🎓 Data Academy - Aprenda Fazendo")
    st.markdown("Bem-vindo à escola de dados! Aqui você entende o conceito por trás das ferramentas.")
    
    tab1, tab2, tab3 = st.tabs(["📚 Glossário Essencial", "🧪 Guia de Ferramentas", "💻 Guia de Código (Python & SQL)"])
    
    with tab1:
        st.header("Dicionário do Cientista de Dados")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### Conceitos de Dados
            - **DataFrame:** É como chamamos uma tabela de dados no Python (linhas e colunas).
            - **Null / NaN:** "Not a Number". Representa dados faltando ou vazios.
            - **Outlier:** Um valor muito fora do padrão (ex: alguém com 200 anos de idade).
            - **Cardinalidade:** Quantos valores únicos existem em uma coluna. Ex: "Sexo" tem cardinalidade baixa (M/F), "Nome" tem alta.
            """)
        with cols[1]:
            st.markdown("""
            ### Conceitos de ML
            - **Target (Alvo):** A coluna que você quer prever (ex: Preço, Vendeu/Não Vendeu).
            - **Features (Variáveis):** As colunas usadas para explicar o alvo (ex: Tamanho, Cor).
            - **Regressão:** Prever um número contínuo (ex: R$ 500,00).
            - **Classificação:** Prever uma categoria (ex: Sim/Não, A/B/C).
            - **Overfitting:** Quando o modelo decora os dados de treino e não sabe generalizar para novos dados.
            """)

    with tab2:
        st.header("Quando usar cada ferramenta?")
        
        st.info("🛠️ **ETL Studio (Data Studio)**")
        st.markdown("Use quando seus dados estiverem 'sujos'. Ex: Datas como texto, células vazias, nomes de colunas ruins.")
        
        st.success("📈 **Visual Studio**")
        st.markdown("Use para explorar padrões. O sistema 'Smart Viz' vai sugerir o melhor gráfico, mas lembre-se:\n"
                    "- Comparar categorias? **Barra**.\n"
                    "- Ver evolução no tempo? **Linha**.\n"
                    "- Ver correlação entre números? **Scatter**.")
        
        st.warning("🏆 **AutoML**")
        st.markdown("Use quando quiser criar previsões. O sistema testa vários algoritmos para você. Lembre-se: **Garanta que seus dados não tenham nulos antes de treinar!**")

    with tab3:
        st.header("Como o código funciona?")
        st.markdown("Abaixo estão os snippets reais de Python e SQL para as tarefas mais comuns.")
        
        c_py, c_sql = st.columns(2)
        
        with c_py:
            st.subheader("🐍 Python (Pandas)")
            st.caption("Usado para manipular dados na memória (como este app faz).")
            
            st.markdown("**1. Filtrar Dados**")
            st.code("df_filtrado = df[df['vendas'] > 100]", language="python")
            
            st.markdown("**2. Criar Coluna Calculada**")
            st.code("df['margem'] = df['receita'] - df['custo']", language="python")
            
            st.markdown("**3. Agrupar (Pivot)**")
            st.code("df.groupby('categoria')['vendas'].sum()", language="python")

            st.markdown("**4. Juntar Tabelas (Merge)**")
            st.code("pd.merge(tabela_a, tabela_b, on='id')", language="python")
            
            st.markdown("**5. Remover Duplicatas**")
            st.code("df.drop_duplicates()", language="python")

        with c_sql:
            st.subheader("🗄️ SQL (Banco de Dados)")
            st.caption("Usado para buscar dados direto da fonte (Banco de Dados).")
            
            st.markdown("**1. Filtrar Dados (WHERE)**")
            st.code("SELECT * FROM vendas \nWHERE valor > 100;", language="sql")
            
            st.markdown("**2. Criar Coluna (SELECT)**")
            st.code("SELECT receita - custo AS margem \nFROM financeiro;", language="sql")
            
            st.markdown("**3. Agrupar (GROUP BY)**")
            st.code("SELECT categoria, SUM(vendas) \nFROM transacoes \nGROUP BY categoria;", language="sql")

            st.markdown("**4. Juntar Tabelas (JOIN)**")
            st.code("SELECT * \nFROM tabela_a \nJOIN tabela_b ON tabela_a.id = tabela_b.id;", language="sql")

            st.markdown("**5. Remover Duplicatas (DISTINCT)**")
            st.code("SELECT DISTINCT * FROM tabela;", language="sql")

        st.markdown("---")
        st.info("💡 **Dica Pro:** O Python (Pandas) é ótimo para análises exploratórias e modelagem. O SQL é imbatível para extrair e agregar dados brutos de bancos gigantescos antes de trazer para o Python.")

# ---------------------------
# PAGES: OPERATIONAL
# ---------------------------
def page_home():
    st.title("🏠 Home")
    df = st.session_state['df']
    
    if df.empty:
        st.markdown("### Comece aqui 👇")
        c1, c2 = st.columns(2)
        with c1:
            st.info("Para aprender e testar tudo, carregue os dados de demonstração.")
            if st.button("🚀 Carregar Dados Demo (Full Stack)", type="primary"):
                df = get_demo_data()
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.rerun()
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[0]:,}</div><div class='kpi-lbl'>Linhas</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.shape[1]}</div><div class='kpi-lbl'>Colunas</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.duplicated().sum()}</div><div class='kpi-lbl'>Duplicatas</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-box'><div class='kpi-val'>{df.isna().mean().mean()*100:.1f}%</div><div class='kpi-lbl'>Nulos</div></div>", unsafe_allow_html=True)

    st.markdown("### 🔎 Visão Geral Automática")
    with st.expander("Ver Tabela de Dados", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

def page_etl():
    st.header("🛠️ ETL Studio")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    c1, c2 = st.columns([1, 4])
    if c1.button("↩️ Desfazer"): pipeline_engine.undo(); st.rerun()
    
    t1, t2 = st.tabs(["Transformar", "Limpar"])
    
    with t1:
        with st.expander("🧠 Dica de Python: Cálculos", expanded=True):
            st.caption("Em Pandas, operações são vetorizadas. `df['C'] = df['A'] + df['B']` soma a coluna inteira de uma vez!")
            
        with st.form("calc"):
            nums = df.select_dtypes(include=np.number).columns
            if len(nums)>0:
                c_a, c_op, c_b = st.columns([2,1,2])
                a = c_a.selectbox("A", nums)
                op = c_op.selectbox("Op", ["+","-","*","/"])
                b = c_b.selectbox("B", nums)
                nm = st.text_input("Nome Coluna", "res")
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
        with st.expander("🧠 Dica de Python: Limpeza", expanded=True):
            st.caption("`dropna()` remove linhas com qualquer valor vazio. `get_dummies()` converte categorias em números binários.")
            
        if st.button("Remover Nulos (Drop NA)"):
            df = df.dropna()
            st.session_state['df'] = df
            pipeline_engine.add_step("dropna", {}, "Drop NA")
            st.rerun()
        
        cats = df.select_dtypes(include=['object']).columns
        if len(cats)>0:
            cols = st.multiselect("Gerar Dummies (One-Hot)", cats)
            if st.button("Aplicar") and cols:
                df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
                st.session_state['df'] = df
                pipeline_engine.add_step("dummies", {"cols":cols}, "Dummies")
                st.rerun()

def page_viz():
    st.header("📈 Visual Studio")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    with st.expander("🧠 Dica de Python: Plotly", expanded=True):
        st.caption("Usamos a biblioteca `plotly.express` (px). Ela cria gráficos interativos que permitem zoom e tooltips automaticamente.")

    with st.form("viz"):
        l, r = st.columns([1,3])
        with l:
            x = st.selectbox("Eixo X", df.columns)
            y = st.selectbox("Eixo Y", [None]+list(df.select_dtypes(include=np.number).columns))
            
            smart = SmartViz.analyze(df, x, y)
            st.info(f"💡 Sugestão: **{smart['recommended']}**\n\n_{smart['reason']}_")
            
            ct = st.selectbox("Tipo", ["Barras","Linha","Scatter","Histograma","Pizza","Boxplot","Heatmap"], index=["Barras","Linha","Scatter","Histograma","Pizza","Boxplot","Heatmap"].index(smart['recommended']))
            clr = st.selectbox("Cor", [None]+list(df.columns))
            tt = st.text_input("Título", f"Análise de {x}")
            sub = st.form_submit_button("Gerar")

    if sub:
        try:
            if ct=="Barras": fig = px.bar(df, x=x, y=y, color=clr, title=tt)
            elif ct=="Linha": fig = px.line(df, x=x, y=y, color=clr, title=tt)
            elif ct=="Scatter": fig = px.scatter(df, x=x, y=y, color=clr, title=tt)
            elif ct=="Histograma": fig = px.histogram(df, x=x, color=clr, title=tt)
            elif ct=="Boxplot": fig = px.box(df, x=x, y=y, color=clr, title=tt)
            elif ct=="Pizza": fig = px.pie(df, names=x, values=y, title=tt)
            elif ct=="Heatmap": fig = px.imshow(df.corr(numeric_only=True), text_auto=True, title=tt)
            
            st.session_state['last_fig'] = fig
            st.session_state['last_meta'] = {'title':tt, 'type':ct}
        except Exception as e: st.error(e)

    if st.session_state.get('last_fig'):
        st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
        if st.button("Add Relatório"):
            st.session_state['report_charts'].append({"fig":st.session_state['last_fig'], "title":st.session_state['last_meta']['title'], "note":""})
            st.success("Adicionado!")

def page_ml():
    st.header("🏆 AutoML Pro")
    df = st.session_state['df'].copy()
    if df.empty: st.warning("Sem dados."); return

    with st.expander("🧠 Dica de Python: Scikit-Learn", expanded=True):
        st.caption("Usamos `Pipeline` para garantir que o tratamento de dados (imputer, scaler) seja aplicado corretamente tanto no treino quanto nos testes.")

    t1, t2, t3 = st.tabs(["AutoML (Tabular)", "NLP (Texto)", "Predições"])
    
    # Tabular ML
    with t1:
        c1, c2 = st.columns(2)
        target = c1.selectbox("Target (Alvo)", df.columns)
        feats = c2.multiselect("Features", [c for c in df.columns if c!=target])
        
        if st.button("🚀 Rodar AutoML"):
            if not feats: st.error("Features?"); return
            
            with st.spinner("Treinando e validando..."):
                try:
                    X = df[feats]
                    y = df[target]
                    
                    # Preprocessing
                    nums = X.select_dtypes(include=np.number).columns
                    cats = X.select_dtypes(include=['object']).columns
                    pre = ColumnTransformer([
                        ('num', SimpleImputer(strategy='median'), nums),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), cats)
                    ])

                    is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
                    results = []

                    if is_reg:
                        y = y.fillna(y.mean())
                        models = {
                            "Linear Reg": LinearRegression(),
                            "Random Forest": RandomForestRegressor(n_estimators=30)
                        }
                        metric = "R2 Mean"
                    else:
                        y = y.fillna(y.mode()[0]).astype(str)
                        models = {
                            "Logistic Reg": LogisticRegression(),
                            "Random Forest": RandomForestClassifier(n_estimators=30)
                        }
                        metric = "Accuracy Mean"

                    best_score = -999
                    best_model = None
                    best_name = ""

                    for name, model in models.items():
                        pipe = Pipeline([('pre', pre), ('clf', model)])
                        cv_scores = cross_val_score(pipe, X, y, cv=3, scoring='r2' if is_reg else 'accuracy')
                        mean_score = cv_scores.mean()
                        results.append({"Modelo": name, metric: mean_score})

                        if mean_score > best_score:
                            best_score = mean_score
                            best_model_name = name
                            pipe.fit(X, y) # Retrain full
                            best_model = pipe

                    st.success(f"Vencedor: {best_model_name}")
                    st.dataframe(pd.DataFrame(results))
                    model_registry.register(best_model, f"{best_model_name}_{target}", {metric: best_score}, feats)
                    
                except Exception as e: st.error(e)

    # NLP
    with t2:
        st.subheader("Classificação de Texto")
        txt_cols = df.select_dtypes(include='object').columns
        if len(txt_cols) > 0:
            tc = st.selectbox("Coluna Texto", txt_cols)
            tg = st.selectbox("Target (Categoria)", df.columns)
            if st.button("Treinar Modelo de Texto"):
                try:
                    pipe = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=1000)),
                        ('clf', LogisticRegression())
                    ])
                    X = df[tc].astype(str)
                    y = df[tg].astype(str)
                    pipe.fit(X, y)
                    acc = pipe.score(X, y)
                    st.success(f"Treinado! Acurácia no treino: {acc:.2f}")
                    model_registry.register(pipe, f"NLP_{tc}_to_{tg}", {"Acc": acc}, [tc])
                except Exception as e: st.error(e)
        else: st.info("Sem colunas de texto.")

    # Predictions
    with t3:
        models = model_registry.get_models()
        if not models: st.info("Treine um modelo primeiro."); return
        
        m_sel = st.selectbox("Escolha Modelo", [m['name'] for m in models])
        selected = next(m for m in models if m['name'] == m_sel)
        
        st.markdown("#### Testar Modelo")
        input_data = {}
        
        # Handle inputs differently for NLP vs Tabular
        if "NLP" in selected['name']:
            txt_in = st.text_input(f"Texto para {selected['features'][0]}")
            if st.button("Classificar Texto"):
                pred = selected['model'].predict([txt_in])[0]
                st.metric("Previsão", pred)
        else:
            for f in selected['features']:
                input_data[f] = st.text_input(f"Valor para {f}", "0")
            if st.button("Prever Valor"):
                try:
                    df_in = pd.DataFrame([input_data])
                    # auto-cast basic
                    for c in df_in.columns:
                        try: df_in[c] = pd.to_numeric(df_in[c])
                        except: pass
                    pred = selected['model'].predict(df_in)[0]
                    st.metric("Resultado", str(pred))
                except Exception as e: st.error(e)

def page_sql():
    st.header("💻 SQL Engine")
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    if not _HAS_DUCKDB: st.error("DuckDB ausente."); return

    with st.expander("🧠 Dica de Python: SQL no Pandas", expanded=True):
        st.caption("O DuckDB permite rodar SQL diretamente em DataFrames Pandas como se fossem tabelas de banco de dados. É super rápido!")

    q = st.text_area("Query (use 'df')", "SELECT * FROM df LIMIT 10")
    if st.button("Executar"):
        valid, msg = SQLSanitizer.validate(q)
        if not valid: st.error(msg)
        else:
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
        try:
            kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
            pdf = generate_report_v9(df, charts, kpis)
            st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")
        except Exception as e: st.error(e)

# ---------------------------
# MAIN
# ---------------------------
def main():
    with st.sidebar:
        st.title("🎓 Academy v9")
        st.caption("App + Escola de Dados")
        
        uploaded = st.file_uploader("Arquivo", type=['csv','xlsx'])
        if uploaded:
            uid = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get('last_uid') != uid:
                if uploaded.size > 200*1024*1024: st.error("Arquivo muito grande.")
                else:
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
        menu = st.radio("Menu", ["🏠 Home", "🎓 Academy", "🛠️ ETL", "💻 SQL", "📈 Viz", "🏆 ML", "📑 Report"])
        st.markdown("---")
        if st.button("Hard Reset"):
            st.session_state.clear()
            st.rerun()

    if menu == "🏠 Home": page_home()
    elif menu == "🎓 Academy": page_academy()
    elif menu == "🛠️ ETL": page_etl()
    elif menu == "💻 SQL": page_sql()
    elif menu == "📈 Viz": page_viz()
    elif menu == "🏆 ML": page_ml()
    elif menu == "📑 Report": page_report()

if __name__ == "__main__":
    main()