# app.py
"""
Enterprise Analytics — BI Edition (No-Code) — MAX-PRO
Single-file Streamlit app, "C" tier (Enterprise).
Requirements (approx):
- streamlit==1.51.0
- pandas==2.3.3
- numpy==2.3.4
- plotly==6.4.0
- fpdf2==2.8.5
- openpyxl
- pyarrow==21.0.0
- joblib
- duckdb==1.0.0 (optional but recommended)
- kaleido (optional, for Plotly -> image export)
- scikit-learn, scipy
"""

import streamlit as st
st.set_page_config(page_title="Enterprise Analytics — BI MAX-PRO", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import base64
import logging
import math
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Optional heavy libs
try:
    import duckdb
    _HAS_DUCKDB = True
except Exception:
    import sqlite3
    _HAS_DUCKDB = False

try:
    import joblib
except Exception:
    joblib = None

# ML & stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Robustness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enterprise-bi")

# -----------------------
# Helpers: IO & Cleaning
# -----------------------
def try_read_csv(file_obj, encodings=("utf-8", "latin1", "cp1252", "iso-8859-1")):
    last_exc = None
    for enc in encodings:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception as e:
            last_exc = e
            continue
    # final fallback
    file_obj.seek(0)
    return pd.read_csv(file_obj, engine="python", encoding_errors="ignore")

def safe_read(file):
    """
    Accepts Streamlit UploadedFile or local path str.
    Returns pandas.DataFrame or raises.
    """
    if file is None:
        raise ValueError("No file provided.")
    try:
        if hasattr(file, "read"):
            name = getattr(file, "name", "")
            if name.lower().endswith(".csv"):
                try:
                    df = try_read_csv(file)
                except Exception as e:
                    logger.warning("CSV read fallback, attempting engine=python")
                    file.seek(0)
                    df = pd.read_csv(file, engine="python", encoding_errors="ignore")
            else:
                # Excel
                file.seek(0)
                df = pd.read_excel(file, engine="openpyxl")
        else:
            path = str(file)
            if path.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_csv(path, encoding="latin1", engine="python")
            else:
                df = pd.read_excel(path, engine="openpyxl")
        # basic cleanup of column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        )
        return df
    except Exception as e:
        logger.exception("safe_read failed")
        raise

def format_number(n):
    try:
        n = int(n)
    except Exception:
        try:
            n = float(n)
        except Exception:
            return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n}"

def safe_display_dataframe(df: pd.DataFrame, height: int = 400):
    """
    When PyArrow conversion fails, Streamlit raises. Fallback to string-coerced df.
    """
    try:
        st.dataframe(df, width="stretch", height=height)
    except Exception as e:
        logger.warning("st.dataframe Arrow conversion failed; coercing problematic columns to str.")
        # find columns with mixed types that could cause conversion errors
        df2 = df.copy()
        for c in df2.columns:
            # if dtype is integer but some string values present -> cast to string
            if df2[c].dtype.kind in ("i","u") and df2[c].apply(lambda x: isinstance(x, str)).any():
                df2[c] = df2[c].astype(str)
            # pyarrow sometimes fails on object columns with mixed types; cast those to str as well
            if df2[c].dtype == object:
                # but keep memory use in mind; only cast if necessary
                if df2[c].apply(lambda x: isinstance(x, (list, dict))).any():
                    df2[c] = df2[c].astype(str)
        st.dataframe(df2.astype(str), width="stretch", height=height)

# -----------------------
# Data Quality Tools
# -----------------------
def missing_heatmap(df: pd.DataFrame, max_cols=60):
    m = df.isna().astype(int)
    if m.shape[1] > max_cols:
        m = m.iloc[:, :max_cols]
    fig = px.imshow(m.T, aspect='auto', color_continuous_scale=['#e0e0e0','#ff6b6b'],
                    labels={'x':'Index', 'y':'Columns'}, title='Mapa de Missing (1 = missing)')
    fig.update_layout(margin=dict(l=60,r=10,t=40,b=30))
    return fig

def detect_outliers_iqr(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if s.empty:
        return pd.DataFrame(columns=df.columns)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df[(df[col] < low) | (df[col] > high)]

def zscore_anomaly(df: pd.DataFrame, col: str, thresh=3.0):
    s = df[col].dropna()
    zs = np.abs(stats.zscore(s))
    return df.loc[s.index[zs > thresh]]

# -----------------------
# ETL & Data Studio utils
# -----------------------
def create_arithmetic(df: pd.DataFrame, new_col: str, a: str, b: Optional[str], op: str, b_value: Optional[float] = None):
    df = df.copy()
    try:
        if b is not None:
            if op == '+': df[new_col] = df[a] + df[b]
            elif op == '-': df[new_col] = df[a] - df[b]
            elif op == '*': df[new_col] = df[a] * df[b]
            elif op == '/': df[new_col] = df[a] / df[b].replace(0, np.nan)
        else:
            if op == '+': df[new_col] = df[a] + b_value
            elif op == '-': df[new_col] = df[a] - b_value
            elif op == '*': df[new_col] = df[a] * b_value
            elif op == '/': df[new_col] = df[a] / b_value if b_value != 0 else np.nan
    except Exception:
        raise
    return df

def create_conditional(df: pd.DataFrame, new_col: str, col: str, op: str, thr: float, true_label: str, false_label: str):
    df = df.copy()
    ops = {
        ">": df[col] > thr,
        "<": df[col] < thr,
        ">=": df[col] >= thr,
        "<=": df[col] <= thr,
        "==": df[col] == thr,
        "!=": df[col] != thr
    }
    mask = ops.get(op, df[col] > thr)
    df[new_col] = np.where(mask, true_label, false_label)
    return df

def extract_date_part(df: pd.DataFrame, date_col: str, component: str, new_name: str):
    df = df.copy()
    if component == 'year': df[new_name] = df[date_col].dt.year
    elif component == 'month': df[new_name] = df[date_col].dt.month
    elif component == 'day': df[new_name] = df[date_col].dt.day
    elif component == 'weekday': df[new_name] = df[date_col].dt.day_name()
    elif component == 'quarter': df[new_name] = df[date_col].dt.quarter
    return df

def split_text(df: pd.DataFrame, col: str, sep: str, idx: int, new_col: str):
    df = df.copy()
    s = df[col].astype(str).str.split(sep, expand=True)
    if idx < s.shape[1]:
        df[new_col] = s[idx]
    else:
        raise IndexError("Index out of range for split.")
    return df

def pivot_transform(df: pd.DataFrame, index: List[str], columns: str, values: str, agg='sum'):
    df = df.copy()
    aggmap = {'sum':'sum', 'mean':'mean', 'count':'count', 'min':'min', 'max':'max'}
    if agg not in aggmap: agg = 'sum'
    res = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggmap[agg]).reset_index()
    res.columns = [("_".join(map(str,c)) if isinstance(c, tuple) else str(c)).strip() for c in res.columns]
    return res

def unpivot_transform(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name='variable', value_name='value'):
    return df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: List[str], how='left'):
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)

# -----------------------
# Visual Studio: Build charts
# -----------------------
def build_chart(chart_type: str, df: pd.DataFrame, x: Optional[str]=None, y: Optional[str]=None, color: Optional[str]=None,
                agg: Optional[str]=None, theme: str='plotly', show_labels: bool=True, height: int=450, size: Optional[str]=None):
    plot_df = df.copy()
    if agg and x and y and chart_type in ("Barras","Linha","Pizza","Treemap"):
        if color:
            plot_df = plot_df.groupby([x, color], dropna=False)[y].agg(agg).reset_index()
        else:
            plot_df = plot_df.groupby(x, dropna=False)[y].agg(agg).reset_index()

    if chart_type == "Barras":
        fig = px.bar(plot_df, x=x, y=y, color=color, text_auto=show_labels, template=theme)
    elif chart_type == "Linha":
        fig = px.line(plot_df, x=x, y=y, color=color, markers=True, template=theme)
    elif chart_type == "Dispersão" or chart_type == "Scatter":
        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, template=theme)
    elif chart_type == "Pizza":
        fig = px.pie(plot_df, names=x, values=y, template=theme)
    elif chart_type == "Histograma":
        fig = px.histogram(plot_df, x=x, color=color, nbins=30, template=theme, text_auto=show_labels)
    elif chart_type == "Box":
        fig = px.box(plot_df, x=x, y=y, color=color, template=theme)
    elif chart_type == "Heatmap":
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True, template=theme, title="Matriz de Correlação")
    elif chart_type == "Treemap":
        fig = px.treemap(plot_df, path=[x, color] if color else [x], values=y, template=theme)
    elif chart_type == "Sunburst":
        fig = px.sunburst(plot_df, path=[x, color] if color else [x], values=y, template=theme)
    else:
        fig = go.Figure()
    fig.update_layout(height=height, title=f"{chart_type}: {y or ''} vs {x or ''}")
    return fig

# -----------------------
# Export utilities (Plotly images + PDF)
# -----------------------
def fig_to_image_bytes(fig: go.Figure, fmt='png', scale=2) -> Optional[bytes]:
    """
    Try to export Plotly fig to image bytes (requires kaleido).
    Returns bytes or None on failure.
    """
    try:
        img_bytes = fig.to_image(format=fmt, scale=scale)
        return img_bytes
    except Exception as e:
        logger.warning(f"fig_to_image_bytes failed: {e}")
        return None

def generate_pdf_with_charts(df: pd.DataFrame, charts: List[Dict[str,Any]], kpis: Dict[str,Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Executivo — Enterprise BI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # KPIs
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,40,190,28,'F')
    pdf.set_y(44)
    colw = 190/4
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    vals = [kpis.get('rows',''), kpis.get('cols',''), kpis.get('nulls',''), kpis.get('dups','')]
    for i,t in enumerate(titles):
        pdf.set_font("Helvetica","B",11)
        pdf.cell(colw,8,t, align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica","",11)
    for i,v in enumerate(vals):
        pdf.cell(colw,8,str(v), align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(12)

    # Short stats
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Resumo Estatistico (Top variaveis numericas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    desc = df.describe().T.reset_index().head(8)
    if not desc.empty:
        cols = ['index','mean','min','max']
        if set(cols).issubset(desc.columns):
            desc = desc[cols]
            desc.columns = ['Variavel','Media','Min','Max']
            w=[70,40,40,40]
            pdf.set_font("Helvetica","B",10)
            for i,c in enumerate(desc.columns):
                pdf.cell(w[i],8,c,1,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
            pdf.set_font("Helvetica","",9)
            for _,row in desc.iterrows():
                for i,val in enumerate(row):
                    txt = str(val)[:28]
                    pdf.cell(w[i],7,txt,1,align='C' if i>0 else 'L', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)

    # Charts: attempt to embed images
    for ch in charts:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0,8, ch.get('title','(sem titulo)'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        img = None
        try:
            img = fig_to_image_bytes(ch['fig'], fmt='png', scale=2)
        except Exception:
            img = None
        if img:
            # write to temporary buffer and embed
            tmp_name = f"/tmp/plot_{abs(hash(ch.get('title','')))}.png"
            try:
                with open(tmp_name, "wb") as f:
                    f.write(img)
                pdf.image(tmp_name, x=15, y=None, w=180)
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
            except Exception:
                pdf.multi_cell(0,6, " (Não foi possível gerar a imagem do gráfico) ")
        else:
            # fallback: list dataset summary for that chart
            pdf.multi_cell(0,6, f"Tipo: {ch.get('type','')}. Nota: {ch.get('note','')}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# -----------------------
# NLP: lightweight sentiment (lexicon-based)
# -----------------------
_POS_WORDS = {"good","great","excelent","excellent","bom","otimo","ótimo","agradavel","satisfeito","positivo","love","adorei","recomendo"}
_NEG_WORDS = {"bad","terrible","horrible","ruim","péssimo","pessimo","odioso","odio","frustrado","frustrante","pior","hate","detestei","não gostei","nao gostei","dinheiro jogado fora","devolvi"}
def simple_sentiment(text: str) -> Tuple[str,float]:
    """
    Very simple lexicon sentiment that supports Portuguese + English words.
    Returns label and score (pos-neg normalized).
    """
    if not isinstance(text, str) or text.strip()=="":
        return ("neutral", 0.0)
    txt = text.lower()
    pos = sum(txt.count(w) for w in _POS_WORDS)
    neg = sum(txt.count(w) for w in _NEG_WORDS)
    score = (pos - neg) / (max(1, pos+neg))
    if score > 0.1:
        return ("positive", score)
    if score < -0.1:
        return ("negative", score)
    return ("neutral", score)

# -----------------------
# SQL Lab (duckdb preferred)
# -----------------------
def run_sql(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        if _HAS_DUCKDB:
            con = duckdb.connect(database=':memory:')
            con.register('dados', df)
            res = con.execute(query).df()
            con.close()
            return res, None
        else:
            conn = sqlite3.connect(':memory:')
            df.to_sql('dados', conn, index=False, if_exists='replace')
            res = pd.read_sql_query(query, conn)
            conn.close()
            return res, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# -----------------------
# UI: Styles & session init
# -----------------------
st.markdown("""
<style>
.block-container {padding-top:1.2rem;padding-bottom:2rem}
.stButton>button {width:100%}
.metric-card {background:#f3f6fb;border-left:6px solid #4F8BF9;padding:12px;border-radius:6px;margin-bottom:8px}
@media (prefers-color-scheme: dark){
.metric-card{background:#262730;color:#ddd;border-left:6px solid #ffbd45}
}
</style>
""", unsafe_allow_html=True)

# session state keys
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
if 'df_main' not in st.session_state:
    st.session_state['df_main'] = pd.DataFrame()
if 'report_charts' not in st.session_state:
    st.session_state['report_charts'] = []  # list of dicts {fig,title,type,note}
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}  # saved dashboards
if 'last_fig' not in st.session_state:
    st.session_state['last_fig'] = None
if 'last_meta' not in st.session_state:
    st.session_state['last_meta'] = {}

# -----------------------
# Main UI
# -----------------------
def sidebar_area():
    st.sidebar.title("Enterprise Analytics — BI MAX-PRO")
    uploaded = st.sidebar.file_uploader("Arraste CSV ou Excel (XLSX)", type=['csv','xlsx'], help="Use CSV para arquivos grandes.")
    use_local = st.sidebar.checkbox("✔️ Usar arquivo de dev local (DEV)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.header("SQL Lab")
    st.sidebar.info("Você pode executar SQL sobre a tabela 'dados' (duckdb se disponível).")
    if st.sidebar.button("🔄 Resetar Sessão (limpar tudo)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
    return uploaded, use_local

def main():
    uploaded, use_local = sidebar_area()
    st.title("Enterprise Analytics — BI MAX-PRO (No-Code / Full)")
    st.caption("C: versão MAX-PRO — completo: EDA, ETL no-code, Visual Builder, Dashboard Builder, NLP, Anomalia, SQL")

    # load logic
    if uploaded is None and use_local:
        # try to load a sample path if exists
        local_path = os.environ.get("DEV_LOCAL_PATH", "/mnt/data/uploaded_dataset.csv")
        try:
            df = safe_read(local_path)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success("Dados carregados do caminho local.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar arquivo local: {e}")
    elif uploaded is not None:
        try:
            df = safe_read(uploaded)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success(f"Arquivo '{getattr(uploaded,'name','uploaded')}' carregado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler arquivo: {e}")

    if st.session_state['df_main'].empty:
        st.info("""
        Sem dados carregados — arraste um CSV/Excel na barra lateral.
        
        **Fluxo recomendado:**
        1. Data Studio -> preparar / criar colunas
        2. Visual Studio -> criar gráficos e adicionar ao relatório
        3. Relatório/Dashboard -> montar e exportar
        4. SQL Lab -> queries ad-hoc
        """)
        return

    df = st.session_state['df_main']

    # top menu
    menu = st.radio("", ["Data Quality", "Data Studio", "Visual Studio", "Relatório/Dashboard", "NLP & Texto", "Anomalias & Clustering", "SQL Lab", "Exportar"], horizontal=True)

    # ----- Data Quality -----
    if menu == "Data Quality":
        st.header("🔍 Data Quality & EDA")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Linhas</h3><h2>{format_number(df.shape[0])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Colunas</h3><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{format_number(int(df.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{format_number(int(df.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Missing Heatmap")
        st.plotly_chart(missing_heatmap(df), width="stretch")

        st.markdown("---")
        st.subheader("Perfil das Colunas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("🔢 Numéricas")
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.dataframe(numeric.describe().T[['mean','std','min','max']], width="stretch")
                col_sel = st.selectbox("Detectar outliers (IQR) em:", numeric.columns.tolist(), key="dq_outlier_col")
                out_df = detect_outliers_iqr(df, col_sel)
                st.write(f"Outliers IQR: {len(out_df)}")
                if st.checkbox("Mostrar amostra de outliers", key="dq_outlier_show"):
                    safe_display_dataframe(out_df.head(200))
            else:
                st.info("Sem colunas numéricas detectadas.")
        with col2:
            st.caption("🔤 Categóricas / Texto")
            cat = df.select_dtypes(include=['object','category'])
            if not cat.empty:
                df_cat_stat = pd.DataFrame({
                    "unique": cat.nunique(),
                    "missing": cat.isna().sum(),
                    "% missing": (cat.isna().mean()*100).round(2)
                })
                st.dataframe(df_cat_stat, width="stretch")
            else:
                st.info("Sem colunas de texto.")
        with col3:
            st.caption("📅 Datas")
            date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if date_cols:
                for dc in date_cols:
                    st.write(f"**{dc}**: {df[dc].min()} → {df[dc].max()}")
            else:
                st.info("Sem colunas de data detectadas automaticamente. Use Data Studio para converter colunas para data.")

        st.markdown("---")
        st.subheader("Visualizar tabela (segurança contra Arrow errors)")
        safe_display_dataframe(df, height=400)

    # ----- Data Studio (ETL) -----
    elif menu == "Data Studio":
        st.header("🛠 Data Studio — ETL No-Code")
        tabs = st.tabs(["Criar Coluna", "Renomear/Converter", "Pivot/Merge", "Filtros & Limpeza"])
        # Create col
        with tabs[0]:
            st.subheader("➕ Criar Coluna")
            op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
            if op == "Aritmética":
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not num_cols:
                    st.info("Não há colunas numéricas para operações.")
                else:
                    col_a = st.selectbox("Coluna A", num_cols, key="ds_calc_a")
                    mode = st.radio("Coluna B", ["Outra Coluna", "Valor Fixo"], key="ds_calc_mode")
                    col_b = None
                    val_b = None
                    if mode == "Outra Coluna":
                        col_b = st.selectbox("Coluna B", num_cols, key="ds_calc_b")
                    else:
                        val_b = st.number_input("Valor Fixo", value=1.0, key="ds_calc_val")
                    op_sym = st.selectbox("Operador", ["+","-","*","/"], key="ds_calc_op")
                    new_name = st.text_input("Nome nova coluna", f"{col_a}{op_sym}res", key="ds_calc_name")
                    if st.button("Criar Coluna Aritmética"):
                        try:
                            df_new = create_arithmetic(df, new_name, col_a, col_b, op_sym, val_b)
                            st.session_state['df_main'] = df_new
                            st.success(f"Coluna '{new_name}' criada.")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            elif op == "Condicional (IF)":
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not num_cols:
                    st.info("Sem colunas numéricas para condicional.")
                else:
                    col_target = st.selectbox("Coluna alvo", num_cols, key="ds_if_col")
                    operator = st.selectbox("Operador", [">","<",">=","<=","==","!="], key="ds_if_op")
                    thr = st.number_input("Threshold", value=0.0, key="ds_if_thr")
                    true_lab = st.text_input("Rótulo se True", "ALTO", key="ds_if_true")
                    false_lab = st.text_input("Rótulo se False", "BAIXO", key="ds_if_false")
                    name_if = st.text_input("Nome nova coluna", f"{col_target}_cat", key="ds_if_name")
                    if st.button("Criar Condicional"):
                        df_new = create_conditional(df, name_if, col_target, operator, thr, true_lab, false_lab)
                        st.session_state['df_main'] = df_new
                        st.success("Coluna condicional criada.")
            elif op == "Extrair Data":
                date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
                if not date_cols:
                    st.warning("Converta a coluna para Data na aba Renomear/Converter primeiro.")
                else:
                    dc = st.selectbox("Coluna Data", date_cols, key="ds_date_col")
                    comp = st.selectbox("Componente", ["year","month","day","weekday","quarter"], key="ds_date_comp")
                    newn = st.text_input("Nome nova coluna", f"{dc}_{comp}", key="ds_date_new")
                    if st.button("Extrair componente"):
                        df_new = extract_date_part(df, dc, comp, newn)
                        st.session_state['df_main'] = df_new
                        st.success("Componente extraído.")
            elif op == "Split Texto":
                text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
                if not text_cols:
                    st.info("Sem colunas de texto para split.")
                else:
                    tc = st.selectbox("Coluna Texto", text_cols, key="ds_split_col")
                    sep = st.text_input("Separador", " ", key="ds_split_sep")
                    idx = st.number_input("Índice da parte (0=primeiro)", min_value=0, value=0, key="ds_split_idx")
                    new_name = st.text_input("Nome da nova coluna", f"{tc}_part{idx}", key="ds_split_new")
                    if st.button("Split & Criar"):
                        try:
                            df_new = split_text(df, tc, sep, int(idx), new_name)
                            st.session_state['df_main'] = df_new
                            st.success("Split aplicado.")
                        except Exception as e:
                            st.error(f"Erro: {e}")

        # Rename / Convert
        with tabs[1]:
            st.subheader("✏️ Renomear / Converter Tipos")
            c1, c2 = st.columns(2)
            with c1:
                colr = st.selectbox("Renomear coluna", df.columns.tolist(), key="ds_ren_col")
                newname = st.text_input("Novo nome", key="ds_newname")
                if st.button("Aplicar Renome"):
                    if newname:
                        df.rename(columns={colr: newname}, inplace=True)
                        st.session_state['df_main'] = df
                        st.success("Renomeado.")
            with c2:
                colc = st.selectbox("Converter coluna", df.columns.tolist(), key="ds_conv_col")
                to = st.selectbox("Converter para", ["Data","Número (float)", "Texto"], key="ds_conv_to")
                if st.button("Converter Tipo"):
                    try:
                        if to == "Data":
                            df[colc] = pd.to_datetime(df[colc], errors='coerce')
                        elif to == "Número (float)":
                            df[colc] = pd.to_numeric(df[colc], errors='coerce')
                        else:
                            df[colc] = df[colc].astype(str)
                        st.session_state['df_main'] = df
                        st.success("Conversão aplicada.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        # Pivot/Merge
        with tabs[2]:
            st.subheader("🔀 Pivot / Unpivot / Merge")
            mode = st.radio("Modo", ["Pivot","Unpivot","Merge"], key="ds_pivot_mode")
            if mode == "Pivot":
                idx = st.multiselect("Index (linhas)", df.columns.tolist(), key="ds_pivot_idx")
                colp = st.selectbox("Columns", df.columns.tolist(), key="ds_pivot_col")
                val = st.selectbox("Values (num preferred)", df.select_dtypes(include=np.number).columns.tolist(), key="ds_pivot_val")
                agg = st.selectbox("Agg func", ["sum","mean","count"], key="ds_pivot_agg")
                if st.button("Pivotar"):
                    try:
                        res = pivot_transform(df, idx, colp, val, agg)
                        st.session_state['df_main'] = res
                        st.success("Pivot OK")
                    except Exception as e:
                        st.error(e)
            elif mode == "Unpivot":
                ids = st.multiselect("Id vars", df.columns.tolist(), key="ds_unpivot_id")
                vals = st.multiselect("Value vars", df.columns.tolist(), key="ds_unpivot_vals")
                if st.button("Unpivot"):
                    res = unpivot_transform(df, ids, vals)
                    st.session_state['df_main'] = res
                    st.success("Unpivot OK")
            else:
                st.markdown("🔁 Fazer merge com outro arquivo")
                uf = st.file_uploader("Carregar arquivo para mesclar", type=['csv','xlsx'], key="ds_merge_file")
                if uf:
                    try:
                        other = safe_read(uf)
                        other.columns = other.columns.astype(str).str.strip()
                        left_cols = st.multiselect("Left keys (este df)", df.columns.tolist(), key="ds_merge_left")
                        right_cols = st.multiselect("Right keys (arquivo carregado)", other.columns.tolist(), key="ds_merge_right")
                        how = st.selectbox("Tipo merge", ["left","inner","right","outer"], key="ds_merge_how")
                        if st.button("Executar Merge"):
                            if left_cols and right_cols and len(left_cols)==len(right_cols):
                                merged = merge_dfs(df, other, left_cols, right_cols, how=how)
                                st.session_state['df_main'] = merged
                                st.success("Merge OK")
                            else:
                                st.error("Selecione chaves correspondentes com o mesmo comprimento.")

        with tabs[3]:
            st.subheader("🔎 Filtros & Limpeza")
            colf = st.selectbox("Coluna para filtrar", df.columns.tolist(), key="ds_filter_col")
            if pd.api.types.is_numeric_dtype(df[colf]):
                mn, mx = float(df[colf].min()), float(df[colf].max())
                rv = st.slider("Intervalo", mn, mx, (mn, mx), key="ds_filter_range")
                if st.button("Aplicar filtro"):
                    st.session_state['df_main'] = df[(df[colf]>=rv[0]) & (df[colf]<=rv[1])]
                    st.success("Filtro aplicado.")
            elif np.issubdtype(df[colf].dtype, np.datetime64):
                min_d, max_d = df[colf].min().date(), df[colf].max().date()
                dr = st.date_input("Período", [min_d, max_d], key="ds_filter_date")
                if st.button("Aplicar filtro (data)"):
                    st.session_state['df_main'] = df[(df[colf].dt.date>=dr[0]) & (df[colf].dt.date<=dr[1])]
                    st.success("Filtro aplicado.")
            else:
                vals = df[colf].dropna().unique().tolist()
                sel = st.multiselect("Valores", vals[:500], default=vals[:10], key="ds_filter_vals")
                if st.button("Aplicar filtro (categoria)"):
                    st.session_state['df_main'] = df[df[colf].isin(sel)]
                    st.success("Filtro aplicado.")
            if st.button("Remover linhas com NA"):
                before = len(df)
                st.session_state['df_main'] = df.dropna()
                after = len(st.session_state['df_main'])
                st.success(f"Removidas {before-after} linhas.")

    # ----- Visual Studio -----
    elif menu == "Visual Studio":
        st.header("🎨 Visual Studio — Criador de Gráficos (PowerBI-like)")
        left, right = st.columns([1,2])
        with left:
            chart_type = st.selectbox("Tipo", ["Barras","Linha","Dispersão","Pizza","Histograma","Box","Heatmap","Treemap","Sunburst"])
            x = st.selectbox("X (categoria / tempo)", df.columns.tolist(), key="vs_x")
            y = None
            if chart_type not in ("Pizza","Histograma","Heatmap","Treemap","Sunburst"):
                y = st.selectbox("Y (valor)", df.select_dtypes(include=np.number).columns.tolist(), key="vs_y") if not df.select_dtypes(include=np.number).empty else None
            color = st.selectbox("Cor / Legenda (opcional)", ["Nenhum"] + df.columns.tolist(), key="vs_color")
            if color == "Nenhum": color = None
            agg = None
            if chart_type in ("Barras","Linha","Treemap","Sunburst"):
                use_agg = st.checkbox("Agrupar e agregar", value=True, key="vs_agg_use")
                if use_agg:
                    agg = st.selectbox("Função de agregação", ["sum","mean","count","min","max"], key="vs_agg_fun")
            theme = st.selectbox("Tema", ["plotly","plotly_dark","ggplot2","seaborn"], key="vs_theme")
            height = st.slider("Altura do gráfico", 300, 900, 500, key="vs_height")
            show_labels = st.checkbox("Mostrar rótulos", value=False, key="vs_labels")
            size_col = None
            if chart_type in ("Dispersão",):
                size_col = st.selectbox("Tamanho (opcional)", ["Nenhum"] + df.select_dtypes(include=np.number).columns.tolist(), key="vs_size")
                if size_col == "Nenhum": size_col = None
            title = st.text_input("Título do gráfico", f"{chart_type}: {y or ''} por {x}", key="vs_title")

            if st.button("Gerar Gráfico"):
                try:
                    fig = build_chart(chart_type, df, x=x, y=y, color=color, agg=agg, theme=theme, show_labels=show_labels, height=height, size=size_col)
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {"title": title, "type": chart_type}
                    st.success("Gráfico gerado — visualize à direita.")
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")

        with right:
            if st.session_state['last_fig'] is not None:
                st.plotly_chart(st.session_state['last_fig'], width="stretch")
                note = st.text_area("Nota para o relatório", key="vs_note")
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("➕ Adicionar ao Relatório"):
                        st.session_state['report_charts'].append({
                            "fig": st.session_state['last_fig'],
                            "title": st.session_state['last_meta'].get('title','(sem titulo)'),
                            "type": st.session_state['last_meta'].get('type',''),
                            "note": note
                        })
                        st.success("Adicionado ao relatório.")
                with c2:
                    if st.button("Salvar Preset"):
                        # store last chart spec in presets
                        presets = st.session_state.get('presets', {})
                        preset_name = st.text_input("Nome do preset (digite e pressione Enter)")
                        # we can't capture text input submit easily — quick fallback: timestamp
                        if not preset_name:
                            preset_name = f"preset_{len(presets)+1}_{datetime.now().strftime('%H%M%S')}"
                        presets[preset_name] = {"meta": st.session_state['last_meta'], "note": note}
                        st.session_state['presets'] = presets
                        st.success(f"Preset salvo: {preset_name}")
            else:
                st.info("Gere um gráfico na esquerda para visualizar aqui.")

    # ----- Dashboard Builder -----
    elif menu == "Relatório/Dashboard":
        st.header("📑 Relatório / Dashboard Builder")
        st.write("Arrume seu relatório. Você pode salvar/abrir presets (listas de gráficos).")
        cs = st.session_state['report_charts']
        left, right = st.columns([3,1])
        with right:
            if st.button("Salvar Relatório (preset)"):
                pres_name = st.text_input("Nome preset", key="rb_preset_name")
                if not pres_name:
                    pres_name = f"report_{len(st.session_state['presets'])+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state['presets'][pres_name] = cs.copy()
                st.success(f"Relatório salvo: {pres_name}")
            presets_list = list(st.session_state['presets'].keys())
            if presets_list:
                sel = st.selectbox("Abrir Preset salvo", ["Nenhum"] + presets_list, key="rb_open_preset")
                if sel and sel != "Nenhum":
                    st.session_state['report_charts'] = st.session_state['presets'][sel].copy()
                    st.success(f"Abrindo preset: {sel}")
            if st.button("Limpar Relatório"):
                st.session_state['report_charts'] = []
                st.success("Relatório limpo.")

        if not cs:
            st.info("Relatório vazio — vá em Visual Studio e adicione gráficos.")
        else:
            for i, c in enumerate(cs):
                st.markdown("---")
                c1, c2 = st.columns([3,1])
                with c1:
                    try:
                        st.plotly_chart(c['fig'], width="stretch")
                    except Exception:
                        # fallback: write title
                        st.write(c.get('title','(sem titulo)'))
                with c2:
                    st.write(f"**{c.get('title','(sem titulo)')}**")
                    st.write(f"Tipo: {c.get('type','')}")
                    if c.get('note'):
                        st.info(c.get('note'))
                    if st.button(f"Remover gráfico #{i+1}", key=f"rb_del_{i}"):
                        st.session_state['report_charts'].pop(i)
                        st.experimental_rerun()

    # ----- NLP & Texto -----
    elif menu == "NLP & Texto":
        st.header("🧠 NLP & Texto (leve)")
        st.write("Ferramentas básicas: sentimento (lexicon), tokenização simples, word frequencies.")
        text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
        if not text_cols:
            st.info("Nenhuma coluna de texto identificada. Converta colunas para texto no Data Studio.")
        else:
            tc = st.selectbox("Coluna de texto", text_cols)
            st.write("Amostra de 5 linhas:")
            safe_display_dataframe(df[[tc]].head(5), height=200)
            if st.button("Analisar Sentimentos (lexicon rápido)"):
                srs = df[tc].fillna("").astype(str)
                senti = srs.apply(simple_sentiment)
                df_res = pd.DataFrame(list(senti), columns=['sent_label','sent_score'])
                df_out = pd.concat([df[[tc]].reset_index(drop=True), df_res], axis=1)
                st.write("Distribuição de sentimento:")
                st.bar_chart(df_out['sent_label'].value_counts())
                safe_display_dataframe(df_out.head(200))
                # add to df_main if user wants
                if st.button("Adicionar coluna 'sentimento' ao dataset"):
                    st.session_state['df_main'][f"{tc}_sentiment"] = df_out['sent_label'].values
                    st.success("Coluna adicionada ao dataset.")

            if st.button("Frequência de palavras (top 40)"):
                import re
                s = " ".join(df[tc].dropna().astype(str).tolist()).lower()
                tokens = re.findall(r"\b\w{3,}\b", s)
                from collections import Counter
                cnt = Counter(tokens)
                top = pd.Series(dict(cnt.most_common(40)))
                st.bar_chart(top)

    # ----- Anomalias & Clustering -----
    elif menu == "Anomalias & Clustering":
        st.header("🚨 Anomalias & Clusterização")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("Sem colunas numéricas para clustering/anomalia.")
        else:
            st.subheader("Anomalias (Isolation Forest + Z-score)")
            col_an = st.selectbox("Coluna (z-score)", numeric_cols, key="an_col")
            z_thresh = st.slider("Z-score thresh", 2.0, 6.0, 3.0, step=0.5, key="an_z_thresh")
            if st.button("Detectar anomalias (z-score)"):
                out_z = zscore_anomaly(df, col_an, thresh=z_thresh)
                st.write(f"Anomalias por z-score: {len(out_z)}")
                if not out_z.empty:
                    safe_display_dataframe(out_z.head(200))
            if st.button("Detectar anomalias (IsolationForest)"):
                # Use 1D or multivariate
                try:
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    arr = df[numeric_cols].dropna()
                    iso.fit(arr)
                    preds = iso.predict(arr)
                    out_iso = arr[preds==-1]
                    st.write(f"Anomalias IsolationForest: {len(out_iso)}")
                    safe_display_dataframe(out_iso.head(200))
                except Exception as e:
                    st.error(f"Erro IsolationForest: {e}")

            st.markdown("---")
            st.subheader("Clusterização (KMeans) com PCA (seguro)")
            feats = st.multiselect("Features (numéricas)", numeric_cols, default=numeric_cols[:4], key="cl_feats")
            k = st.slider("K", 2, 12, 3, key="cl_k")
            if st.button("Rodar KMeans"):
                if len(feats) < 1:
                    st.error("Selecione ao menos 1 feature.")
                else:
                    X = df[feats].dropna()
                    if X.shape[0] < k:
                        st.error(f"Não há amostras suficientes ({X.shape[0]}) para k={k}.")
                    else:
                        try:
                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(X)
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            labels = kmeans.fit_predict(Xs)
                            # PCA safe: check dims
                            n_samples, n_features = Xs.shape
                            comps = min(2, n_samples, n_features)
                            if comps < 1:
                                st.error("PCA não aplicável (amostras/features insuficientes).")
                            else:
                                pca = PCA(n_components=comps)
                                pcs = pca.fit_transform(Xs)
                                if comps == 1:
                                    fig = px.scatter(x=pcs[:,0], y=np.zeros(len(pcs)), color=labels.astype(str), title="Clusters (1D PCA)", labels={'x':'PC1','y':''})
                                else:
                                    fig = px.scatter(x=pcs[:,0], y=pcs[:,1], color=labels.astype(str), title="Clusters (PCA 2D)", labels={'x':'PC1','y':'PC2'})
                                st.plotly_chart(fig, width="stretch")
                                # attach cluster to working df (align by index)
                                df_cl = X.copy()
                                df_cl['cluster'] = labels
                                st.session_state['df_main'] = pd.merge(df, df_cl[['cluster']], left_index=True, right_index=True, how='left')
                                st.success("Clusterização aplicada — coluna 'cluster' adicionada ao dataset.")
                        except Exception as e:
                            st.error(f"Erro no clustering/PCA: {e}")

    # ----- SQL Lab -----
    elif menu == "SQL Lab":
        st.header("💾 SQL Lab (duckdb preferred; sqlite fallback)")
        st.info("A tabela disponível chamará 'dados'. Use SQL para explorar/transformar.")
        q = st.text_area("Query SQL (ex: SELECT count(*) FROM dados)", value="SELECT * FROM dados LIMIT 100", height=180)
        if st.button("Executar Query"):
            res, err = run_sql(df, q)
            if err:
                st.error(f"Erro SQL: {err}")
            else:
                st.success(f"{len(res)} linhas retornadas")
                safe_display_dataframe(res.head(1000))
                csv = res.to_csv(index=False).encode("utf-8")
                st.download_button("Baixar resultado (CSV)", csv, "sql_result.csv", "text/csv")

    # ----- Export -----
    elif menu == "Exportar":
        st.header("📤 Exportar & Relatórios")
        st.markdown("Baixe o dataset atual, relatórios e o PDF do relatório com imagens (quando possível).")
        df_export = st.session_state['df_main']
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (tratado)", csv, "dados_tratados.csv", "text/csv")
        # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Dados')
            # include summary sheet
            desc = df_export.describe(include='all').transpose()
            desc.to_excel(writer, sheet_name='Resumo')
        st.download_button("Download Excel", buffer.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # PDF report
        kpis = {"rows": len(df_export), "cols": df_export.shape[1], "nulls": int(df_export.isna().sum().sum()), "dups": int(df_export.duplicated().sum())}
        if st.button("Gerar PDF Executivo com gráficos do relatório"):
            charts = st.session_state.get('report_charts', [])
            try:
                pdf_bytes = generate_pdf_with_charts(df_export, charts, kpis)
                st.download_button("Baixar PDF", pdf_bytes, "report.pdf", "application/pdf")
                st.success("PDF gerado.")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    # persist
    st.session_state['df_main'] = st.session_state.get('df_main', df)

if __name__ == "__main__":
    main()

