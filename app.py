# app.py
"""
Enterprise Analytics — BI MAX-PRO (Corrected single-file app)
Created for robust no-code EDA / Visual / ETL / Dashboard workflows.
"""

import streamlit as st

# Config deve ser o primeiro comando Streamlit
st.set_page_config(
    page_title="Enterprise Analytics — BI MAX-PRO",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, os, logging, json, math, tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Optional heavy libs
_HAS_DUCKDB = False
try:
    import duckdb
    _HAS_DUCKDB = True
except Exception:
    import sqlite3

_HAS_JOBLIB = False
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None

# --- ALTERAÇÃO AQUI: KALEIDO DESATIVADO PARA EVITAR TRAVAMENTO ---
_HAS_KALEIDO = False
# try:
#     import kaleido
#     _HAS_KALEIDO = True
# except Exception:
#     _HAS_KALEIDO = False
# -----------------------------------------------------------------

# ML libs (light usage)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enterprise-bi")

# -------------------- Utilities --------------------

def try_read_csv(file_obj, encodings=("utf-8","latin1","cp1252","iso-8859-1")):
    last_exc = None
    for enc in encodings:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception as e:
            last_exc = e
            continue
    file_obj.seek(0)
    return pd.read_csv(file_obj, engine="python", encoding_errors="ignore")

def safe_read(file):
    """Read uploaded file or local path robustly. Returns DataFrame or raises."""
    if file is None:
        raise ValueError("No file provided.")
    try:
        if hasattr(file, "read"):
            name = getattr(file, "name", "")
            if name.lower().endswith(".csv"):
                try:
                    df = try_read_csv(file)
                except Exception:
                    file.seek(0)
                    df = pd.read_csv(file, engine="python", encoding_errors="ignore")
            else:
                # excel
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
        
        # clean column names
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
    """Display dataframe with fallback to string coercion on Arrow errors."""
    try:
        st.dataframe(df, width=None, height=height, use_container_width=True)
    except Exception as e:
        logger.warning("st.dataframe Arrow conversion failed; coercing problematic columns to str.")
        df2 = df.copy()
        for c in df2.columns:
            if df2[c].dtype.kind in ("i","u") and df2[c].apply(lambda x: isinstance(x, str)).any():
                df2[c] = df2[c].astype(str)
            if df2[c].dtype == object:
                # avoid heavy casts; only cast dict/list entries
                if df2[c].apply(lambda x: isinstance(x, (list, dict))).any():
                    df2[c] = df2[c].astype(str)
        st.dataframe(df2.astype(str), width=None, height=height, use_container_width=True)

# -------------------- Data quality --------------------

def missing_heatmap(df: pd.DataFrame, max_cols=60):
    m = df.isna().astype(int)
    if m.shape[1] > max_cols:
        m = m.iloc[:, :max_cols]
    fig = px.imshow(m.T, aspect='auto', color_continuous_scale=['#e0e0e0','#ff6b6b'],
                    labels={'x':'Index','y':'Columns'}, title='Missing map (1=missing)')
    fig.update_layout(margin=dict(l=60,r=10,t=40,b=30))
    return fig

def detect_outliers_iqr(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if s.empty:
        return pd.DataFrame(columns=df.columns)
    q1, q3 = s.quantile([0.25,0.75])
    iqr = q3 - q1
    low = q1 - 1.5*iqr
    high = q3 + 1.5*iqr
    return df[(df[col] < low) | (df[col] > high)]

def zscore_anomaly(df: pd.DataFrame, col: str, thresh=3.0):
    s = df[col].dropna()
    if s.empty:
        return pd.DataFrame(columns=df.columns)
    zs = np.abs(stats.zscore(s))
    return df.loc[s.index[zs > thresh]]

# -------------------- ETL helpers --------------------

def create_arithmetic(df: pd.DataFrame, new_col: str, a: str, b: Optional[str], op: str, b_value: Optional[float] = None):
    df = df.copy()
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
    aggmap = {'sum':'sum','mean':'mean','count':'count','min':'min','max':'max'}
    if agg not in aggmap: agg = 'sum'
    res = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggmap[agg]).reset_index()
    res.columns = [("_".join(map(str,c)) if isinstance(c, tuple) else str(c)).strip() for c in res.columns]
    return res

def unpivot_transform(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name='variable', value_name='value'):
    return df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: List[str], how='left'):
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)

# -------------------- Visualization builder --------------------

def build_chart(chart_type: str, df: pd.DataFrame, x: Optional[str]=None, y: Optional[str]=None, color: Optional[str]=None,
                agg: Optional[str]=None, theme: str='plotly', show_labels: bool=True, height: int=450, size: Optional[str]=None):
    plot_df = df.copy()
    if agg and x and y and chart_type in ("Barras","Linha","Pizza","Treemap","Sunburst"):
        if color:
            plot_df = plot_df.groupby([x,color], dropna=False)[y].agg(agg).reset_index()
        else:
            plot_df = plot_df.groupby(x, dropna=False)[y].agg(agg).reset_index()

    if chart_type == "Barras":
        fig = px.bar(plot_df, x=x, y=y, color=color, text_auto=show_labels, template=theme)
    elif chart_type == "Linha":
        fig = px.line(plot_df, x=x, y=y, color=color, markers=True, template=theme)
    elif chart_type in ("Dispersão","Scatter"):
        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, template=theme)
    elif chart_type == "Pizza":
        fig = px.pie(plot_df, names=x, values=y, template=theme)
    elif chart_type == "Histograma":
        fig = px.histogram(plot_df, x=x, color=color, nbins=30, template=theme, text_auto=show_labels)
    elif chart_type == "Box":
        fig = px.box(plot_df, x=x, y=y, color=color, template=theme)
    elif chart_type == "Heatmap":
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True, template=theme, title="Correlation matrix")
    elif chart_type == "Treemap":
        fig = px.treemap(plot_df, path=[x, color] if color else [x], values=y, template=theme)
    elif chart_type == "Sunburst":
        fig = px.sunburst(plot_df, path=[x, color] if color else [x], values=y, template=theme)
    else:
        fig = go.Figure()
    fig.update_layout(height=height, title=f"{chart_type}: {y or ''} vs {x or ''}")
    return fig

# -------------------- Export helpers --------------------

def fig_to_image_bytes(fig: go.Figure, fmt='png', scale=2) -> Optional[bytes]:
    try:
        return fig.to_image(format=fmt, scale=scale)
    except Exception as e:
        logger.warning(f"fig_to_image_bytes failed: {e}")
        return None

def generate_pdf_with_charts(df: pd.DataFrame, charts: List[Dict[str,Any]], kpis: Dict[str,Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,"Relatorio Executivo — Enterprise BI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,8,f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    
    # KPIs table
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
    
    # short stats
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Resumo Estatitico (Top numericas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
                    pdf.cell(w[i],7,str(val)[:28],1,align='C' if i>0 else 'L', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)
    
    # charts
    for ch in charts:
        pdf.add_page()
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,ch.get('title','(sem titulo)'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        img = None
        try:
            img = fig_to_image_bytes(ch['fig'], fmt='png', scale=2)
        except Exception:
            img = None
        if img:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            try:
                tmp.write(img)
                tmp.close()
                pdf.image(tmp.name, x=15, w=180)
            except Exception:
                pdf.multi_cell(0,6,"(Nao foi possivel inserir imagem)")
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        else:
            pdf.multi_cell(0,6, f"Tipo: {ch.get('type','')} - Nota: {ch.get('note','')}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# -------------------- Simple sentiment --------------------
_POS_WORDS = {"good","great","excelent","excellent","bom","otimo","ótimo","agradavel","satisfeito","positivo","love","adorei","recomendo"}
_NEG_WORDS = {"bad","terrible","horrible","ruim","péssimo","pessimo","odioso","odio","frustrado","frustrante","pior","hate","detestei","não gostei","nao gostei","dinheiro jogado fora","devolvi"}

def simple_sentiment(text: str) -> Tuple[str,float]:
    if not isinstance(text, str) or text.strip()=="":
        return ("neutral",0.0)
    txt = text.lower()
    pos = sum(txt.count(w) for w in _POS_WORDS)
    neg = sum(txt.count(w) for w in _NEG_WORDS)
    score = (pos-neg)/max(1,(pos+neg))
    if score > 0.1: return ("positive",score)
    if score < -0.1: return ("negative",score)
    return ("neutral",score)

# -------------------- SQL Lab --------------------

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

# -------------------- Session init & UI helpers --------------------

if 'df_raw' not in st.session_state: st.session_state['df_raw'] = pd.DataFrame()
if 'df_main' not in st.session_state: st.session_state['df_main'] = pd.DataFrame()
if 'report_charts' not in st.session_state: st.session_state['report_charts'] = []
if 'last_fig' not in st.session_state: st.session_state['last_fig'] = None
if 'last_meta' not in st.session_state: st.session_state['last_meta'] = {}
if 'presets' not in st.session_state: st.session_state['presets'] = {}

st.markdown("""<style>
.block-container{padding-top:1rem}
.stButton>button{width:100%}
.metric-card{background:#f3f6fb;border-left:6px solid #4F8BF9;padding:12px;border-radius:6px;margin-bottom:8px}
@media (prefers-color-scheme: dark){
.metric-card{background:#262730;color:#ddd;border-left:6px solid #ffbd45}}
</style>
""", unsafe_allow_html=True)

def sidebar_area():
    st.sidebar.title("Enterprise Analytics")
    uploaded = st.sidebar.file_uploader("Arraste CSV ou Excel (XLSX)", type=['csv','xlsx'], help="CSV recomendado para arquivos grandes.")
    use_local = st.sidebar.checkbox("Usar arquivo local (DEV)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.header("SQL Lab (opcional)")
    st.sidebar.info("DuckDB usado se estiver disponível; caso contrário sqlite é usado.")
    if st.sidebar.button("Resetar sessão"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    return uploaded, use_local

# -------------------- Main app --------------------

def main():
    uploaded, use_local = sidebar_area()
    st.title("Enterprise Analytics — BI MAX-PRO")
    st.caption("Robusto: EDA, No-code ETL, Visual Builder, Dashboard, NLP, Clustering, SQL, Export")

    # load file only when explicitly provided
    if uploaded is None and use_local:
        local_path = os.environ.get("DEV_LOCAL_PATH", "/mnt/data/uploaded_dataset.csv")
        if os.path.exists(local_path):
            try:
                df = safe_read(local_path)
                st.session_state['df_raw'] = df.copy()
                st.session_state['df_main'] = df.copy()
                st.success("Dados carregados do caminho local.")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar local: {e}")
        else:
            st.sidebar.warning("Caminho local selecionado, mas arquivo não encontrado. Faça upload manualmente.")
    elif uploaded is not None:
        try:
            df = safe_read(uploaded)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success(f"Arquivo '{getattr(uploaded,'name','uploaded')}' carregado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler arquivo: {e}")

    if st.session_state['df_main'].empty:
        st.info("""Sem dados carregados — arraste um CSV/Excel na barra lateral.
Fluxo recomendado:
1) Data Studio -> preparar dados
2) Visual Studio -> criar graficos
3) Dashboard -> montar e exportar
""")
        return

    df = st.session_state['df_main']

    menu = st.radio("", ["Data Quality","Data Studio","Visual Studio","Relatório/Dashboard","NLP & Texto","Anomalias & Clustering","SQL Lab","Exportar"], horizontal=True)

    # Data Quality
    if menu == "Data Quality":
        st.header("Data Quality & EDA")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Linhas</h3><h2>{format_number(df.shape[0])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Colunas</h3><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{format_number(int(df.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{format_number(int(df.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Missing Heatmap")
        st.plotly_chart(missing_heatmap(df), use_container_width=True)
        
        st.subheader("Perfil das Colunas")
        col1,col2,col3 = st.columns(3)
        with col1:
            st.caption("Numéricas")
            num = df.select_dtypes(include=np.number)
            if not num.empty:
                st.dataframe(num.describe().T[['mean','std','min','max']], use_container_width=True)
                col_sel = st.selectbox("Detectar outliers (IQR) em:", num.columns.tolist(), key="dq_out_col")
                out_df = detect_outliers_iqr(df, col_sel)
                st.write(f"Outliers IQR: {len(out_df)}")
                if st.checkbox("Mostrar amostra de outliers", key="dq_show_outs"):
                    safe_display_dataframe(out_df.head(200))
            else:
                st.info("Sem colunas numericas.")
        with col2:
            st.caption("Categóricas")
            cat = df.select_dtypes(include=['object','category'])
            if not cat.empty:
                st.dataframe(pd.DataFrame({'unique':cat.nunique(),'missing':cat.isna().sum(), '%missing':(cat.isna().mean()*100).round(2)}), use_container_width=True)
            else:
                st.info("Sem colunas de texto.")
        with col3:
            st.caption("Datas")
            date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if date_cols:
                for dc in date_cols:
                    st.write(f"**{dc}**: {df[dc].min()} → {df[dc].max()}")
            else:
                st.info("Nenhuma coluna de data detectada automaticamente.")
        st.markdown("---")
        st.subheader("Visualizar tabela")
        safe_display_dataframe(df, height=400)

    # Data Studio
    elif menu == "Data Studio":
        st.header("Data Studio — ETL No-Code")
        tabs = st.tabs(["Criar Coluna","Renomear/Converter","Pivot/Merge","Filtros & Limpeza"])
        with tabs[0]:
            st.subheader("Criar Coluna")
            op = st.selectbox("Operacao", ["Aritmetica","Condicional (IF)","Extrair Data","Split Texto"])
            if op == "Aritmetica":
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not num_cols:
                    st.info("Sem colunas numericas para operacoes.")
                else:
                    a = st.selectbox("Col A", num_cols, key="ds_a")
                    mode = st.radio("Col B/Valor", ["Outra Coluna","Valor Fixo"], key="ds_mode")
                    b = None
                    val = None
                    if mode == "Outra Coluna":
                        b = st.selectbox("Col B", num_cols, key="ds_b")
                    else:
                        val = st.number_input("Valor", value=1.0, key="ds_val")
                    op_sym = st.selectbox("Operador", ["+","-","*","/"], key="ds_op")
                    new_name = st.text_input("Nome nova coluna", f"{a}{op_sym}res", key="ds_newcol")
                    if st.button("Criar Coluna Aritmetica"):
                        try:
                            df_new = create_arithmetic(df, new_name, a, b, op_sym, val)
                            st.session_state['df_main'] = df_new
                            st.success("Coluna criada.")
                        except Exception as e:
                            st.error(e)
            elif op == "Condicional (IF)":
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    col = st.selectbox("Col alvo", num_cols, key="ds_if_col")
                    operator = st.selectbox("Operador", [">","<",">=","<=","==","!="], key="ds_if_op")
                    thr = st.number_input("Threshold", value=0.0, key="ds_if_thr")
                    tlabel = st.text_input("Label True","ALTO", key="ds_if_true")
                    flabel = st.text_input("Label False","BAIXO", key="ds_if_false")
                    newn = st.text_input("Nome nova coluna", f"{col}_cat", key="ds_if_new")
                    if st.button("Criar Condicional"):
                        st.session_state['df_main'] = create_conditional(df, newn, col, operator, thr, tlabel, flabel)
                        st.success("Condicional criada.")
                else:
                    st.info("Sem colunas numericas.")
            elif op == "Extrair Data": 
                date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
                if not date_cols:
                    st.warning("Converta coluna para data na aba Renomear/Converter.")
                else:
                    dc = st.selectbox("Coluna data", date_cols, key="ds_date_col")
                    comp = st.selectbox("Componente", ["year","month","day","weekday","quarter"], key="ds_date_comp")
                    newn = st.text_input("Nome nova coluna", f"{dc}_{comp}", key="ds_date_new")
                    if st.button("Extrair componente"):
                        st.session_state['df_main'] = extract_date_part(df, dc, comp, newn)
                        st.success("Extraido.")
            else:
                text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
                if not text_cols:
                    st.info("Sem colunas de texto.")
                else:
                    tc = st.selectbox("Coluna texto", text_cols, key="ds_split_tc")
                    sep = st.text_input("Separador", " ", key="ds_split_sep")
                    idx = st.number_input("Indice (0=primeiro)", min_value=0, value=0, key="ds_split_idx")
                    newn = st.text_input("Nome nova coluna", f"{tc}_part{idx}", key="ds_split_new")
                    if st.button("Split & Criar"):
                        try:
                            st.session_state['df_main'] = split_text(df, tc, sep, int(idx), newn)
                            st.success("Split aplicado.")
                        except Exception as e:
                            st.error(e)
        with tabs[1]:
            st.subheader("Renomear / Converter")
            colr = st.selectbox("Renomear coluna", df.columns.tolist(), key="ds_ren_col")
            newname = st.text_input("Novo nome", key="ds_ren_new")
            if st.button("Aplicar Renome"):
                if newname:
                    df.rename(columns={colr:newname}, inplace=True)
                    st.session_state['df_main'] = df
                    st.success("Renomeado.")
            st.markdown("---")
            colc = st.selectbox("Converter coluna", df.columns.tolist(), key="ds_conv_col")
            to = st.selectbox("Converter para", ["Data","Numero (float)","Texto"], key="ds_conv_to")
            if st.button("Converter Tipo"):
                try:
                    if to == "Data": df[colc] = pd.to_datetime(df[colc], errors='coerce')
                    elif to == "Numero (float)": df[colc] = pd.to_numeric(df[colc], errors='coerce')
                    else: df[colc] = df[colc].astype(str)
                    st.session_state['df_main'] = df
                    st.success("Convertido.")
                except Exception as e:
                    st.error(e)
        with tabs[2]:
            st.subheader("Pivot / Unpivot / Merge")
            mode = st.radio("Modo", ["Pivot","Unpivot","Merge"], key="ds_mode_pm")
            if mode == "Pivot":
                idx = st.multiselect("Index", df.columns.tolist(), key="ds_pivot_idx")
                colp = st.selectbox("Columns", df.columns.tolist(), key="ds_pivot_col")
                val = st.selectbox("Values", df.select_dtypes(include=np.number).columns.tolist(), key="ds_pivot_val")
                agg = st.selectbox("Agg", ["sum","mean","count"], key="ds_pivot_agg")
                if st.button("Pivotar"):
                    try:
                        st.session_state['df_main'] = pivot_transform(df, idx, colp, val, agg)
                        st.success("Pivot OK")
                    except Exception as e:
                        st.error(e)
            elif mode == "Unpivot":
                ids = st.multiselect("Id vars", df.columns.tolist(), key="ds_unpivot_id")
                vals = st.multiselect("Value vars", df.columns.tolist(), key="ds_unpivot_vals")
                if st.button("Unpivot"):
                    st.session_state['df_main'] = unpivot_transform(df, ids, vals)
                    st.success("Unpivot ok")
            else:
                uf = st.file_uploader("Arquivo para merge (csv/xlsx)", key="ds_merge_file")
                if uf:
                    try:
                        other = safe_read(uf)
                        left = st.multiselect("Chaves left (este df)", df.columns.tolist(), key="ds_merge_left")
                        right = st.multiselect("Chaves right (arquivo)", other.columns.tolist(), key="ds_merge_right")
                        how = st.selectbox("Tipo", ["left","inner","right","outer"], key="ds_merge_how")
                        if st.button("Executar merge"):
                            if left and right and len(left)==len(right):
                                st.session_state['df_main'] = merge_dfs(df, other, left, right, how=how)
                                st.success("Merge OK")
                            else:
                                st.error("Selecione chaves correspondentes.")
                    except Exception as e:
                        st.error(e)
        with tabs[3]:
            st.subheader("Filtros e Limpeza")
            colf = st.selectbox("Coluna para filtrar", df.columns.tolist(), key="ds_filter_col")
            if pd.api.types.is_numeric_dtype(df[colf]):
                mn, mx = float(df[colf].min()), float(df[colf].max())
                rng = st.slider("Range", mn, mx, (mn, mx), key="ds_filter_range")
                if st.button("Aplicar filtro"):
                    st.session_state['df_main'] = df[(df[colf]>=rng[0]) & (df[colf]<=rng[1])]
                    st.success("Filtrado.")
            elif np.issubdtype(df[colf].dtype, np.datetime64):
                min_d, max_d = df[colf].min().date(), df[colf].max().date()
                dr = st.date_input("Periodo", [min_d, max_d], key="ds_filter_date")
                if st.button("Aplicar filtro (data)"):
                    st.session_state['df_main'] = df[(df[colf].dt.date>=dr[0]) & (df[colf].dt.date<=dr[1])]
                    st.success("Filtrado.")
            else:
                vals = df[colf].dropna().unique().tolist()
                sel = st.multiselect("Valores", vals[:500], default=vals[:10], key="ds_filter_vals")
                if st.button("Aplicar filtro (categoria)"):
                    st.session_state['df_main'] = df[df[colf].isin(sel)]
                    st.success("Filtrado.")
            if st.button("Remover linhas com NA"):
                before = len(df)
                st.session_state['df_main'] = df.dropna()
                after = len(st.session_state['df_main'])
                st.success(f"Removidas {before-after} linhas.")

    # Visual Studio
    elif menu == "Visual Studio":
        st.header("Visual Studio — Criador de Graficos")
        left,right = st.columns([1,2])
        with left:
            chart_type = st.selectbox("Tipo", ["Barras","Linha","Dispersao","Pizza","Histograma","Box","Heatmap","Treemap","Sunburst"], key="vs_type")
            x = st.selectbox("X", df.columns.tolist(), key="vs_x")
            y = None
            if chart_type not in ("Pizza","Histograma","Heatmap","Treemap","Sunburst"):
                y = st.selectbox("Y", df.select_dtypes(include=np.number).columns.tolist(), key="vs_y") if not df.select_dtypes(include=np.number).empty else None
            color = st.selectbox("Cor", ["Nenhum"]+df.columns.tolist(), key="vs_color")
            if color == "Nenhum": color = None
            agg = None
            if chart_type in ("Barras","Linha","Treemap","Sunburst"):
                use_agg = st.checkbox("Agrupar e agregar", value=True, key="vs_use_agg")
                if use_agg:
                    agg = st.selectbox("Agg func", ["sum","mean","count","min","max"], key="vs_agg_fun")
            theme = st.selectbox("Tema", ["plotly","plotly_dark","ggplot2","seaborn"], key="vs_theme")
            height = st.slider("Altura", 300, 900, 500, key="vs_height")
            show_labels = st.checkbox("Mostrar rotulos", value=False, key="vs_labels")
            size_col = None
            if chart_type in ("Dispersao",):
                size_col = st.selectbox("Tamanho (opcional)", ["Nenhum"]+df.select_dtypes(include=np.number).columns.tolist(), key="vs_size")
                if size_col == "Nenhum": size_col = None
            title = st.text_input("Titulo", f"{chart_type}: {y or ''} por {x}", key="vs_title")
            if st.button("Gerar Grafico"):
                try:
                    # map names to builder types
                    map_type = chart_type.replace("Dispersao","Scatter")
                    fig = build_chart(map_type, df, x=x, y=y, color=color, agg=agg, theme=theme, show_labels=show_labels, height=height, size=size_col)
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {'title': title, 'type': chart_type}
                    st.success("Grafico gerado.")
                except Exception as e:
                    st.error(e)
        with right:
            if st.session_state['last_fig'] is not None:
                st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
                note = st.text_area("Nota para o relatorio", key="vs_note")
                c1,c2 = st.columns([1,1])
                with c1:
                    if st.button("Adicionar ao relatorio"):
                        st.session_state['report_charts'].append({'fig': st.session_state['last_fig'], 'title': st.session_state['last_meta'].get('title','(sem titulo)'), 'type': st.session_state['last_meta'].get('type',''), 'note': note})
                        st.success("Adicionado.")
                with c2:
                    if st.button("Salvar preset"):
                        presets = st.session_state.get('presets',{})
                        pname = f"preset_{len(presets)+1}_{datetime.now().strftime('%H%M%S')}"
                        presets[pname] = {'meta': st.session_state['last_meta'], 'note': note}
                        st.session_state['presets'] = presets
                        st.success(f"Preset salvo: {pname}")
            else:
                st.info("Gere um grafico para visualizar.")

    # Dashboard
    elif menu == "Relatório/Dashboard":
        st.header("Relatorio / Dashboard Builder")
        cs = st.session_state.get('report_charts',[])
        right = st.sidebar
        right.header("Relatorio")
        if right.button("Salvar relatorio (preset)"):
            pname = right.text_input("Nome do preset", key="rb_pres_name")
            if not pname:
                pname = f"report_{len(st.session_state['presets'])+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state['presets'][pname] = cs.copy()
            st.success(f"Preset salvo: {pname}")
        if not cs:
            st.info("Relatorio vazio — adicione graficos no Visual Studio.")
        else:
            for i,c in enumerate(cs):
                st.markdown("---")
                cols = st.columns([3,1])
                try:
                    cols[0].plotly_chart(c['fig'], use_container_width=True)
                except Exception:
                    cols[0].write(c.get('title','(sem titulo)'))
                cols[1].write(f"**{c.get('title','(sem titulo)')}**")
                if c.get('note'): cols[1].info(c.get('note'))
                if cols[1].button(f"Remover #{i+1}", key=f"rem_{i}"):
                    st.session_state['report_charts'].pop(i)
                    st.rerun()

    # NLP & Texto
    elif menu == "NLP & Texto":
        st.header("NLP & Texto (leve)")
        text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
        if not text_cols:
            st.info("Nenhuma coluna de texto detectada.")
        else:
            tc = st.selectbox("Coluna de texto", text_cols)
            safe_display_dataframe(df[[tc]].head(5), height=200)
            if st.button("Analisar sentimentos (lexicon)"):
                srs = df[tc].fillna("").astype(str)
                senti = srs.apply(simple_sentiment)
                df_res = pd.DataFrame(list(senti), columns=['sent_label','sent_score'])
                df_out = pd.concat([df[[tc]].reset_index(drop=True), df_res], axis=1)
                st.bar_chart(df_out['sent_label'].value_counts())
                safe_display_dataframe(df_out.head(200))
                if st.button("Adicionar coluna sentimento ao dataset"):
                    st.session_state['df_main'][f"{tc}_sentiment"] = df_out['sent_label'].values
                    st.success("Coluna adicionada.")
            if st.button("Frequencia de palavras (top 40)"):
                import re
                s = " ".join(df[tc].dropna().astype(str).tolist()).lower()
                tokens = re.findall(r"\b\w{3,}\b", s)
                from collections import Counter
                top = pd.Series(dict(Counter(tokens).most_common(40)))
                st.bar_chart(top)

    # Anomalias & Clustering
    elif menu == "Anomalias & Clustering":
        st.header("Anomalias & Clustering")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("Sem colunas numericas.")
        else:
            st.subheader("Deteccao de anomalias")
            col_an = st.selectbox("Coluna (z-score)", numeric_cols, key="an_col")
            zt = st.slider("Z thresh", 2.0, 6.0, 3.0, step=0.5, key="an_z")
            if st.button("Detectar zscore"):
                outz = zscore_anomaly(df, col_an, thresh=zt)
                st.write(f"Encontradas: {len(outz)}")
                safe_display_dataframe(outz.head(200))
            if st.button("Detectar IsolationForest"):
                try:
                    arr = df[numeric_cols].dropna()
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    preds = iso.fit_predict(arr)
                    out_iso = arr[preds==-1]
                    st.write(f"Anomalias IF: {len(out_iso)}")
                    safe_display_dataframe(out_iso.head(200))
                except Exception as e:
                    st.error(e)
            st.markdown("---")
            st.subheader("Clusterizacao (KMeans com PCA seguro)")
            feats = st.multiselect("Features", numeric_cols, default=numeric_cols[:4], key="cl_feats")
            k = st.slider("K", 2, 12, 3, key="cl_k")
            if st.button("Rodar KMeans"):
                if len(feats) < 1:
                    st.error("Selecione ao menos 1 feature.")
                else:
                    X = df[feats].dropna()
                    if X.shape[0] < k:
                        st.error(f"Amostras insuficientes: {X.shape[0]} < k={k}")
                    else:
                        try:
                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(X)
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            labels = kmeans.fit_predict(Xs)
                            # safe PCA dims
                            n_samples, n_features = Xs.shape
                            comps = min(2, n_samples, n_features)
                            if comps < 1:
                                st.error("PCA nao aplicavel")
                            else:
                                pca = PCA(n_components=comps)
                                pcs = pca.fit_transform(Xs)
                                if comps == 1:
                                    fig = px.scatter(x=pcs[:,0], y=np.zeros(len(pcs)), color=labels.astype(str), title="Clusters (1D PCA)")
                                else:
                                    fig = px.scatter(x=pcs[:,0], y=pcs[:,1], color=labels.astype(str), title="Clusters (PCA 2D)")
                                st.plotly_chart(fig, use_container_width=True)
                                # attach cluster - align by index by reindexing
                                df_clusters = X.copy()
                                df_clusters['cluster'] = labels
                                st.session_state['df_main'] = df.merge(df_clusters['cluster'], left_index=True, right_index=True, how='left')
                                st.success("Clusterizacao aplicada.")
                        except Exception as e:
                            st.error(e)

    # SQL Lab
    elif menu == "SQL Lab":
        st.header("SQL Lab")
        st.info("Tabela 'dados' disponivel; use duckdb se presente (melhor performance). Ex: SELECT count(*) FROM dados")
        q = st.text_area("SQL", value="SELECT * FROM dados LIMIT 100", height=180)
        if st.button("Executar SQL"):
            res, err = run_sql(df, q)
            if err:
                st.error(err)
            else:
                st.success(f"{len(res)} linhas retornadas")
                safe_display_dataframe(res.head(1000))
                st.download_button("Download CSV", res.to_csv(index=False).encode('utf-8'), "sql_result.csv", "text/csv")
    
    # Export
    elif menu == "Exportar":
        st.header("Exportar & Relatorios")
        st.download_button("Download CSV (tratado)", df.to_csv(index=False).encode('utf-8'), "dados_tratados.csv", "text/csv")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
            desc = df.describe(include='all').transpose()
            desc.to_excel(writer, sheet_name='Resumo')
        st.download_button("Download Excel", buf.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        kpis = {'rows':len(df),'cols':df.shape[1],'nulls':int(df.isna().sum().sum()),'dups':int(df.duplicated().sum())}
        if st.button("Gerar PDF Executivo com graficos do relatorio"):
            try:
                pdf_bytes = generate_pdf_with_charts(df, st.session_state.get('report_charts',[]), kpis)
                st.download_button("Baixar PDF", pdf_bytes, "report.pdf", "application/pdf")
                st.success("PDF gerado.")
            except Exception as e:
                st.error(e)

    # persist main df
    st.session_state['df_main'] = st.session_state.get('df_main', df)

if __name__ == '__main__':
    main()