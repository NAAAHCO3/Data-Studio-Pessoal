"""
Enterprise Analytics Ultra — Single-file Streamlit App (Complete)
Features:
- EDA / Data Quality / Column Explorer with auto-insights
- No-code Data Studio (create columns, conditional, date extract, split, pivot/unpivot, merge)
- Visual Studio (Plotly charts, presets, add-to-report)
- Dashboard / Report builder + PDF export (with chart images if kaleido available)
- SQL Lab (duckdb preferred, sqlite fallback)
- AutoML (basic pipeline + feature selection + model export)
- Time Series Auto-ARIMA (light grid search)
- NLP: TF-IDF top tokens
- Clustering & Anomaly detection
Notes:
- Designed for Streamlit 1.51.0 compatibility (uses width="stretch" instead of deprecated use_container_width)
"""

import os
import io
import re
import math
import json
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Optional high-perf libs
try:
    import duckdb
    _HAS_DUCKDB = True
except Exception:
    import sqlite3
    _HAS_DUCKDB = False

# ML / Stats
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

# Statsmodels (time series)
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enterprise-ultra")

# Streamlit page config
st.set_page_config(page_title="Enterprise Analytics Ultra", layout="wide", page_icon="💠", initial_sidebar_state="expanded")

# ---------------------------
# Utilities
# ---------------------------
def try_read_csv(file_obj, encodings=("utf-8", "latin1", "cp1252", "iso-8859-1")) -> pd.DataFrame:
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

def safe_read(file) -> pd.DataFrame:
    """
    Read Streamlit UploadedFile or local path robustly (CSV/Excel).
    """
    if file is None:
        raise ValueError("No file provided")
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
                file.seek(0)
                df = pd.read_excel(file, engine="openpyxl")
        else:
            path = str(file)
            if path.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_csv(path, engine="python", encoding="latin1")
            else:
                df = pd.read_excel(path, engine="openpyxl")
        # basic column cleaning
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
    Display dataframe robustly: attempt normal st.dataframe with width="stretch",
    fallback to string-casted version if Arrow/pyarrow conversion fails.
    """
    try:
        st.dataframe(df, width="stretch", height=height)
    except Exception as e:
        logger.warning("st.dataframe Arrow conversion failed; coercing columns to safe types.")
        df2 = df.copy()
        for c in df2.columns:
            # If column has mixed numeric & string -> cast to str
            if df2[c].dtype.kind in ("i","u","f") and df2[c].apply(lambda x: isinstance(x, str)).any():
                df2[c] = df2[c].astype(str)
            if df2[c].dtype == object:
                # cast complex types to str
                if df2[c].apply(lambda x: isinstance(x, (list, dict))).any():
                    df2[c] = df2[c].astype(str)
        st.dataframe(df2.astype(str), width="stretch", height=height)

# ---------------------------
# PDF Report utils
# ---------------------------
def fig_to_image_bytes(fig: go.Figure, fmt='png', scale=2) -> Optional[bytes]:
    try:
        # Plotly's to_image requires kaleido installed. If not available, this will raise.
        return fig.to_image(format=fmt, scale=scale)
    except Exception as e:
        logger.debug(f"fig_to_image_bytes failed: {e}")
        return None

def generate_pdf_report(df: pd.DataFrame, charts: List[Dict[str,Any]], kpis: Dict[str,Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Executivo - Enterprise Analytics Ultra", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # KPIs banner
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,40,190,28,'F')
    pdf.set_y(44)
    colw = 190/4
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    vals = [kpis.get('rows',''), kpis.get('cols',''), kpis.get('nulls',''), kpis.get('dups','')]
    pdf.set_font("Helvetica","B",11)
    for i,t in enumerate(titles):
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

    # Charts pages (embed image if possible)
    for ch in charts:
        pdf.add_page()
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,ch.get('title','(sem titulo)'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        img = fig_to_image_bytes(ch.get('fig'))
        if img:
            tmp = f"/tmp/ea_chart_{abs(hash(ch.get('title','')))}.png"
            try:
                with open(tmp, "wb") as f:
                    f.write(img)
                pdf.image(tmp, x=15, y=None, w=180)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            except Exception:
                pdf.multi_cell(0,6,"(Imagem do gráfico indisponível)")
        else:
            pdf.multi_cell(0,6, f"Tipo: {ch.get('type','')}. Nota: {ch.get('note','')}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------
# Insights Engine
# ---------------------------
class InsightsEngine:
    def generate_column_insights(self, df: pd.DataFrame, col: str) -> List[str]:
        insights = []
        if col not in df.columns:
            return ["Coluna não encontrada."]
        s = df[col]
        null_pct = s.isna().mean()
        if null_pct > 0.05:
            insights.append(f"⚠️ Dados faltantes: {null_pct:.1%} nulos ( > 5% )")

        if pd.api.types.is_numeric_dtype(s):
            skew = s.dropna().skew()
            if abs(skew) > 1:
                insights.append(f"📈 Assimetria (skew): {skew:.2f}. Considere transformações.")
            # outliers
            z = np.abs(stats.zscore(s.dropna()))
            if len(z) and (z > 3).sum() > 0:
                insights.append(f"🚨 Outliers (Z>3): {(z>3).sum()} valores detectados.")
            # strong correlations
            num = df.select_dtypes(include=np.number)
            if num.shape[1] > 1:
                corrs = num.corr()[col].drop(labels=col, errors='ignore')
                strong = corrs[abs(corrs) > 0.85]
                for c_name, c_val in strong.items():
                    insights.append(f"🔗 Correlação forte com {c_name}: {c_val:.2f}")
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique > 50:
                insights.append(f"🔢 Alta cardinalidade: {n_unique} categorias.")
            if n_unique == 1:
                insights.append("🛑 Variância zero: coluna constante.")
            top = s.value_counts(normalize=True, dropna=True).head(1)
            if not top.empty and top.iloc[0] > 0.8:
                val = top.index[0]
                insights.append(f"⚖️ Desbalanceamento: '{val}' compõe {top.iloc[0]:.1%} dos registros.")
        return insights

# ---------------------------
# AutoML Engine (light)
# ---------------------------
class AutoMLEngine:
    def train(self, df: pd.DataFrame, target: str, features: List[str], algo: str, use_fs: bool=False):
        X = df[features].copy()
        y = df[target].copy()
        # drop rows with missing target
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        # define numeric & categorical
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        preproc = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ], remainder='drop')

        fs_step = []
        if use_fs and len(features) > 1:
            # choose k
            k = min(15, max(1, len(features)//2))
            score_func = f_regression if pd.api.types.is_numeric_dtype(y) else f_classif
            fs_step = [('fs', SelectKBest(score_func=score_func, k=k))]

        models = {
            "Random Forest (reg)": RandomForestRegressor(n_jobs=-1),
            "Random Forest (clf)": RandomForestClassifier(n_jobs=-1)
        }

        is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
        if is_reg:
            model = RandomForestRegressor(n_jobs=-1) if "Random" in algo else RandomForestRegressor(n_jobs=-1)
        else:
            model = RandomForestClassifier(n_jobs=-1)

        steps = [('pre', preproc)] + fs_step + [('model', model)]
        pipeline = Pipeline(steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        imp = None
        try:
            r = permutation_importance(pipeline, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
            imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': r.importances_mean}).sort_values('Importance', ascending=False)
        except Exception:
            imp = None

        metrics = {}
        if is_reg:
            metrics['r2'] = r2_score(y_test, preds)
            metrics['mae'] = mean_absolute_error(y_test, preds)
        else:
            metrics['acc'] = accuracy_score(y_test, preds)
            metrics['confusion'] = confusion_matrix(y_test, preds)

        return pipeline, X_test, y_test, preds, imp, metrics

# ---------------------------
# Time Series Engine (light ARIMA grid)
# ---------------------------
class TimeSeriesEngine:
    def auto_forecast(self, df: pd.DataFrame, date_col: str, val_col: str, horizon: int = 30):
        ts = df[[date_col, val_col]].dropna().copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce')
        ts = ts.dropna(subset=[date_col])
        ts = ts.set_index(date_col).sort_index()[val_col]

        if ts.empty:
            return None, None, None, None

        # attempt to infer freq, else set daily
        freq = pd.infer_freq(ts.index)
        try:
            ts = ts.asfreq(freq or 'D')
        except Exception:
            ts = ts.asfreq('D')

        ts = ts.fillna(method='ffill')

        best_aic = float('inf')
        best_model = None
        best_order = (1, 0, 1)
        # small grid
        for p in [0, 1, 2]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        model = ARIMA(ts, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_model = model
                            best_order = (p, d, q)
                    except Exception:
                        continue
        if best_model is None:
            return None, None, None, None
        forecast_res = best_model.get_forecast(steps=horizon)
        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
        return ts, pred_mean, conf_int, best_order

# ---------------------------
# SQL Lab util
# ---------------------------
def run_sql(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        if _HAS_DUCKDB:
            con = duckdb.connect(database=':memory:')
            con.register('dados', df)
            con.register('df', df)
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

# ---------------------------
# Simple sentiment (light lexicon)
# ---------------------------
_POS = {"good","great","excellent","bom","otimo","ótimo","positivo","recomendo","satisfeito","love","adorei"}
_NEG = {"bad","terrible","ruim","pessimo","péssimo","frustrado","odio","detestei","dinheiro","devolvi","nao gostei","não gostei"}
def simple_sentiment(text: str) -> Tuple[str,float]:
    if not isinstance(text, str) or text.strip()=="":
        return "neutral", 0.0
    t = text.lower()
    pos = sum(t.count(w) for w in _POS)
    neg = sum(t.count(w) for w in _NEG)
    denom = max(1, pos+neg)
    score = (pos - neg) / denom
    if score > 0.1: return "positive", score
    if score < -0.1: return "negative", score
    return "neutral", score

# ---------------------------
# UI Styling & session init
# ---------------------------
st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:2rem}
.metric-card {background:#0b1020;padding:14px;border-left:6px solid #00CC96;border-radius:8px;margin-bottom:8px;color:#fff}
.metric-card h3{margin:0;font-size:12px;color:#9aa7b1}
.metric-card h2{margin:4px 0 0 0;font-size:20px}
.insight-box{background:rgba(0,204,150,0.05);border:1px solid #00CC96;padding:10px;border-radius:6px}
</style>
""", unsafe_allow_html=True)

for key in ['df_raw','df_work','report_charts','sql_history','presets','last_fig','last_meta']:
    if key not in st.session_state:
        if key == 'report_charts':
            st.session_state[key] = []
        elif key == 'sql_history':
            st.session_state[key] = []
        elif key == 'presets':
            st.session_state[key] = {}
        elif key == 'last_fig':
            st.session_state[key] = None
        elif key == 'last_meta':
            st.session_state[key] = {}
        else:
            st.session_state[key] = pd.DataFrame()

# ---------------------------
# Main App
# ---------------------------
def main():
    st.sidebar.title("Enterprise Analytics Ultra")
    uploaded = st.sidebar.file_uploader("Carregar CSV / Excel", type=['csv','xlsx'])
    use_local = st.sidebar.checkbox("Usar arquivo local (DEV)", value=False)
    if st.sidebar.button("Resetar sessão"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # Load data
    if uploaded is None and use_local:
        # Uses the UPLOADED_FILE_PATH defined in the prompt or environment
        local_path = os.environ.get("DEV_LOCAL_PATH", "/mnt/data/uploaded_dataset.csv")
        try:
            df = safe_read(local_path)
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy()
            st.success("Dados carregados localmente.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar local: {e}")
    elif uploaded is not None:
        try:
            df = safe_read(uploaded)
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy()
            st.success(f"Arquivo '{getattr(uploaded,'name','uploaded')}' carregado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler arquivo: {e}")

    if st.session_state['df_work'].empty:
        st.title("Enterprise Analytics Ultra")
        st.markdown("""
        Ferramenta completa para analistas e cientistas de dados.
        Carregue um CSV/XLSX para começar.

        Fluxo sugerido:
        1. **Data Studio** -> preparar dados
        2. **Dashboard/Explorer** -> entender variáveis
        3. **Visual Studio** -> montar gráficos e adicionar ao relatório
        4. **Export** -> gerar relatórios e PDF
        """)
        return

    df_work = st.session_state['df_work']

    menu = st.sidebar.radio("Menu", [
        "Dashboard & Explorer",
        "Data Studio",
        "Visual Studio",
        "Relatório/Dashboard",
        "SQL Lab",
        "AutoML",
        "Time Series",
        "NLP",
        "Anomalias & Clustering",
        "Export"
    ])

    # ---------- Dashboard & Explorer ----------
    if menu == "Dashboard & Explorer":
        st.title("Visão Executiva & Column Explorer")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Registros</h3><h2>{format_number(len(df_work))}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Variáveis</h3><h2>{df_work.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{format_number(int(df_work.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{format_number(int(df_work.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Column Explorer")
        col = st.selectbox("Selecionar Coluna", df_work.columns.tolist())
        ie = InsightsEngine()
        insights = ie.generate_column_insights(df_work, col)
        if insights:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Auto Insights**")
            for it in insights:
                st.markdown(f"- {it}")
            st.markdown('</div>', unsafe_allow_html=True)

        left, right = st.columns([2,1])
        with left:
            if pd.api.types.is_numeric_dtype(df_work[col]):
                fig = px.histogram(df_work, x=col, marginal="box", title=f"Distribuição: {col}", template="plotly_dark")
                st.plotly_chart(fig, width="stretch")
                try:
                    qq = stats.probplot(df_work[col].dropna(), dist='norm')
                    qq_fig = px.scatter(x=qq[0][0], y=qq[0][1], title="QQ-Plot", template="plotly_dark", labels={'x':'Teórico','y':'Real'})
                    st.plotly_chart(qq_fig, width="stretch")
                except Exception:
                    pass
            else:
                top = df_work[col].value_counts().head(20).reset_index()
                top.columns = [col, 'count']
                fig = px.bar(top, x='count', y=col, orientation='h', title=f"Top categories: {col}", template="plotly_dark")
                st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("**Resumo Estatístico**")
            try:
                summary = df_work[col].describe()
                safe_display_dataframe(pd.DataFrame(summary), height=250)
            except Exception:
                st.write("Resumo indisponível.")

    # ---------- Data Studio ----------
    elif menu == "Data Studio":
        st.title("Data Studio — ETL No-Code")
        tabs = st.tabs(["Criar Coluna", "Renomear/Converter", "Pivot/Merge", "Filtros & Limpeza"])
        # Create Column
        with tabs[0]:
            st.header("Criar Coluna")
            op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
            if op == "Aritmética":
                num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
                if not num_cols:
                    st.info("Nenhuma coluna numérica disponível.")
                else:
                    a = st.selectbox("Coluna A", num_cols, key="ds_a")
                    mode = st.radio("Coluna B ou valor", ["Coluna", "Valor Fixo"], key="ds_mode")
                    b = None
                    bval = None
                    if mode == "Coluna":
                        b = st.selectbox("Coluna B", num_cols, key="ds_b")
                    else:
                        bval = st.number_input("Valor Fixo", value=1.0, key="ds_bval")
                    op_sym = st.selectbox("Operador", ["+","-","*","/"], key="ds_op")
                    new = st.text_input("Nome nova coluna", f"{a}{op_sym}res", key="ds_new")
                    if st.button("Criar Coluna"):
                        try:
                            dfn = df_work.copy()
                            if b is not None:
                                if op_sym == "+": dfn[new] = dfn[a] + dfn[b]
                                elif op_sym == "-": dfn[new] = dfn[a] - dfn[b]
                                elif op_sym == "*": dfn[new] = dfn[a] * dfn[b]
                                elif op_sym == "/": dfn[new] = dfn[a] / dfn[b].replace(0, np.nan)
                            else:
                                if op_sym == "+": dfn[new] = dfn[a] + bval
                                elif op_sym == "-": dfn[new] = dfn[a] - bval
                                elif op_sym == "*": dfn[new] = dfn[a] * bval
                                elif op_sym == "/": dfn[new] = dfn[a] / bval if bval != 0 else np.nan
                            st.session_state['df_work'] = dfn
                            st.success("Coluna criada.")
                        except Exception as e:
                            st.error(e)
            elif op == "Condicional (IF)":
                num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    target = st.selectbox("Coluna alvo", num_cols, key="if_target")
                    operator = st.selectbox("Operador", [">","<",">=","<=","==","!="], key="if_op")
                    thr = st.number_input("Threshold", value=0.0, key="if_thr")
                    tlabel = st.text_input("Rótulo True", "ALTO", key="if_t")
                    flabel = st.text_input("Rótulo False", "BAIXO", key="if_f")
                    newnm = st.text_input("Nome nova coluna", f"{target}_cat", key="if_new")
                    if st.button("Criar IF"):
                        ops = {
                            ">": df_work[target] > thr,
                            "<": df_work[target] < thr,
                            ">=": df_work[target] >= thr,
                            "<=": df_work[target] <= thr,
                            "==": df_work[target] == thr,
                            "!=": df_work[target] != thr
                        }
                        mask = ops.get(operator, df_work[target] > thr)
                        dfn = df_work.copy()
                        dfn[newnm] = np.where(mask, tlabel, flabel)
                        st.session_state['df_work'] = dfn
                        st.success("Condicional criada.")
                else:
                    st.info("Sem colunas numéricas para condicional.")
            elif op == "Extrair Data":
                date_cols = [c for c in df_work.columns if np.issubdtype(df_work[c].dtype, np.datetime64)]
                if not date_cols:
                    st.warning("Converta colunas para datetime primeiro (Renomear/Converter).")
                else:
                    dc = st.selectbox("Coluna de data", date_cols, key="ex_date_col")
                    comp = st.selectbox("Componente", ["year","month","day","weekday","quarter"], key="ex_comp")
                    newn = st.text_input("Nome nova coluna", f"{dc}_{comp}", key="ex_new")
                    if st.button("Extrair componente"):
                        dfn = df_work.copy()
                        if comp == "year": dfn[newn] = dfn[dc].dt.year
                        elif comp == "month": dfn[newn] = dfn[dc].dt.month
                        elif comp == "day": dfn[newn] = dfn[dc].dt.day
                        elif comp == "weekday": dfn[newn] = dfn[dc].dt.day_name()
                        elif comp == "quarter": dfn[newn] = dfn[dc].dt.quarter
                        st.session_state['df_work'] = dfn
                        st.success("Componente extraído.")
            elif op == "Split Texto":
                text_cols = df_work.select_dtypes(include=['object','string']).columns.tolist()
                if not text_cols:
                    st.info("Nenhuma coluna de texto disponível.")
                else:
                    tc = st.selectbox("Texto Col", text_cols, key="sp_col")
                    sep = st.text_input("Separador", " ", key="sp_sep")
                    idx = st.number_input("Índice (0..)", min_value=0, value=0, key="sp_idx")
                    newn = st.text_input("Nome nova coluna", f"{tc}_part{idx}", key="sp_new")
                    if st.button("Aplicar Split"):
                        try:
                            s = df_work[tc].astype(str).str.split(sep, expand=True)
                            if idx < s.shape[1]:
                                dfn = df_work.copy()
                                dfn[newn] = s[idx]
                                st.session_state['df_work'] = dfn
                                st.success("Split aplicado.")
                            else:
                                st.error("Índice fora do range do split.")
                        except Exception as e:
                            st.error(e)

        # Rename / Convert
        with tabs[1]:
            st.header("Renomear / Converter")
            c1, c2 = st.columns(2)
            with c1:
                colr = st.selectbox("Coluna para renomear", df_work.columns.tolist(), key="ren_col")
                newn = st.text_input("Novo nome", value=colr, key="ren_new")
                if st.button("Aplicar Renome"):
                    dfn = df_work.copy()
                    dfn.rename(columns={colr:newn}, inplace=True)
                    st.session_state['df_work'] = dfn
                    st.success("Renomeado.")
            with c2:
                colc = st.selectbox("Coluna para converter", df_work.columns.tolist(), key="conv_col")
                to = st.selectbox("Converter para", ["Data","Número","Texto"], key="conv_to")
                if st.button("Converter Tipo"):
                    try:
                        dfn = df_work.copy()
                        if to == "Data":
                            dfn[colc] = pd.to_datetime(dfn[colc], errors='coerce')
                        elif to == "Número":
                            dfn[colc] = pd.to_numeric(dfn[colc], errors='coerce')
                        else:
                            dfn[colc] = dfn[colc].astype(str)
                        st.session_state['df_work'] = dfn
                        st.success("Conversão aplicada.")
                    except Exception as e:
                        st.error(e)

        # Pivot / Merge
        with tabs[2]:
            st.header("Pivot / Unpivot / Merge")
            mode = st.radio("Modo", ["Pivot","Unpivot","Merge"], key="pm_mode")
            if mode == "Pivot":
                idx = st.multiselect("Index cols", df_work.columns.tolist(), key="pv_idx")
                colp = st.selectbox("Columns", df_work.columns.tolist(), key="pv_col")
                val = st.selectbox("Values (num)", df_work.select_dtypes(include=np.number).columns.tolist(), key="pv_val")
                agg = st.selectbox("Agg", ["sum","mean","count"], key="pv_agg")
                if st.button("Pivotar"):
                    try:
                        res = df_work.pivot_table(index=idx, columns=colp, values=val, aggfunc=agg).reset_index()
                        # flatten columns
                        res.columns = [("_".join(map(str,c)) if isinstance(c, tuple) else str(c)).strip() for c in res.columns]
                        st.session_state['df_work'] = res
                        st.success("Pivot realizado.")
                    except Exception as e:
                        st.error(e)
            elif mode == "Unpivot":
                ids = st.multiselect("Id vars", df_work.columns.tolist(), key="un_id")
                vals = st.multiselect("Value vars", df_work.columns.tolist(), key="un_vals")
                if st.button("Unpivot"):
                    try:
                        res = df_work.melt(id_vars=ids, value_vars=vals, var_name='variable', value_name='value')
                        st.session_state['df_work'] = res
                        st.success("Unpivot realizado.")
                    except Exception as e:
                        st.error(e)
            else:
                uf = st.file_uploader("Arquivo para mesclar", type=['csv','xlsx'], key="merge_file")
                if uf:
                    try:
                        other = safe_read(uf)
                        left_keys = st.multiselect("Chaves (este DF)", df_work.columns.tolist(), key="merge_left")
                        right_keys = st.multiselect("Chaves (arquivo)", other.columns.tolist(), key="merge_right")
                        how = st.selectbox("Tipo de merge", ["left","inner","right","outer"], key="merge_how")
                        if st.button("Executar Merge"):
                            if left_keys and right_keys and len(left_keys)==len(right_keys):
                                res = pd.merge(df_work, other, left_on=left_keys, right_on=right_keys, how=how)
                                st.session_state['df_work'] = res
                                st.success("Merge executado.")
                            else:
                                st.error("As chaves devem ser selecionadas e ter mesmo comprimento.")
                    except Exception as e:
                        st.error(e)

        # Filters & cleaning
        with tabs[3]:
            st.header("Filtros & Limpeza")
            colf = st.selectbox("Coluna", df_work.columns.tolist(), key="filter_col")
            if pd.api.types.is_numeric_dtype(df_work[colf]):
                mn, mx = float(df_work[colf].min()), float(df_work[colf].max())
                r = st.slider("Intervalo", mn, mx, (mn, mx), key="filter_range")
                if st.button("Aplicar filtro"):
                    st.session_state['df_work'] = df_work[(df_work[colf] >= r[0]) & (df_work[colf] <= r[1])]
                    st.success("Filtro aplicado.")
            elif np.issubdtype(df_work[colf].dtype, np.datetime64):
                min_d, max_d = df_work[colf].min().date(), df_work[colf].max().date()
                dr = st.date_input("Período", [min_d, max_d], key="filter_date")
                if st.button("Aplicar filtro (data)"):
                    st.session_state['df_work'] = df_work[(df_work[colf].dt.date >= dr[0]) & (df_work[colf].dt.date <= dr[1])]
                    st.success("Filtro aplicado.")
            else:
                vals = df_work[colf].dropna().unique().tolist()
                sel = st.multiselect("Valores", vals[:500], default=vals[:min(10,len(vals))], key="filter_vals")
                if st.button("Aplicar filtro (cat)"):
                    st.session_state['df_work'] = df_work[df_work[colf].isin(sel)]
                    st.success("Filtro aplicado.")
            if st.button("Remover linhas com NA"):
                before = len(df_work)
                st.session_state['df_work'] = df_work.dropna()
                after = len(st.session_state['df_work'])
                st.success(f"Removidas {before-after} linhas.")

    # ---------- Visual Studio ----------
    elif menu == "Visual Studio":
        st.title("Visual Studio — Criador de Gráficos")
        left, right = st.columns([1,2])
        with left:
            chart_type = st.selectbox("Tipo", ["Barras","Linha","Dispersão","Pizza","Histograma","Box","Heatmap","Treemap","Sunburst"])
            x = st.selectbox("X", df_work.columns.tolist(), key="vs_x")
            y = None
            if chart_type not in ("Pizza","Histograma","Heatmap","Treemap","Sunburst"):
                y = st.selectbox("Y (valor)", df_work.select_dtypes(include=np.number).columns.tolist(), key="vs_y") if not df_work.select_dtypes(include=np.number).empty else None
            color = st.selectbox("Cor (opcional)", ["Nenhum"] + df_work.columns.tolist(), key="vs_color")
            color = None if color == "Nenhum" else color
            agg = None
            if chart_type in ("Barras","Linha","Treemap","Sunburst"):
                use_agg = st.checkbox("Agrupar e agregar", value=True, key="vs_agg")
                if use_agg:
                    agg = st.selectbox("Função", ["sum","mean","count","min","max"], key="vs_agg_fun")
            theme = st.selectbox("Tema", ["plotly","plotly_dark","ggplot2","seaborn"], key="vs_theme")
            height = st.slider("Altura", 300, 900, 500, key="vs_height")
            show_labels = st.checkbox("Mostrar rótulos", value=False, key="vs_labels")
            size_col = None
            if chart_type == "Dispersão":
                size_col = st.selectbox("Tamanho (opcional)", ["Nenhum"] + df_work.select_dtypes(include=np.number).columns.tolist(), key="vs_size")
                size_col = None if size_col == "Nenhum" else size_col
            title = st.text_input("Título", f"{chart_type} {y or ''} por {x}", key="vs_title")
            if st.button("Gerar gráfico"):
                try:
                    plot_df = df_work.copy()
                    if agg and x and y and chart_type in ("Barras","Linha","Treemap","Sunburst"):
                        if color:
                            plot_df = plot_df.groupby([x, color], dropna=False)[y].agg(agg).reset_index()
                        else:
                            plot_df = plot_df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                    fig = go.Figure()
                    if chart_type == "Barras":
                        fig = px.bar(plot_df, x=x, y=y, color=color, text_auto=show_labels, template=theme)
                    elif chart_type == "Linha":
                        fig = px.line(plot_df, x=x, y=y, color=color, markers=True, template=theme)
                    elif chart_type == "Dispersão":
                        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size_col, template=theme)
                    elif chart_type == "Pizza":
                        fig = px.pie(plot_df, names=x, values=y, template=theme)
                    elif chart_type == "Histograma":
                        fig = px.histogram(plot_df, x=x, color=color, nbins=30, template=theme, text_auto=show_labels)
                    elif chart_type == "Box":
                        fig = px.box(plot_df, x=x, y=y, color=color, template=theme)
                    elif chart_type == "Heatmap":
                        corr = df_work.select_dtypes(include=np.number).corr()
                        fig = px.imshow(corr, text_auto=True, template=theme, title="Matriz de Correlação")
                    elif chart_type == "Treemap":
                        fig = px.treemap(plot_df, path=[x, color] if color else [x], values=y, template=theme)
                    elif chart_type == "Sunburst":
                        fig = px.sunburst(plot_df, path=[x, color] if color else [x], values=y, template=theme)
                    fig.update_layout(height=height, title=title)
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {"title": title, "type": chart_type}
                    st.success("Gráfico gerado.")
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")
        with right:
            if st.session_state.get('last_fig') is not None:
                st.plotly_chart(st.session_state['last_fig'], width="stretch")
                note = st.text_area("Nota para o relatório", key="vs_note")
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("➕ Adicionar ao Relatório"):
                        st.session_state['report_charts'].append({
                            "fig": st.session_state['last_fig'],
                            "title": st.session_state['last_meta'].get('title','(sem título)'),
                            "type": st.session_state['last_meta'].get('type',''),
                            "note": note
                        })
                        st.success("Adicionado ao relatório.")
                with c2:
                    if st.button("Salvar Preset"):
                        presets = st.session_state.get('presets', {})
                        pname = f"preset_{len(presets)+1}_{datetime.now().strftime('%H%M%S')}"
                        presets[pname] = {"meta": st.session_state['last_meta'], "note": note}
                        st.session_state['presets'] = presets
                        st.success(f"Preset salvo: {pname}")
            else:
                st.info("Gere um gráfico no painel esquerdo para visualizar aqui.")

    # ---------- Report / Dashboard ----------
    elif menu == "Relatório/Dashboard":
        st.title("Relatório / Dashboard Builder")
        charts = st.session_state.get('report_charts', [])
        c_left, c_right = st.columns([3,1])
        with c_right:
            pres_name = st.text_input("Nome do preset (salvar relatório)", key="rb_preset_name")
            if st.button("Salvar relatório como preset"):
                name = pres_name or f"report_{len(st.session_state['presets'])+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state['presets'][name] = charts.copy()
                st.success(f"Relatório salvo: {name}")
            if st.session_state['presets']:
                sel = st.selectbox("Abrir preset salvo", ["Nenhum"] + list(st.session_state['presets'].keys()), key="rb_open_preset")
                if sel and sel != "Nenhum":
                    st.session_state['report_charts'] = st.session_state['presets'][sel].copy()
                    st.success(f"Abrindo preset: {sel}")
            if st.button("Limpar relatório"):
                st.session_state['report_charts'] = []
                st.success("Relatório limpo.")
        if not charts:
            st.info("Relatório vazio — adicione gráficos no Visual Studio.")
        else:
            for i, c in enumerate(charts):
                st.markdown("---")
                c1, c2 = st.columns([3,1])
                with c1:
                    try:
                        st.plotly_chart(c['fig'], width="stretch")
                    except Exception:
                        st.write(c.get('title','(sem título)'))
                with c2:
                    st.write(f"**{c.get('title','(sem título)')}**")
                    st.write(f"Tipo: {c.get('type','')}")
                    if c.get('note'):
                        st.info(c.get('note'))
                    if st.button(f"Remover #{i+1}", key=f"rm_{i}"):
                        st.session_state['report_charts'].pop(i)
                        st.experimental_rerun()

    # ---------- SQL Lab ----------
    elif menu == "SQL Lab":
        st.title("SQL Lab (duckdb preferred)")
        query = st.text_area("SQL (tabela 'dados' ou 'df')", value="SELECT * FROM df LIMIT 100", height=180)
        if st.button("Executar query"):
            res, err = run_sql(df_work, query)
            if err:
                st.error(err)
            else:
                st.session_state['sql_history'].append(query)
                st.success(f"{len(res)} linhas retornadas")
                safe_display_dataframe(res.head(1000))
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar resultado (CSV)", csv, "sql_result.csv", "text/csv")
        if st.session_state.get('sql_history'):
            with st.expander("Histórico (últimas 10)"):
                for q in reversed(st.session_state['sql_history'][-10:]):
                    st.code(q, language='sql')

    # ---------- AutoML ----------
    elif menu == "AutoML":
        st.title("AutoML (Light)")
        automl = AutoMLEngine()
        tgt = st.selectbox("Target", df_work.columns.tolist(), key="aml_target")
        features = st.multiselect("Features", [c for c in df_work.columns if c != tgt], default=[c for c in df_work.columns if c!=tgt][:6], key="aml_feats")
        use_fs = st.checkbox("SelectKBest (feature selection)", value=False, key="aml_fs")
        algo = st.selectbox("Algoritmo (RandomForest chosen)", ["RandomForest"], key="aml_algo")
        if st.button("Treinar modelo") and features:
            with st.spinner("Treinando..."):
                model, X_test, y_test, preds, imp, metrics = automl.train(df_work, tgt, features, algo, use_fs)
                st.success("Treinamento concluído.")
                if 'r2' in metrics:
                    st.metric("R2", f"{metrics['r2']:.4f}")
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                    fig = px.scatter(x=y_test, y=preds, labels={'x':'Real','y':'Previsto'}, title="Real vs Previsto", template="plotly_dark")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.metric("Acurácia", f"{metrics.get('acc',0):.2%}")
                    if metrics.get('confusion') is not None:
                        st.write("Matriz de Confusão")
                        st.dataframe(pd.DataFrame(metrics['confusion']), width="stretch")
                if imp is not None:
                    st.subheader("Importância das features (permutation)")
                    st.dataframe(imp.head(30), width="stretch")
                # model download
                try:
                    buf = io.BytesIO()
                    pickle.dump(model, buf)
                    buf.seek(0)
                    st.download_button("Baixar modelo (pickle)", buf.getvalue(), "modelo.pkl", "application/octet-stream")
                except Exception as e:
                    st.warning(f"Não foi possível serializar o modelo: {e}")

    # ---------- Time Series ----------
    elif menu == "Time Series":
        st.title("Time Series Forecast (Auto-ARIMA lite)")
        tse = TimeSeriesEngine()
        dt_cols = [c for c in df_work.columns if np.issubdtype(df_work[c].dtype, np.datetime64)]
        if not dt_cols:
            # try to infer a date column by name
            for c in df_work.columns:
                if re.search(r"(date|data|dt)", c, flags=re.I):
                    try:
                        df_work[c] = pd.to_datetime(df_work[c], errors='coerce')
                        dt_cols.append(c)
                    except Exception:
                        pass
        if not dt_cols:
            st.info("Nenhuma coluna de data detectada. Converta uma coluna para datetime no Data Studio.")
        else:
            cdt = st.selectbox("Coluna de data", dt_cols, key="ts_date")
            cval = st.selectbox("Coluna valor (numérica)", df_work.select_dtypes(include=np.number).columns.tolist(), key="ts_val")
            horizon = st.slider("Horizonte (periods)", 7, 90, 30)
            if st.button("Gerar Forecast"):
                with st.spinner("Ajustando ARIMA..."):
                    ts, pred, conf, order = tse.auto_forecast(df_work, cdt, cval, horizon)
                    if ts is None:
                        st.error("Não foi possível ajustar o modelo.")
                    else:
                        st.success(f"Melhor ordem ARIMA: {order}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Histórico"))
                        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Previsão", line=dict(color='red')))
                        fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,0], showlegend=False, line=dict(width=0)))
                        fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,1], name="Intervalo (conf)", fill='tonexty', fillcolor='rgba(255,0,0,0.15)', line=dict(width=0)))
                        st.plotly_chart(fig, width="stretch")

    # ---------- NLP ----------
    elif menu == "NLP":
        st.title("NLP / Texto (TF-IDF + Sentiment lexicon)")
        text_cols = df_work.select_dtypes(include=['object','string']).columns.tolist()
        if not text_cols:
            st.info("Nenhuma coluna de texto detectada.")
        else:
            tc = st.selectbox("Coluna de texto", text_cols)
            if st.button("Top tokens TF-IDF"):
                vec = TfidfVectorizer(max_features=40, stop_words='english')
                try:
                    X = vec.fit_transform(df_work[tc].astype(str).fillna(""))
                    sums = np.array(X.sum(axis=0)).ravel()
                    tokens = vec.get_feature_names_out()
                    ser = pd.Series(sums, index=tokens).sort_values(ascending=False)
                    st.bar_chart(ser)
                except Exception as e:
                    st.error(f"Erro TF-IDF: {e}")
            if st.button("Analisar sentimento (rápido)"):
                s = df_work[tc].fillna("").astype(str)
                res = s.apply(simple_sentiment)
                df_sent = pd.DataFrame(list(res), columns=['sent_label','sent_score'])
                out = pd.concat([df_work[[tc]].reset_index(drop=True), df_sent], axis=1)
                st.write(out['sent_label'].value_counts())
                safe_display_dataframe(out.head(500))

    # ---------- Anomalias & Clustering ----------
    elif menu == "Anomalias & Clustering":
        st.title("Anomalias & Clustering")
        numeric_cols = df_work.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("Sem colunas numéricas para análise.")
        else:
            st.subheader("Detecção de Anomalias")
            col_an = st.selectbox("Coluna (1D z-score)", numeric_cols, key="an_col")
            z_th = st.slider("Z-score threshold", 2.0, 6.0, 3.0, step=0.5)
            if st.button("Detectar (z-score)"):
                s = df_work[col_an].dropna()
                z = np.abs(stats.zscore(s))
                out = df_work.loc[s.index[z > z_th]]
                st.write(f"Anomalias detectadas: {len(out)}")
                safe_display_dataframe(out.head(200))
            if st.button("Detectar IsolationForest (multivariado)"):
                try:
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    arr = df_work[numeric_cols].dropna()
                    iso.fit(arr)
                    preds = iso.predict(arr)
                    out_iso = arr[preds == -1]
                    st.write(f"Anomalias IsolationForest: {len(out_iso)}")
                    safe_display_dataframe(out_iso.head(200))
                except Exception as e:
                    st.error(f"Erro IsolationForest: {e}")
            st.subheader("Clusterização (KMeans + PCA)")
            feats = st.multiselect("Features (numéricas)", numeric_cols, default=numeric_cols[:4])
            k = st.slider("K", 2, 12, 3)
            if st.button("Rodar KMeans") and feats:
                X = df_work[feats].dropna()
                if X.shape[0] < k:
                    st.error("Amostras insuficientes para k.")
                else:
                    Xs = StandardScaler().fit_transform(X)
                    km = KMeans(n_clusters=k, random_state=42)
                    labels = km.fit_predict(Xs)
                    comps = min(2, Xs.shape[1], Xs.shape[0])
                    if comps < 1:
                        st.error("Dimensões insuficientes para PCA.")
                    else:
                        pca = PCA(n_components=comps)
                        pcs = pca.fit_transform(Xs)
                        if comps == 1:
                            fig = px.scatter(x=pcs[:,0], y=np.zeros(len(pcs)), color=labels.astype(str), title="Clusters (1D PCA)", template="plotly_dark")
                        else:
                            fig = px.scatter(x=pcs[:,0], y=pcs[:,1], color=labels.astype(str), title="Clusters (2D PCA)", template="plotly_dark")
                        st.plotly_chart(fig, width="stretch")
                        # attach cluster labels to original df
                        df_cl = X.copy()
                        df_cl['cluster'] = labels
                        st.session_state['df_work'] = df_work.merge(df_cl[['cluster']], left_index=True, right_index=True, how='left')
                        st.success("Clusterization applied, column 'cluster' added.")

    # ---------- Export ----------
    elif menu == "Export":
        st.title("Export & Reports")
        df_export = st.session_state['df_work']
        c1, c2 = st.columns(2)
        with c1:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "dados_tratados.csv", "text/csv")
        with c2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Dados')
                desc = df_export.describe(include='all').transpose()
                desc.to_excel(writer, sheet_name='Resumo')
            st.download_button("Download Excel", buffer.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        kpis = {"rows": len(df_export), "cols": df_export.shape[1], "nulls": int(df_export.isna().sum().sum()), "dups": int(df_export.duplicated().sum())}
        if st.button("Gerar PDF Executivo (com gráficos do relatório se possível)"):
            charts = st.session_state.get('report_charts', [])
            try:
                pdf_bytes = generate_pdf_report(df_export, charts, kpis)
                st.download_button("Baixar PDF", pdf_bytes, "report.pdf", "application/pdf")
                st.success("PDF gerado.")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    # persist working df
    st.session_state['df_work'] = st.session_state.get('df_work', df_work)

if __name__ == "__main__":
    main()