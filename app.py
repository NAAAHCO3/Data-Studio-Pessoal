# app.py
"""
Enterprise Data Studio — Professional Single-File App
Objetivo: Ferramenta completa para analistas / cientistas de dados iniciantes/intermediários.
Inglês/Português: UX em Português (br).
Requisitos (recomendados):
 - streamlit, pandas, numpy, plotly, fpdf2, openpyxl, pyarrow, joblib, scikit-learn, statsmodels, kaleido (opcional), duckdb (opcional)
"""

import streamlit as st
st.set_page_config(page_title="Enterprise Data Studio", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# Core
import os
import io
import sys
import math
import time
import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

# Data + viz
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# PDF
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ML & stats
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance

# Statsmodels for ARIMA
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enterprise-data-studio")

# -----------------------
# Utility functions
# -----------------------
def try_read_csv(file_obj, encodings=("utf-8", "latin1", "cp1252", "iso-8859-1")) -> pd.DataFrame:
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

def safe_read(file) -> pd.DataFrame:
    """Lê CSV/Excel robustamente a partir de Streamlit UploadedFile ou path."""
    if file is None:
        raise ValueError("Nenhum arquivo fornecido.")
    if hasattr(file, "read"):
        name = getattr(file, "name", "")
        if name.lower().endswith(".csv"):
            return try_read_csv(file)
        else:
            file.seek(0)
            return pd.read_excel(file, engine="openpyxl")
    else:
        path = str(file)
        if path.lower().endswith(".csv"):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_csv(path, encoding="latin1", engine="python")
        else:
            return pd.read_excel(path, engine="openpyxl")

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return df

def format_number(n) -> str:
    try:
        n = float(n)
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000: return f"{n/1_000:.2f}K"
        if n.is_integer(): return f"{int(n)}"
        return f"{n:.2f}"
    except Exception:
        return str(n)

def safe_display_dataframe(df: pd.DataFrame, height: int = 400):
    """Exibe dataframe protegendo contra erros com Arrow (pyarrow)."""
    try:
        st.dataframe(df, height=height, width='stretch')
    except Exception:
        df2 = df.copy()
        for c in df2.columns:
            if df2[c].dtype == object:
                # coerce complex objects to str
                df2[c] = df2[c].apply(lambda x: str(x) if not pd.isna(x) else "")
        st.dataframe(df2, height=height, width='stretch')

# -----------------------
# Small NLP sentiment lexicon (PT/EN)
# -----------------------
_POS = {"bom","otimo","ótimo","excelente","excelent","satisfeito","positivo","love","adoro","adorei","recomendo","happy","good","great"}
_NEG = {"ruim","péssimo","péssima","pessimo","pessima","frustrado","frustrante","odio","odeio","hate","bad","terrible","devolvi","dinheiro_jogado_fora","nao_gostei","não_gostei"}

def lexicon_sentiment(text: str) -> Tuple[str, float]:
    if not isinstance(text, str) or text.strip()=="":
        return "neutral", 0.0
    s = text.lower()
    # simple token-based
    pos = sum(s.count(w) for w in _POS)
    neg = sum(s.count(w) for w in _NEG)
    denom = max(1, pos+neg)
    score = (pos-neg)/denom
    if score > 0.1: return "positive", score
    if score < -0.1: return "negative", score
    return "neutral", score

# -----------------------
# Plot helpers
# -----------------------
def build_chart(chart_type: str, df: pd.DataFrame, x: Optional[str]=None, y: Optional[str]=None, color: Optional[str]=None, size: Optional[str]=None, agg: Optional[str]=None, height: int=450, theme: str='plotly_white'):
    plot_df = df.copy()
    # Aggregate if needed
    if agg and x and y and chart_type in ("Barras","Linha","Treemap","Sunburst"):
        if color:
            plot_df = plot_df.groupby([x, color], dropna=False)[y].agg(agg).reset_index()
        else:
            plot_df = plot_df.groupby(x, dropna=False)[y].agg(agg).reset_index()
    fig = go.Figure()
    try:
        if chart_type == "Barras":
            fig = px.bar(plot_df, x=x, y=y, color=color, text_auto=True, template=theme)
        elif chart_type == "Linha":
            fig = px.line(plot_df, x=x, y=y, color=color, markers=True, template=theme)
        elif chart_type in ("Dispersão", "Scatter"):
            fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, template=theme)
        elif chart_type == "Pizza":
            fig = px.pie(plot_df, names=x, values=y, template=theme)
        elif chart_type == "Histograma":
            fig = px.histogram(plot_df, x=x, color=color, nbins=40, template=theme)
        elif chart_type == "Box":
            fig = px.box(plot_df, x=x, y=y, color=color, template=theme)
        elif chart_type == "Heatmap":
            corr = df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, template=theme)
        elif chart_type == "Treemap":
            path = [x, color] if color else [x]
            fig = px.treemap(plot_df, path=path, values=y, template=theme)
        else:
            fig = go.Figure()
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro ao criar gráfico: {e}", showarrow=False)
    fig.update_layout(height=height, title=f"{chart_type}: {y or ''} por {x or ''}")
    return fig

# -----------------------
# SQL Engine
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
# PDF export
# -----------------------
def fig_to_image_bytes(fig: go.Figure, fmt='png', scale=2) -> Optional[bytes]:
    try:
        return fig.to_image(format=fmt, scale=scale)
    except Exception as e:
        logger.warning(f"fig_to_image_bytes failed: {e}")
        return None

def generate_pdf_report(df: pd.DataFrame, charts: List[Dict[str,Any]], kpis: Dict[str,Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatório Executivo — Enterprise Data Studio", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # KPIs
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,40,190,28,'F')
    pdf.set_y(44)
    colw = 190/4
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    vals = [kpis.get('rows',''), kpis.get('cols',''), kpis.get('nulls',''), kpis.get('dups','')]
    pdf.set_font("Helvetica","B",11)
    for i,t in enumerate(titles):
        pdf.cell(colw,8,t,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica","",11)
    for i,v in enumerate(vals):
        pdf.cell(colw,8,str(v),align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(12)

    # Short stats
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Resumo Estatístico (Top variáveis numéricas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    desc = df.select_dtypes(include=np.number).describe().T.reset_index().head(8)
    if not desc.empty:
        cols = ['index','mean','min','max']
        if set(cols).issubset(desc.columns):
            desc = desc[cols]
            desc.columns = ['Variável','Média','Min','Max']
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

    # Charts list and attempts to embed images
    for ch in charts:
        pdf.add_page()
        pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,ch.get('title','(sem título)'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        img = None
        try:
            img = fig_to_image_bytes(ch['fig'], fmt='png', scale=2)
        except Exception:
            img = None
        if img:
            tmp_name = f"/tmp/plot_{abs(hash(ch.get('title','')))}.png"
            try:
                with open(tmp_name, "wb") as f:
                    f.write(img)
                pdf.image(tmp_name, x=15, y=None, w=180)
                try: os.remove(tmp_name)
                except: pass
            except Exception:
                pdf.multi_cell(0,6,"(não foi possível embutir a imagem do gráfico)")
        else:
            pdf.multi_cell(0,6,f"Tipo: {ch.get('type','')} - Nota: {ch.get('note','')}")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# -----------------------
# Auto insights engine (simplified)
# -----------------------
def generate_auto_insights(df: pd.DataFrame, col: str) -> List[str]:
    insights = []
    s = df[col]
    # missing
    missing_pct = s.isna().mean()
    if missing_pct > 0.05:
        insights.append(f"⚠️ {missing_pct:.1%} valores nulos — considere imputação ou remoção.")
    if pd.api.types.is_numeric_dtype(s):
        skew = s.dropna().skew()
        if abs(skew) > 1:
            insights.append(f"📈 Assimetria alta (skew={skew:.2f}) — considerar transformação (log/power).")
        z = np.abs(stats.zscore(s.dropna()))
        out = int((z > 3).sum())
        if out > 0:
            insights.append(f"🚨 {out} outliers (z>3) detectados.")
    else:
        nunique = s.nunique(dropna=True)
        if nunique == 1:
            insights.append("🛑 Coluna constante — sem variabilidade.")
        if nunique > 100:
            insights.append(f"🔢 Alta cardinalidade: {nunique} categorias.")
    return insights

# -----------------------
# Time series quick forecast (Auto ARIMA light)
# -----------------------
def auto_arima_forecast(df: pd.DataFrame, date_col: str, value_col: str, horizon: int = 30):
    try:
        ts = df[[date_col, value_col]].dropna()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.sort_values(date_col).set_index(date_col)
        # infer freq or fallback to daily
        freq = pd.infer_freq(ts.index)
        if freq is None:
            ts = ts.asfreq('D').fillna(method='ffill')
        else:
            ts = ts.asfreq(freq).fillna(method='ffill')
        best_aic = float('inf')
        best_model = None
        best_order = (1,0,0)
        # small grid
        for p in [0,1,2]:
            for d in [0,1]:
                for q in [0,1]:
                    try:
                        m = ARIMA(ts[value_col], order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_model = m
                            best_order = (p,d,q)
                    except Exception:
                        continue
        if best_model is None:
            return None, None, None
        forecast_res = best_model.get_forecast(steps=horizon)
        pred = forecast_res.predicted_mean
        conf = forecast_res.conf_int()
        return best_model, pred, conf
    except Exception as e:
        logger.exception("auto_arima_forecast failed")
        return None, None, None

# -----------------------
# AutoML simplified
# -----------------------
def simple_automl_train(df: pd.DataFrame, target: str, features: List[str], is_regression: bool = True, use_fs: bool = False):
    X = df[features].copy()
    y = df[target].copy()
    # Basic preprocessors
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)])), cat_cols)
    preproc = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    # Feature selection
    fs_steps = []
    if use_fs:
        k = min(30, max(1, len(features)//2))
        fs_steps = [('fs', SelectKBest(score_func=(f_regression if is_regression else f_classif), k=k))]
    # model
    if is_regression:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    steps = [('pre', preproc)] + fs_steps + [('model', model)]
    pipeline = Pipeline(steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = {}
    if is_regression:
        metrics['r2'] = r2_score(y_test, preds)
        metrics['mae'] = mean_absolute_error(y_test, preds)
    else:
        metrics['acc'] = accuracy_score(y_test, preds)
        # try confusion matrix
        try:
            metrics['confusion'] = confusion_matrix(y_test, preds)
        except Exception:
            metrics['confusion'] = None
    # feature importance via permutation
    try:
        pi = permutation_importance(pipeline, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
        feat_names = []
        # attempt to get feature names
        try:
            # handle ColumnTransformer name retrieval (sketch)
            feat_names = []
            if hasattr(pipeline.named_steps['pre'], 'get_feature_names_out'):
                feat_names = pipeline.named_steps['pre'].get_feature_names_out()
            else:
                # fallback: combine columns
                feat_names = X_test.columns.tolist()
        except Exception:
            feat_names = X_test.columns.tolist()
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': pi.importances_mean})
        imp_df = imp_df.sort_values('importance', ascending=False).head(30)
    except Exception:
        imp_df = None
    return pipeline, metrics, imp_df

# -----------------------
# UI styles
# -----------------------
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.metric-card {background:#f3f6fb;border-left:6px solid #4F8BF9;padding:12px;border-radius:6px;margin-bottom:8px}
@media (prefers-color-scheme: dark){
.metric-card{background:#262730;color:#ddd;border-left:6px solid #ffbd45}
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Session init
# -----------------------
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
if 'df_work' not in st.session_state:
    st.session_state['df_work'] = pd.DataFrame()
if 'report_charts' not in st.session_state:
    st.session_state['report_charts'] = []
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}
if 'last_fig' not in st.session_state:
    st.session_state['last_fig'] = None
if 'last_meta' not in st.session_state:
    st.session_state['last_meta'] = {}
if 'sql_history' not in st.session_state:
    st.session_state['sql_history'] = []

# -----------------------
# Sidebar - Load / Quick Settings
# -----------------------
def sidebar_area():
    st.sidebar.title("Enterprise Data Studio")
    uploaded = st.sidebar.file_uploader("Carregue CSV / XLSX", type=['csv','xlsx'])
    sample_mode = st.sidebar.checkbox("Usar amostra de desenvolvimento (dev)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.header("Execução & Utils")
    st.sidebar.markdown("Versão: 1.0 — Professional Single File")
    if st.sidebar.button("🔄 Resetar sessão (limpar tudo)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
    return uploaded, sample_mode

# -----------------------
# Main app
# -----------------------
def main():
    uploaded, sample = sidebar_area()
    st.title("Enterprise Data Studio — Ferramenta Completa para Analistas & Cientistas (Iniciantes → Pro)")
    st.caption("Fluxo: Conectar → Preparar → Explorar → Modelar → Exportar. Use os guias em cada seção.")

    # Load data logic
    if uploaded is not None:
        try:
            df = safe_read(uploaded)
            df = clean_colnames(df)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_work'] = df.copy()
            st.success(f"Arquivo carregado: {getattr(uploaded,'name','uploaded')} — {df.shape[0]} linhas.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    if sample and st.session_state['df_work'].empty:
        # small synthetic sample for dev
        df = pd.DataFrame({
            'data': pd.date_range('2023-01-01', periods=200),
            'vendas': np.random.poisson(50, 200) + np.linspace(0,50,200),
            'categoria': np.random.choice(['A','B','C'], 200),
            'cliente': np.random.choice([f'cliente_{i}' for i in range(1,51)], 200),
            'review': np.random.choice(['bom','ruim','excelente','devolvi','gostei muito','nao gostei'], 200)
        })
        df = clean_colnames(df)
        st.session_state['df_raw'] = df.copy()
        st.session_state['df_work'] = df.copy()
        st.info("Modo amostra (dev) ativo — carregado dataset de exemplo.")

    if st.session_state['df_work'].empty:
        st.info("Sem dados carregados. Carregue um CSV/XLSX na barra lateral ou ative o modo 'dev' para experimentar.")
        return

    df = st.session_state['df_work']

    # Top menu
    menu = st.radio("", ["Data Quality", "Data Studio (ETL)", "Visual Studio (Gráficos)", "Relatório", "NLP & Texto", "Anomalias & Clustering", "AutoML", "Time Series", "SQL Lab", "Export"], horizontal=True)

    # Data Quality
    if menu == "Data Quality":
        st.header("🔍 Data Quality & EDA")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Linhas</h3><h2>{format_number(df.shape[0])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Colunas</h3><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{format_number(int(df.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{format_number(int(df.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Mapa de Dados Faltantes (amostra segura)")
        if df.shape[0] > 2000:
            view = df.sample(2000)
            st.caption("Amostra (2000 linhas) usada para evitar lentidão.")
        else:
            view = df
        try:
            fig = px.imshow(view.isna(), aspect="auto", color_continuous_scale=['#eee','#ff6b6b'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Erro ao gerar heatmap de nulos:", e)

        st.markdown("---")
        st.subheader("Perfil de Colunas")
        col = st.selectbox("Selecione coluna", df.columns)
        st.write("Tipo:", df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            st.write("Estatísticas:")
            st.dataframe(df[col].describe().to_frame().T, width='stretch')
            # histogram + box
            fig = px.histogram(df, x=col, marginal="box", nbins=40, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Top valores:")
            st.dataframe(df[col].value_counts().head(20), width='stretch')

    # Data Studio (ETL)
    elif menu == "Data Studio (ETL)":
        st.header("🛠️ Data Studio — ETL No-Code")
        tabs = st.tabs(["Criar Coluna", "Renomear/Converter", "Dummies & Scaling", "Merge & Filtrar"])
        with tabs[0]:
            st.subheader("➕ Criar Coluna")
            op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
            if op == "Aritmética":
                nums = df.select_dtypes(include=np.number).columns.tolist()
                if not nums:
                    st.info("Sem colunas numéricas para operar.")
                else:
                    a = st.selectbox("Coluna A", nums)
                    mode = st.radio("Coluna B", ["Outra Coluna","Valor Fixo"])
                    if mode == "Outra Coluna":
                        b = st.selectbox("Coluna B", nums, index=0)
                        val_b = None
                    else:
                        b = None
                        val_b = st.number_input("Valor Fixo", value=1.0)
                    op_sym = st.selectbox("Operador", ["+","-","*","/"])
                    new_name = st.text_input("Nome nova coluna", f"{a}{op_sym}calc")
                    if st.button("Criar coluna aritmética"):
                        try:
                            if b is not None:
                                if op_sym == "+": df[new_name] = df[a] + df[b]
                                elif op_sym == "-": df[new_name] = df[a] - df[b]
                                elif op_sym == "*": df[new_name] = df[a] * df[b]
                                else: df[new_name] = df[a] / df[b].replace(0, np.nan)
                            else:
                                if op_sym == "+": df[new_name] = df[a] + val_b
                                elif op_sym == "-": df[new_name] = df[a] - val_b
                                elif op_sym == "*": df[new_name] = df[a] * val_b
                                else: df[new_name] = df[a] / (val_b if val_b!=0 else np.nan)
                            st.session_state['df_work'] = df
                            st.success("Coluna criada.")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            elif op == "Condicional (IF)":
                cols_num = df.select_dtypes(include=np.number).columns.tolist()
                cols_all = df.columns.tolist()
                target = st.selectbox("Coluna", cols_all)
                operator = st.selectbox("Operador", [">","<",">=","<=","==","!="])
                value = st.text_input("Valor comparativo (digite número ou texto)")
                true_label = st.text_input("Valor se True", "TRUE")
                false_label = st.text_input("Valor se False", "FALSE")
                new_col = st.text_input("Nome nova coluna", f"{target}_flag")
                if st.button("Criar condicional"):
                    try:
                        # try numeric cast, else compare as string
                        try:
                            vnum = float(value)
                            if operator == ">": mask = df[target] > vnum
                            elif operator == "<": mask = df[target] < vnum
                            elif operator == ">=": mask = df[target] >= vnum
                            elif operator == "<=": mask = df[target] <= vnum
                            elif operator == "==": mask = df[target] == vnum
                            else: mask = df[target] != vnum
                        except Exception:
                            if operator == "==": mask = df[target].astype(str) == value
                            elif operator == "!=": mask = df[target].astype(str) != value
                            else:
                                st.error("Operador inválido para comparação string.")
                                mask = pd.Series([False]*len(df))
                        df[new_col] = np.where(mask, true_label, false_label)
                        st.session_state['df_work'] = df
                        st.success("Coluna condicional criada.")
                    except Exception as e:
                        st.error(f"Erro: {e}")
            elif op == "Extrair Data":
                date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
                if not date_cols:
                    st.info("Nenhuma coluna datetime detectada. Converta primeiro (Renomear/Converter).")
                else:
                    dc = st.selectbox("Coluna Data", date_cols)
                    comp = st.selectbox("Componente", ["year","month","day","weekday","quarter"])
                    newn = st.text_input("Nome nova coluna", f"{dc}_{comp}")
                    if st.button("Extrair componente"):
                        try:
                            if comp == 'year': df[newn] = df[dc].dt.year
                            elif comp == 'month': df[newn] = df[dc].dt.month
                            elif comp == 'day': df[newn] = df[dc].dt.day
                            elif comp == 'weekday': df[newn] = df[dc].dt.day_name()
                            elif comp == 'quarter': df[newn] = df[dc].dt.quarter
                            st.session_state['df_work'] = df
                            st.success("Componente extraído.")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            elif op == "Split Texto":
                text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
                if not text_cols:
                    st.info("Sem colunas de texto.")
                else:
                    tc = st.selectbox("Coluna texto", text_cols)
                    sep = st.text_input("Separador", " ")
                    idx = st.number_input("Índice (0=primeiro)", min_value=0, value=0)
                    newc = st.text_input("Nome nova coluna", f"{tc}_part{int(idx)}")
                    if st.button("Executar split"):
                        try:
                            parts = df[tc].astype(str).str.split(sep, expand=True)
                            if idx < parts.shape[1]:
                                df[newc] = parts[int(idx)]
                                st.session_state['df_work'] = df
                                st.success("Split aplicado.")
                            else:
                                st.error("Índice fora do range.")
                        except Exception as e:
                            st.error(f"Erro: {e}")

        with tabs[1]:
            st.subheader("✏️ Renomear / Converter Tipos")
            colr = st.selectbox("Renomear coluna", df.columns)
            newname = st.text_input("Novo nome", value=colr)
            if st.button("Aplicar Renome"):
                df.rename(columns={colr: newname}, inplace=True)
                st.session_state['df_work'] = df
                st.success("Renomeado.")
            st.markdown("---")
            colc = st.selectbox("Converter coluna", df.columns, key="convert_col")
            to = st.selectbox("Converter para", ["Data","Número (float)","Texto"])
            if st.button("Converter Tipo"):
                try:
                    if to == "Data":
                        df[colc] = pd.to_datetime(df[colc], errors='coerce')
                    elif to == "Número (float)":
                        df[colc] = pd.to_numeric(df[colc], errors='coerce')
                    else:
                        df[colc] = df[colc].astype(str)
                    st.session_state['df_work'] = df
                    st.success("Conversão aplicada.")
                except Exception as e:
                    st.error(f"Erro: {e}")

        with tabs[2]:
            st.subheader("🧪 Preparação para ML: Dummies & Scaling")
            cats = df.select_dtypes(include=['object','category']).columns.tolist()
            nums = df.select_dtypes(include=np.number).columns.tolist()
            with st.expander("One-Hot Encoding (Dummies)"):
                if not cats:
                    st.info("Sem colunas categóricas.")
                else:
                    cols = st.multiselect("Colunas para Dummies", cats)
                    drop_first = st.checkbox("Drop first (reduzir multicolinearidade)", value=True)
                    if st.button("Gerar Dummies"):
                        try:
                            df = pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)
                            st.session_state['df_work'] = df
                            st.success("Dummies criadas.")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            with st.expander("Escalonamento (Scaling)"):
                if not nums:
                    st.info("Sem colunas numéricas.")
                else:
                    scl_cols = st.multiselect("Colunas para escalar", nums)
                    method = st.selectbox("Método", ["Padronização (Z-Score)","Normalização (0-1)","Robusto (Outliers)"])
                    if st.button("Aplicar escala"):
                        try:
                            df2 = df.copy()
                            if method == "Padronização (Z-Score)": scaler = StandardScaler()
                            elif method == "Normalização (0-1)": scaler = MinMaxScaler()
                            else: scaler = RobustScaler()
                            df2[scl_cols] = scaler.fit_transform(df2[scl_cols])
                            st.session_state['df_work'] = df2
                            st.success("Escala aplicada.")
                        except Exception as e:
                            st.error(f"Erro: {e}")

        with tabs[3]:
            st.subheader("🔗 Merge / Filtrar")
            file2 = st.file_uploader("Carregar segunda tabela (para merge)", type=['csv','xlsx'])
            if file2:
                try:
                    df2 = safe_read(file2)
                    df2 = clean_colnames(df2)
                    st.write("Preview:", df2.head(3))
                    lcol = st.selectbox("Chave (tabela principal)", df.columns)
                    rcol = st.selectbox("Chave (tabela 2)", df2.columns)
                    how = st.selectbox("Tipo de merge", ["left","inner","right","outer"])
                    if st.button("Executar merge"):
                        dfm = pd.merge(df, df2, left_on=lcol, right_on=rcol, how=how)
                        st.session_state['df_work'] = dfm
                        st.success("Merge aplicado.")
                except Exception as e:
                    st.error(f"Erro no merge: {e}")
            st.markdown("---")
            st.subheader("Filtros Rápidos")
            fcol = st.selectbox("Coluna para filtrar", df.columns)
            if pd.api.types.is_numeric_dtype(df[fcol]):
                mn,mx = float(df[fcol].min()), float(df[fcol].max())
                sel = st.slider("Intervalo", mn, mx, (mn,mx))
                if st.button("Aplicar filtro"):
                    st.session_state['df_work'] = df[(df[fcol]>=sel[0]) & (df[fcol]<=sel[1])]
                    st.success("Filtro aplicado.")
            else:
                vals = df[fcol].dropna().unique().tolist()
                sel = st.multiselect("Valores", vals[:500], default=vals[:10])
                if st.button("Aplicar filtro (categoria)"):
                    st.session_state['df_work'] = df[df[fcol].isin(sel)]
                    st.success("Filtro aplicado.")

    # Visual Studio (Graphs)
    elif menu == "Visual Studio (Gráficos)":
        st.header("🎨 Visual Studio — Criador de Gráficos")
        left, right = st.columns([1,2])
        with left:
            chart_type = st.selectbox("Tipo", ["Barras","Linha","Dispersão","Pizza","Histograma","Box","Heatmap","Treemap"])
            x = st.selectbox("X", df.columns.tolist())
            y = None
            if chart_type not in ("Pizza","Heatmap","Treemap"):
                y = st.selectbox("Y", [None]+df.select_dtypes(include=np.number).columns.tolist(), index=1 if df.select_dtypes(include=np.number).any() else 0)
            color = st.selectbox("Cor (opcional)", [None]+df.columns.tolist())
            agg = None
            if chart_type in ("Barras","Linha","Treemap"):
                use_agg = st.checkbox("Agrupar e agregar", value=True)
                if use_agg:
                    agg = st.selectbox("Função de agregação", ["sum","mean","count","min","max"])
            theme = st.selectbox("Tema", ["plotly_white","plotly_dark"])
            height = st.slider("Altura", 300, 900, 500)
            title = st.text_input("Título", f"{chart_type}: {y or ''} por {x}")
            if st.button("Gerar gráfico"):
                try:
                    fig = build_chart(chart_type, df, x=x, y=y, color=color if color else None, agg=agg, height=height, theme=theme)
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {'title': title, 'type': chart_type}
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")
        with right:
            if st.session_state.get('last_fig') is not None:
                st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
                note = st.text_area("Nota para o relatório")
                c1,c2 = st.columns(2)
                with c1:
                    if st.button("➕ Adicionar ao relatório"):
                        st.session_state['report_charts'].append({
                            'fig': st.session_state['last_fig'],
                            'title': st.session_state['last_meta'].get('title','(sem título)'),
                            'type': st.session_state['last_meta'].get('type',''),
                            'note': note
                        })
                        st.success("Adicionado ao relatório.")
                with c2:
                    if st.button("Salvar preset (gráfico)"):
                        pname = st.text_input("Nome do preset", value=f"preset_{len(st.session_state['presets'])+1}")
                        presets = st.session_state.get('presets', {})
                        presets[pname] = {'meta': st.session_state['last_meta'], 'note': note}
                        st.session_state['presets'] = presets
                        st.success("Preset salvo.")

    # Report
    elif menu == "Relatório":
        st.header("📑 Relatório / Dashboard Builder")
        charts = st.session_state.get('report_charts', [])
        right = st.sidebar
        right.header("Relatório rápido")
        if st.sidebar.button("Limpar relatório"):
            st.session_state['report_charts'] = []
            st.experimental_rerun()
        if not charts:
            st.info("Relatório vazio. Adicione gráficos em Visual Studio.")
        else:
            for i,c in enumerate(charts):
                st.markdown("---")
                c1,c2 = st.columns([3,1])
                with c1:
                    try:
                        st.plotly_chart(c['fig'], use_container_width=True)
                    except Exception:
                        st.write(c.get('title','(sem título)'))
                with c2:
                    st.write(f"**{c.get('title','(sem título)')}**")
                    st.write(c.get('note',''))
                    if st.button(f"Remover #{i+1}", key=f"del_{i}"):
                        st.session_state['report_charts'].pop(i)
                        st.experimental_rerun()
            st.markdown("---")
            kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
            if st.button("📄 Gerar PDF Executivo"):
                try:
                    pdf_bytes = generate_pdf_report(df, charts, kpis)
                    st.download_button("Baixar PDF", pdf_bytes, "report.pdf", "application/pdf")
                    st.success("PDF gerado.")
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")

    # NLP & Text
    elif menu == "NLP & Texto":
        st.header("🧠 NLP & Text Tools")
        text_cols = df.select_dtypes(include=['object','string']).columns.tolist()
        if not text_cols:
            st.info("Sem colunas de texto. Converta colunas se necessário.")
        else:
            tc = st.selectbox("Coluna de texto", text_cols)
            st.markdown("Exemplos de ferramentas leves: sentimento (lexicon), TF-IDF top terms.")
            if st.button("Analisar sentimento (lexicon)"):
                srs = df[tc].fillna("").astype(str)
                senti = srs.apply(lexicon_sentiment)
                out = pd.DataFrame(list(senti), columns=['sent_label','sent_score'])
                df_out = pd.concat([df[[tc]].reset_index(drop=True), out], axis=1)
                st.write("Distribuição:")
                st.bar_chart(df_out['sent_label'].value_counts())
                safe_display_dataframe(df_out.head(200))
                if st.button("Adicionar coluna sentimento ao dataset"):
                    st.session_state['df_work'][f"{tc}_sentiment"] = df_out['sent_label'].values
                    st.success("Coluna adicionada.")
            if st.button("Top TF-IDF (max 30)"):
                vec = TfidfVectorizer(max_features=30, stop_words='english')
                try:
                    X = vec.fit_transform(df[tc].astype(str).values)
                    freqs = np.asarray(X.sum(axis=0)).ravel()
                    terms = vec.get_feature_names_out()
                    top = pd.Series(freqs, index=terms).sort_values(ascending=False)
                    st.bar_chart(top)
                except Exception as e:
                    st.error(f"Erro TF-IDF: {e}")

    # Anomalias & Clustering
    elif menu == "Anomalias & Clustering":
        st.header("🚨 Anomalias & Clusterização")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("Sem colunas numéricas.")
        else:
            st.subheader("Anomalias")
            col_an = st.selectbox("Coluna (Z-score)", numeric_cols, key="an_col")
            z_thresh = st.slider("Z-threshold", 2.0, 6.0, 3.0)
            if st.button("Detectar anomalias (zscore)"):
                out = df[np.abs(stats.zscore(df[col_an].dropna())) > z_thresh]
                st.write(f"Anomalias detectadas: {len(out)}")
                safe_display_dataframe(out.head(200))
            if st.button("Detectar anomalias (IsolationForest)"):
                try:
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    arr = df[numeric_cols].dropna()
                    iso.fit(arr)
                    preds = iso.predict(arr)
                    out_iso = arr[preds == -1]
                    st.write(f"Anomalias IsolationForest: {len(out_iso)}")
                    safe_display_dataframe(out_iso.head(200))
                except Exception as e:
                    st.error(f"Erro IsolationForest: {e}")

            st.markdown("---")
            st.subheader("Clusterização (KMeans com PCA seguro)")
            feats = st.multiselect("Features", numeric_cols, default=numeric_cols[:4])
            k = st.slider("K", 2, 12, 3)
            if st.button("Rodar KMeans"):
                if len(feats) < 1:
                    st.error("Selecione ao menos 1 feature.")
                else:
                    X = df[feats].dropna()
                    if X.shape[0] < k:
                        st.error("Amostras insuficientes.")
                    else:
                        scaler = StandardScaler()
                        Xs = scaler.fit_transform(X)
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        labels = kmeans.fit_predict(Xs)
                        comps = min(2, Xs.shape[1])
                        if comps < 1:
                            st.error("PCA não aplicável.")
                        else:
                            pca = PCA(n_components=comps)
                            pcs = pca.fit_transform(Xs)
                            if comps ==1:
                                fig = px.scatter(x=pcs[:,0], y=np.zeros(len(pcs)), color=labels.astype(str), title="Clusters (1D)")
                            else:
                                fig = px.scatter(x=pcs[:,0], y=pcs[:,1], color=labels.astype(str), title="Clusters (PCA 2D)")
                            st.plotly_chart(fig, use_container_width=True)
                            # attach cluster (safe by index alignment)
                            df_cl = X.copy()
                            df_cl['cluster'] = labels
                            st.session_state['df_work'] = pd.merge(df, df_cl[['cluster']], left_index=True, right_index=True, how='left')
                            st.success("Cluster adicionado (coluna 'cluster').")

    # AutoML
    elif menu == "AutoML":
        st.header("🤖 AutoML Simplificado")
        df_cols = df.columns.tolist()
        target = st.selectbox("Target (coluna a prever)", df_cols)
        feats = st.multiselect("Features (colunas de entrada)", [c for c in df_cols if c != target])
        use_fs = st.checkbox("Usar Feature Selection (SelectKBest)", value=False)
        if st.button("Treinar modelo (simples)") and feats:
            is_reg = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20
            with st.spinner("Treinando..."):
                model, metrics, imp = simple_automl_train(df, target, feats, is_regression=is_reg, use_fs=use_fs)
            if is_reg:
                st.metric("R²", f"{metrics.get('r2',0):.4f}")
                st.metric("MAE", f"{metrics.get('mae',0):.4f}")
            else:
                st.metric("Acurácia", f"{metrics.get('acc',0):.2%}")
            if imp is not None:
                st.subheader("Feature importance (permutation)")
                st.dataframe(imp.head(20))
            # allow download model
            if joblib is not None:
                bio = io.BytesIO()
                joblib.dump(model, bio)
                st.download_button("Baixar modelo (.joblib)", bio.getvalue(), "modelo.joblib")
            else:
                st.info("joblib não disponível — não é possível baixar o modelo.")

    # Time Series
    elif menu == "Time Series":
        st.header("🔮 Time Series — Forecast (Auto ARIMA lite)")
        date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if not date_cols:
            st.info("Nenhuma coluna de data detectada. Converta uma coluna para datetime no ETL.")
        else:
            cdt = st.selectbox("Coluna data", date_cols)
            vals = df.select_dtypes(include=np.number).columns.tolist()
            if not vals:
                st.info("Nenhuma coluna numérica para previsão.")
            else:
                cval = st.selectbox("Coluna valor", vals)
                horizon = st.slider("Horizonte (períodos)", 7, 180, 30)
                if st.button("Gerar Forecast"):
                    m, pred, conf = auto_arima_forecast(st.session_state['df_work'], cdt, cval, horizon)
                    if m is None:
                        st.error("Não foi possível ajustar modelo ARIMA.")
                    else:
                        fig = go.Figure()
                        hist = st.session_state['df_work'].set_index(cdt)[cval].sort_index()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico"))
                        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Previsão"))
                        try:
                            fig.add_traces([
                                go.Scatter(x=conf.index, y=conf.iloc[:,0], showlegend=False, line=dict(width=0)),
                                go.Scatter(x=conf.index, y=conf.iloc[:,1], name="Intervalo", fill='tonexty', fillcolor='rgba(200,200,200,0.3)')
                            ])
                        except Exception:
                            pass
                        st.plotly_chart(fig, use_container_width=True)

    # SQL Lab
    elif menu == "SQL Lab":
        st.header("💾 SQL Lab (duckdb preferido / sqlite fallback)")
        st.info("A tabela disponível se chamará 'dados'. Exemplo: SELECT * FROM dados LIMIT 100")
        q = st.text_area("Query SQL", value="SELECT * FROM dados LIMIT 100", height=200)
        if st.button("Executar Query"):
            res, err = run_sql(df, q)
            if err:
                st.error(f"Erro SQL: {err}")
            else:
                st.session_state['sql_history'].append(q)
                st.success(f"{len(res)} linhas retornadas")
                safe_display_dataframe(res.head(1000))
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar CSV", csv, "sql_result.csv", "text/csv")
        if st.session_state['sql_history']:
            with st.expander("Histórico de queries"):
                for qh in reversed(st.session_state['sql_history'][-10:]):
                    st.code(qh, language='sql')

    # Export
    elif menu == "Export":
        st.header("📤 Exportar & Relatórios")
        df_export = st.session_state['df_work']
        st.download_button("Baixar CSV (tratado)", df_export.to_csv(index=False).encode('utf-8'), "dados_tratados.csv", "text/csv")
        # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Dados')
            desc = df_export.describe(include='all').transpose()
            try:
                desc.to_excel(writer, sheet_name='Resumo')
            except Exception:
                pass
        st.download_button("Baixar Excel", buffer.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("---")
        kpis = {"rows": len(df_export), "cols": df_export.shape[1], "nulls": int(df_export.isna().sum().sum()), "dups": int(df_export.duplicated().sum())}
        if st.button("Gerar PDF Executivo (com gráficos do relatório)"):
            charts = st.session_state.get('report_charts', [])
            try:
                pdf_bytes = generate_pdf_report(df_export, charts, kpis)
                st.download_button("Baixar PDF", pdf_bytes, "report_executivo.pdf", "application/pdf")
                st.success("PDF gerado.")
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {e}")

    # persist working df
    st.session_state['df_work'] = st.session_state.get('df_work', df)

if __name__ == "__main__":
    main()
