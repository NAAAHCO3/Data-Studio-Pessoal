"""
Enterprise Analytics — Ultra Refactored Streamlit App v2.0
Single-file app (paste & run).

Key Improvements in v2.0:
1. DuckDB Integration: Zero-copy SQL querying for high performance.
2. AutoML Leaderboard: Compare multiple models simultaneously.
3. State Management: "Reset Dataset" button to undo filters.
4. Python Sandbox: Execute custom Pandas transformations on the fly.

Features:
- Robust file reading (CSV/XLSX)
- Modular engines: DataEngine (DuckDB), AutoML, TimeSeries, NLP
- SQL playground (DuckDB powered)
- Python Code Sandbox
- Visuals: Plotly integration
- Export: CSV and PDF (FPDF)
"""

# Dev path used previously in conversation (replace if needed)
UPLOADED_FILE_PATH = "/mnt/data/uploaded_dataset.csv"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb  # NEW: High performance SQL
import joblib
import logging
import io
import traceback
from io import BytesIO
from datetime import datetime
from typing import Tuple, List, Optional, Dict

# Stats & TS
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# ML
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression # NEW: For comparison
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB

# PDF
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Logging
logging.basicConfig(level=logging.INFO)

# Streamlit page config
st.set_page_config(page_title="Enterprise Analytics — Ultra v2", layout="wide", page_icon="📊")

# ---------------- Utility functions ----------------

def try_read_file(file_like) -> pd.DataFrame:
    """
    Robust read for CSV and Excel.
    Accepts uploaded file-like object or local path string.
    """
    if isinstance(file_like, str):
        path = file_like
        if path.lower().endswith('.csv'):
            for enc in ('utf-8', 'latin1', 'cp1252', 'iso-8859-1'):
                try:
                    return pd.read_csv(path, encoding=enc)
                except Exception as e:
                    last_exc = e
            raise last_exc
        else:
            return pd.read_excel(path)
    else:
        try:
            file_like.seek(0)
            return pd.read_csv(file_like)
        except Exception:
            try:
                file_like.seek(0)
                data = file_like.read()
                if isinstance(data, bytes):
                    for enc in ('utf-8', 'latin1', 'cp1252', 'iso-8859-1'):
                        try:
                            s = data.decode(enc)
                            return pd.read_csv(io.StringIO(s))
                        except Exception:
                            continue
                file_like.seek(0)
                return pd.read_excel(io.BytesIO(data))
            except Exception as e:
                raise

def safe_to_str(x):
    try:
        return str(x)
    except Exception:
        return ""

def format_big(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

# ---------------- Theme ----------------

def set_theme():
    st.markdown(
        """
        <style>
        .metric-card{
            background:#262730;padding:16px;border-radius:10px;border-left:5px solid #00CC96;color:#FAFAFA;margin-bottom:10px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        }
        .metric-card h3{margin:0;color:#BBB;font-size:0.9rem}
        .metric-card h2{margin:6px 0 0 0;font-size:1.6rem}
        [data-baseweb="tag"]{background:#E1C16E !important;color:#111 !important}
        div[data-testid="stDataFrame"]{border:1px solid #444 !important}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Data Engine (DuckDB Enhanced) ----------------

class DataEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load(file) -> pd.DataFrame:
        try:
            df = try_read_file(file)
            # Basic normalization
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            logging.error(f"load error: {e}")
            raise

    @staticmethod
    def run_query(df: pd.DataFrame, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Executes SQL directly on the DataFrame using DuckDB.
        Zero-copy (mostly) and extremely fast.
        """
        try:
            # DuckDB memory connection
            conn = duckdb.connect(database=':memory:')
            
            # Register the dataframe as virtual tables
            # 'df' is the standard variable name in the sandbox/context
            conn.register('df', df)
            conn.register('dados', df) # Legacy support alias
            
            res = conn.execute(query).df()
            return res, None
        except Exception as e:
            return None, str(e)

# ---------------- Insights Engine ----------------

class InsightsEngine:
    def summarize_column(self, df: pd.DataFrame, col: str) -> Dict:
        ser = df[col]
        out = {'name': col, 'dtype': str(ser.dtype), 'n_missing': int(ser.isna().sum()), 'pct_missing': float(ser.isna().mean())}
        if pd.api.types.is_numeric_dtype(ser):
            ser_non = ser.dropna()
            out.update({
                'mean': float(ser_non.mean()) if not ser_non.empty else None,
                'median': float(ser_non.median()) if not ser_non.empty else None,
                'std': float(ser_non.std()) if not ser_non.empty else None,
                'min': float(ser_non.min()) if not ser_non.empty else None,
                'max': float(ser_non.max()) if not ser_non.empty else None,
                'skew': float(ser_non.skew()) if not ser_non.empty else None,
                'n_outliers_iqr': int(((ser_non < (ser_non.quantile(0.25) - 1.5 * (ser_non.quantile(0.75)-ser_non.quantile(0.25)))) | (ser_non > (ser_non.quantile(0.75) + 1.5 * (ser_non.quantile(0.75)-ser_non.quantile(0.25))))).sum())
            })
        else:
            out.update({
                'n_unique': int(ser.nunique()),
                'top_values': ser.value_counts().head(10).to_dict()
            })
        return out

    def generate_insights(self, df: pd.DataFrame, col: str) -> List[str]:
        ser = df[col]
        insights = []
        pct_missing = ser.isna().mean()
        if pct_missing > 0.05:
            insights.append(f"⚠️ Alta taxa de nulos: {pct_missing:.1%}. Considere imputação ou excluir a coluna.")
        if pd.api.types.is_numeric_dtype(ser):
            sk = ser.dropna().skew()
            if abs(sk) > 1:
                insights.append(f"📈 Assimetria forte (skew={sk:.2f}). Considere transformação log/power.")
            q1, q3 = ser.dropna().quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ser[(ser < (q1 - 1.5*iqr)) | (ser > (q3 + 1.5*iqr))].shape[0]
            if outliers > 0:
                insights.append(f"🚨 {outliers} outliers detectados (IQR).")
            # correlation
            nums = df.select_dtypes(include=np.number)
            if col in nums.columns and nums.shape[1] > 1:
                corrs = nums.corr()[col].drop(col).abs().sort_values(ascending=False)
                top = corrs.head(3)
                if not top.empty and top.iloc[0] > 0.8:
                    insights.append(f"🔗 Altas correlações detectadas: {list(top.index[top > 0.8])}")
        else:
            nuniq = ser.nunique(dropna=True)
            if nuniq > 50:
                insights.append(f"🔢 Alta cardinalidade: {nuniq} valores únicos.")
            vc = ser.value_counts(normalize=True)
            if not vc.empty and vc.iloc[0] > 0.75:
                top_label = vc.index[0]
                insights.append(f"⚖️ Desbalanceamento: categoria '{top_label}' representa {vc.iloc[0]:.1%} dos registros.")
        return insights

# ---------------- Time Series Engine ----------------

class TimeSeriesEngine:
    def auto_arima_lite(self, ts: pd.Series, max_p=2, max_d=1, max_q=2):
        best_aic = np.inf
        best_tuple = None  # (p,d,q,model)
        for p in range(max_p+1):
            for d in range(max_d+1):
                for q in range(max_q+1):
                    try:
                        model = ARIMA(ts, order=(p,d,q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_tuple = (p,d,q,model)
                    except Exception:
                        continue
        return best_tuple

    def forecast(self, df: pd.DataFrame, date_col: str, val_col: str, steps: int = 30):
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
        tmp = tmp.dropna(subset=[date_col])
        ts = tmp.sort_values(by=date_col).set_index(date_col)[val_col]
        freq = pd.infer_freq(ts.index)
        if freq is None:
            ts = ts.asfreq('D').fillna(method='ffill')
        else:
            ts = ts.asfreq(freq).fillna(method='ffill')
        best = self.auto_arima_lite(ts)
        if best is None:
            raise RuntimeError("Nenhum modelo ARIMA convergiu")
        p,d,q,model = best
        fc = model.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        return ts, mean, ci, (p,d,q)

# ---------------- AutoML Engine (Enhanced) ----------------

class AutoMLEngine:
    def __init__(self, random_state:int=42):
        self.random_state = random_state

    def build_preproc(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        transformers = []
        if num_cols:
            transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols))
        preproc = ColumnTransformer(transformers, remainder='drop')
        return preproc

    def train(self, df: pd.DataFrame, target: str, features: List[str], algo: str = 'Random Forest', is_reg: bool = True, use_fs: bool = False):
        X = df[features].copy()
        y = df[target].copy()
        preproc = self.build_preproc(X)
        steps = [('pre', preproc)]
        if use_fs:
            k = min(20, max(1, len(features)))
            func = f_regression if is_reg else f_classif
            steps.append(('fs', SelectKBest(score_func=func, k=k)))
        
        if is_reg:
            if algo == 'Linear Regression':
                clf = LinearRegression()
            elif algo == 'Hist Gradient Boosting':
                clf = HistGradientBoostingRegressor(random_state=self.random_state)
            else:
                clf = RandomForestRegressor(n_jobs=-1, random_state=self.random_state)
        else:
            if algo == 'Logistic Regression':
                clf = LogisticRegression(max_iter=1000)
            elif algo == 'Hist Gradient Boosting':
                clf = HistGradientBoostingClassifier(random_state=self.random_state)
            else:
                clf = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
        
        steps.append(('clf', clf))
        pipe = Pipeline(steps)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y if (not is_reg and len(y.unique())>1) else None)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        # Permutation Importance
        imp = None
        try:
            res = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=self.random_state, n_jobs=-1)
            # Simple approach to feature names
            cols = X_test.columns.tolist() 
            imp = pd.DataFrame({'feature': cols[:len(res.importances_mean)], 'importance': res.importances_mean}).sort_values('importance', ascending=False)
        except Exception:
            imp = None
            
        return pipe, X_test, y_test, preds, imp

    def compare_models(self, df: pd.DataFrame, target: str, features: List[str], is_reg: bool = True):
        """
        Trains multiple models and returns a leaderboard DataFrame.
        """
        X = df[features].copy()
        y = df[target].copy()
        preproc = self.build_preproc(X)
        
        if is_reg:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=self.random_state),
                "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=self.random_state)
            }
            metric_name = "R2 Score"
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=self.random_state),
                "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=self.random_state)
            }
            metric_name = "Accuracy"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y if (not is_reg and len(y.unique())>1) else None)
        
        results = []
        
        for name, model in models.items():
            try:
                pipe = Pipeline([('pre', preproc), ('clf', model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                
                if is_reg:
                    score = r2_score(y_test, preds)
                    err = mean_absolute_error(y_test, preds)
                    results.append({"Modelo": name, metric_name: score, "MAE": err})
                else:
                    score = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='weighted')
                    results.append({"Modelo": name, metric_name: score, "F1-Score": f1})
            except Exception as e:
                results.append({"Modelo": name, metric_name: 0, "Erro": str(e)})

        return pd.DataFrame(results).sort_values(by=metric_name, ascending=False)

# ---------------- NLP Module ----------------

class NLPModule:
    def __init__(self):
        self.tfidf = None

    def fit_tfidf(self, texts: List[str], max_features=2000):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = self.tfidf.fit_transform(texts)
        return X

    def top_terms(self, X, top_n=30):
        if self.tfidf is None:
            return {}
        sums = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(self.tfidf.get_feature_names_out())
        idx = np.argsort(sums)[::-1][:top_n]
        return dict(zip(terms[idx], sums[idx].round(3)))

    def fit_nmf(self, X_tfidf, n_topics=5):
        nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=400)
        W = nmf.fit_transform(X_tfidf)
        H = nmf.components_
        terms = np.array(self.tfidf.get_feature_names_out())
        topics = []
        for i, comp in enumerate(H):
            idx = np.argsort(comp)[::-1][:15]
            topics.append([terms[t] for t in idx])
        return topics, W

    def supervised_text_model(self, texts, labels, algo='nb'):
        vec = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        if algo == 'nb':
            clf = MultinomialNB()
        elif algo == 'logreg':
            clf = LogisticRegression(max_iter=1000)
        else:
            clf = MultinomialNB()
        pipe = Pipeline([('vec', vec), ('clf', clf)])
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels))>1 else None)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted') if len(set(labels))>1 else None
        return pipe, {'accuracy': acc, 'f1': f1}, (X_test, y_test, preds)

# ---------------- PDF Report ----------------

def generate_pdf_report(df: pd.DataFrame, kpis: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica','B',16)
    pdf.cell(0,10,'Relatorio Executivo — Enterprise Analytics', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,8,f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,42,190,30,'F')
    pdf.set_y(47)
    colw = 190/4
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    vals = [format_big(kpis.get('rows','')), format_big(kpis.get('cols','')), format_big(kpis.get('nulls','')), format_big(kpis.get('dups',''))]
    for i,t in enumerate(titles):
        pdf.set_font('Helvetica','B',11)
        pdf.cell(colw,8,t,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(8)
    pdf.set_font('Helvetica','',12)
    for i,v in enumerate(vals):
        pdf.cell(colw,8,v,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(12)
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Resumo Estatistico (Top 10 variaveis numericas):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    desc = df.describe().T.reset_index().head(10)
    if not desc.empty:
        need = ['index','mean','min','max']
        if set(need).issubset(desc.columns):
            desc = desc[need]
            desc.columns = ['Variavel','Media','Min','Max']
            w=[70,40,40,40]
            pdf.set_font('Helvetica','B',10)
            for i,c in enumerate(desc.columns):
                pdf.cell(w[i],8,c,1,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
            pdf.set_font('Helvetica','',9)
            for _,row in desc.iterrows():
                for i,val in enumerate(row):
                    t = str(val)[:30]
                    pdf.cell(w[i],7,t,1,align='C' if i>0 else 'L', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------- App UI ----------------

def main():
    set_theme()
    st.sidebar.title("Enterprise Analytics — Ultra")
    st.sidebar.markdown("v2.0 - DuckDB & AutoML Leaderboard")

    # Uploader
    uploaded = st.sidebar.file_uploader("Carregar dataset (CSV/XLSX)", type=['csv','xlsx'])
    use_local = st.sidebar.checkbox("Usar caminho local (dev)", value=False)
    
    df = pd.DataFrame()
    if use_local and (uploaded is None):
        try:
            df = DataEngine.load(UPLOADED_FILE_PATH)
        except Exception as e:
            st.sidebar.error(f"Erro local: {e}")
    elif uploaded is not None:
        try:
            df = DataEngine.load(uploaded)
        except Exception as e:
            st.sidebar.error(f"Erro upload: {e}")

    if df.empty:
        st.title("Enterprise Analytics v2.0")
        st.info("Aguardando dados... Carregue um arquivo na barra lateral.")
        return

    # Normalize columns
    df.columns = [safe_to_str(c) for c in df.columns]

    # --- STATE MANAGEMENT (RESET) ---
    # Identify if file changed based on simple logic or sessionID, here we use a key check
    if 'uploader_key' not in st.session_state or st.session_state.get('uploader_key') != uploaded:
        st.session_state['df_original'] = df.copy()
        st.session_state['df_work'] = df.copy()
        st.session_state['uploader_key'] = uploaded
        st.session_state['sql_history'] = []

    # Reset Button
    if st.sidebar.button("🔄 Resetar Dataset"):
        st.session_state['df_work'] = st.session_state['df_original'].copy()
        st.rerun()

    df_work = st.session_state['df_work']

    # Auto-convert dates
    for c in df_work.select_dtypes(include='object').columns:
        if ('date' in c.lower() or 'data' in c.lower()) and df_work[c].dtype == object:
            try:
                df_work[c] = pd.to_datetime(df_work[c])
            except Exception:
                pass

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros (df_work)")
    
    date_cols = [c for c in df_work.columns if np.issubdtype(df_work[c].dtype, np.datetime64)]
    if date_cols:
        date_col = st.sidebar.selectbox("Filtro Data", ["Nenhum"] + date_cols)
        if date_col != "Nenhum":
            min_d, max_d = df_work[date_col].min(), df_work[date_col].max()
            if pd.notnull(min_d):
                dr = st.sidebar.date_input("Período", [min_d.date(), max_d.date()])
                if isinstance(dr, list) and len(dr) == 2:
                    df_work = df_work[(df_work[date_col].dt.date >= dr[0]) & (df_work[date_col].dt.date <= dr[1])]

    cat_cols = [c for c in df_work.select_dtypes(include=['object','category']).columns if df_work[c].nunique() < 50]
    if cat_cols:
        c = st.sidebar.selectbox("Filtro Categoria", ["Nenhum"] + cat_cols)
        if c != "Nenhum":
            uniques = sorted(df_work[c].dropna().unique().tolist())
            vals = st.sidebar.multiselect(f"Valores de {c}", uniques, default=uniques)
            if vals:
                df_work = df_work[df_work[c].isin(vals)]

    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Menu", ["Dashboard","SQL Lab (DuckDB)","Python Sandbox","Engenharia","AutoML","TimeSeries","NLP","Clustering","Export"])

    # --- DASHBOARD ---
    if menu == "Dashboard":
        st.header("Visão Executiva")
        r, c = df_work.shape
        col1,col2,col3,col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><h3>Registros</h3><h2>{format_big(r)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><h3>Variáveis</h3><h2>{format_big(c)}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{format_big(df_work.isna().sum().sum())}</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{format_big(df_work.duplicated().sum())}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Column Explorer")
        col = st.selectbox("Coluna", df_work.columns)
        ie = InsightsEngine()
        with st.expander("Resumo & Insights", expanded=True):
            c1, c2 = st.columns(2)
            summary = ie.summarize_column(df_work, col)
            c1.json(summary)
            ins = ie.generate_insights(df_work, col)
            if ins:
                c2.write("**Insights:**")
                for i in ins: c2.write(f"- {i}")
            else:
                c2.write("Sem insights automáticos.")

        # Quick Plots
        chart_type = st.selectbox("Gráfico", ["Histograma","Barra","Linha","Boxplot","Scatter","Heatmap"])
        left, right = st.columns([2,1])
        
        if chart_type == "Histograma" and pd.api.types.is_numeric_dtype(df_work[col]):
            fig = px.histogram(df_work, x=col, nbins=30, marginal='box', template='plotly_dark')
            left.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Barra":
            vc = df_work[col].value_counts().head(30).reset_index()
            vc.columns = [col,'count']
            left.plotly_chart(px.bar(vc, x=col, y='count', template='plotly_dark'), use_container_width=True)
        elif chart_type == "Linha":
            date_c = [c for c in df_work.columns if np.issubdtype(df_work[c].dtype, np.datetime64)]
            if date_c:
                dc = st.selectbox("Eixo X (Data)", date_c)
                agg = st.selectbox("Agregação", ["sum","mean"])
                tmp = df_work.groupby(dc)[col].agg(agg).reset_index()
                left.plotly_chart(px.line(tmp, x=dc, y=col, template='plotly_dark'), use_container_width=True)
            else:
                left.warning("Sem coluna de data.")
        elif chart_type == "Scatter":
            num_cols = df_work.select_dtypes(include=np.number).columns
            if len(num_cols) > 1:
                x_ax = st.selectbox("Eixo X", num_cols)
                left.plotly_chart(px.scatter(df_work, x=x_ax, y=col, template='plotly_dark'), use_container_width=True)
        elif chart_type == "Heatmap":
            corr = df_work.select_dtypes(include=np.number).corr()
            left.plotly_chart(px.imshow(corr, text_auto=".1f", template='plotly_dark'), use_container_width=True)
        
        right.write(df_work[col].describe())

    # --- SQL LAB (DuckDB) ---
    elif menu == "SQL Lab (DuckDB)":
        st.header("SQL Playground (Powered by DuckDB)")
        st.markdown("Query 'df' directly. Much faster than SQLite.")
        query = st.text_area("SQL Query", "SELECT * FROM df LIMIT 10", height=150)
        if st.button("Run Query"):
            res, err = DataEngine.run_query(df_work, query)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state['sql_history'].append(query)
                st.success(f"Result: {len(res)} rows")
                st.dataframe(res)
                st.download_button("Download CSV", res.to_csv(index=False).encode('utf-8'), "query_result.csv")
        
        if st.session_state['sql_history']:
            with st.expander("History"):
                for q in reversed(st.session_state['sql_history'][-10:]):
                    st.code(q, language='sql')

    # --- PYTHON SANDBOX ---
    elif menu == "Python Sandbox":
        st.header("🐍 Python Sandbox")
        st.markdown("Execute transformações complexas diretamente no DataFrame. A variável disponível é `df`.")
        
        code_default = "# Exemplo:\n# df['nova_coluna'] = df['coluna1'] * 2\n# df = df[df['valor'] > 100]"
        code = st.text_area("Código Python", height=200, value=code_default)
        
        if st.button("Executar Código"):
            try:
                # Contexto local seguro para execução
                local_vars = {'df': df_work.copy(), 'pd': pd, 'np': np}
                
                # Executa o código fornecido
                exec(code, {}, local_vars)
                
                # Recupera o df modificado
                new_df = local_vars['df']
                
                if isinstance(new_df, pd.DataFrame):
                    st.session_state['df_work'] = new_df
                    st.success("Código executado com sucesso! DataFrame atualizado.")
                    st.dataframe(new_df.head())
                else:
                    st.error("O código deve resultar em uma variável 'df' que seja um pandas DataFrame.")
            except Exception as e:
                st.error(f"Erro de execução: {e}")
                st.code(traceback.format_exc())

    # --- ENGENHARIA ---
    elif menu == "Engenharia":
        st.header("Engenharia de Dados")
        t1, t2, t3 = st.tabs(["Renomear/Tipar", "Limpeza", "Features"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Renomear")
                old = st.selectbox("Coluna", df_work.columns)
                new = st.text_input("Novo nome")
                if st.button("Renomear"):
                    df_work.rename(columns={old:new}, inplace=True)
                    st.success("Feito!")
            with c2:
                st.subheader("Converter Tipo")
                ct = st.selectbox("Coluna", df_work.columns, key='ct')
                tt = st.selectbox("Para", ["Numeric","DateTime","String"])
                if st.button("Converter"):
                    try:
                        if tt=="Numeric": df_work[ct] = pd.to_numeric(df_work[ct], errors='coerce')
                        elif tt=="DateTime": df_work[ct] = pd.to_datetime(df_work[ct], errors='coerce')
                        else: df_work[ct] = df_work[ct].astype(str)
                        st.success("Feito!")
                    except Exception as e: st.error(str(e))

        with t2:
            st.subheader("Tratamento de Nulos")
            cn = st.selectbox("Coluna Nulos", df_work.columns)
            met = st.selectbox("Método", ["Drop Rows", "Fill Mean", "Fill Median", "Fill Zero", "Fill Const"])
            if st.button("Aplicar Nulos"):
                if met=="Drop Rows": df_work.dropna(subset=[cn], inplace=True)
                elif met=="Fill Mean": df_work[cn].fillna(df_work[cn].mean(), inplace=True)
                elif met=="Fill Median": df_work[cn].fillna(df_work[cn].median(), inplace=True)
                elif met=="Fill Zero": df_work[cn].fillna(0, inplace=True)
                st.success("Feito!")
            
            if st.button("Remover Duplicatas"):
                df_work.drop_duplicates(inplace=True)
                st.success("Feito!")

        with t3:
            st.subheader("One-Hot Encoding")
            cc = st.selectbox("Coluna Categórica", ["Nenhum"] + list(df_work.select_dtypes(include='object').columns))
            if cc != "Nenhum" and st.button("Gerar Dummies"):
                df_work = pd.get_dummies(df_work, columns=[cc], drop_first=True)
                st.success("Feito!")

        st.session_state['df_work'] = df_work

    # --- AUTOML (Leaderboard) ---
    elif menu == "AutoML":
        st.header("AutoML Pro & Leaderboard")
        automl = AutoMLEngine()
        
        c1, c2 = st.columns(2)
        target = c1.selectbox("Target (Alvo)", df_work.columns)
        features = c2.multiselect("Features", [c for c in df_work.columns if c != target])
        
        is_reg = pd.api.types.is_numeric_dtype(df_work[target]) and df_work[target].nunique() > 20
        st.info(f"Modo detectado: {'Regressão' if is_reg else 'Classificação'}")
        
        tab_compare, tab_single = st.tabs(["🏆 Comparar Modelos (Leaderboard)", "⚙️ Treino Específico"])
        
        with tab_compare:
            if st.button("Rodar Comparação") and features:
                with st.spinner("Treinando múltiplos modelos..."):
                    leaderboard = automl.compare_models(df_work, target, features, is_reg)
                    st.dataframe(leaderboard.style.highlight_max(axis=0, color='green'))
                    
                    best_model_name = leaderboard.iloc[0]['Modelo']
                    st.success(f"Melhor modelo: {best_model_name}")

        with tab_single:
            algo = st.selectbox("Algoritmo", ["Random Forest","Hist Gradient Boosting", "Linear/Logistic Regression"])
            if st.button("Treinar Individual") and features:
                with st.spinner("Treinando..."):
                    pipe, X_test, y_test, preds, imp = automl.train(df_work, target, features, algo, is_reg)
                    st.success("Concluído!")
                    
                    c1, c2 = st.columns(2)
                    if is_reg:
                        c1.metric("R2", f"{r2_score(y_test, preds):.4f}")
                        c2.metric("MAE", f"{mean_absolute_error(y_test, preds):.4f}")
                        st.plotly_chart(px.scatter(x=y_test, y=preds, labels={'x':'Real','y':'Pred'}, title="Real vs Pred"), use_container_width=True)
                    else:
                        c1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                        st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True), use_container_width=True)
                    
                    if imp is not None:
                        st.subheader("Feature Importance")
                        st.dataframe(imp.head(10))

    # --- TIME SERIES ---
    elif menu == "TimeSeries":
        st.header("Time Series Studio")
        tse = TimeSeriesEngine()
        date_cols = [c for c in df_work.columns if np.issubdtype(df_work[c].dtype, np.datetime64)]
        num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
        
        if not date_cols:
            st.warning("Sem coluna de data.")
        elif not num_cols:
            st.warning("Sem coluna numérica.")
        else:
            dc = st.selectbox("Data", date_cols)
            vc = st.selectbox("Valor", num_cols)
            steps = st.slider("Steps Forecast", 7, 90, 30)
            
            if st.button("Forecast (AutoARIMA)"):
                try:
                    ts, mean, ci, order = tse.forecast(df_work, dc, vc, steps)
                    st.success(f"Modelo ARIMA{order}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Histórico'))
                    fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name='Forecast', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,1], name='Upper Bound', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,0], name='Lower Bound', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.2)'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Decomposition
                    dec = seasonal_decompose(ts.fillna(method='ffill'), period=min(12, len(ts)//4))
                    st.subheader("Decomposição")
                    st.line_chart(pd.DataFrame({'Trend':dec.trend, 'Seasonal':dec.seasonal}))
                except Exception as e:
                    st.error(f"Erro TS: {e}")

    # --- NLP ---
    elif menu == "NLP":
        st.header("NLP Studio")
        txt_cols = df_work.select_dtypes(include='object').columns
        if not len(txt_cols):
            st.warning("Sem texto.")
        else:
            tc = st.selectbox("Coluna Texto", txt_cols)
            mode = st.selectbox("Modo", ["TF-IDF Terms", "Topic Modeling", "Supervised Classif"])
            nmod = NLPModule()
            texts = df_work[tc].astype(str).fillna("")
            
            if mode == "TF-IDF Terms":
                X = nmod.fit_tfidf(texts.tolist())
                top = nmod.top_terms(X)
                st.bar_chart(pd.Series(top).sort_values(ascending=False).head(20))
            elif mode == "Topic Modeling":
                n_top = st.slider("Topics", 2, 10, 3)
                X = nmod.fit_tfidf(texts.tolist())
                topics, _ = nmod.fit_nmf(X, n_top)
                for i, t in enumerate(topics):
                    st.write(f"**Topic {i+1}:** {', '.join(t)}")
            elif mode == "Supervised Classif":
                lc = st.selectbox("Label Column", [c for c in df_work.columns if c!=tc])
                if st.button("Train"):
                    pipe, res, _ = nmod.supervised_text_model(texts.tolist(), df_work[lc].astype(str).tolist())
                    st.json(res)

    # --- CLUSTERING ---
    elif menu == "Clustering":
        st.header("Clustering (K-Means)")
        feats = st.multiselect("Features", df_work.select_dtypes(include=np.number).columns)
        k = st.slider("K", 2, 10, 3)
        if st.button("Run") and feats:
            X = StandardScaler().fit_transform(df_work[feats].dropna())
            lbls = KMeans(n_clusters=k).fit_predict(X)
            pca = PCA(2).fit_transform(X)
            fig = px.scatter(x=pca[:,0], y=pca[:,1], color=lbls.astype(str), title="PCA Projection")
            st.plotly_chart(fig, use_container_width=True)

    # --- EXPORT ---
    elif menu == "Export":
        st.header("Exportar Dados & Relatório")
        st.download_button("Download CSV (df_work)", df_work.to_csv(index=False).encode('utf-8'), "data_processed.csv")
        if st.button("Gerar PDF"):
            kpis = {'rows': len(df_work), 'cols': df_work.shape[1], 'nulls': df_work.isna().sum().sum(), 'dups': df_work.duplicated().sum()}
            try:
                pdf_b = generate_pdf_report(df_work, kpis)
                st.download_button("Download PDF", pdf_b, "report.pdf")
            except Exception as e:
                st.error(str(e))

    st.session_state['df_work'] = df_work

if __name__ == "__main__":
    main()