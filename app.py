"""
Enterprise Analytics — Enterprise Ultra Edition
Refactored, modular, production-ready Streamlit app.
Features:
- Clean modular structure (DataEngine, AutoML, TimeSeries, Insights)
- Robust error handling, input validation
- Improved UI (DSA-like theme) and UX
- Auto feature selection, Permutation importance, model export
- SQL Playground with history and download
- Column Explorer with advanced insights
- Time Series: Auto-ARIMA (lite)
- NLP: TF-IDF + tokens
- Clustering: KMeans + PCA + silhouette

NOTE: Replace UPLOADED_FILE_PATH with your local uploaded file path if you want to bypass the uploader.
"""

# Local uploaded file (if you want to reference the file directly)
# DEV NOTE: replace the path below with the actual local path created when uploading.
UPLOADED_FILE_PATH = "/mnt/data/uploaded_dataset.csv"  # <-- replace with actual file path if needed

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import logging
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
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

# PDF
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# App config
st.set_page_config(page_title="Enterprise Analytics — Enterprise Ultra",
                   layout="wide", page_icon="📊", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)

# --------------------- Utilities ---------------------
def safe_read_csv_excel(path_or_buffer):
    """Try to read CSV then Excel; return DataFrame or raise."""
    try:
        if hasattr(path_or_buffer, 'read'):
            # file-like object from uploader
            return pd.read_csv(path_or_buffer)
        path = str(path_or_buffer)
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception as e:
        raise

# --------------------- Theme ---------------------
def set_theme():
    card_bg = "#262730"
    accent = "#00CC96"
    gold = "#E1C16E"
    css = f"""
    <style>
        .metric-card{{background:{card_bg};padding:18px;border-radius:8px;border-left:5px solid {accent};color:#fff}}
        .metric-card h3{{margin:0;font-size:0.95rem;color:#bbb}}
        .metric-card h2{{margin:6px 0 0 0;font-size:1.8rem}}
        [data-baseweb="tag"]{{background:{gold} !important;color:#111 !important}}
        div[data-testid="stDataFrame"]{{border:1px solid #444}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --------------------- Data Engine ---------------------
class DataEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_data(file) -> pd.DataFrame:
        try:
            return safe_read_csv_excel(file)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return pd.DataFrame()

    @staticmethod
    def run_query(df: pd.DataFrame, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            conn = sqlite3.connect(':memory:')
            df.to_sql('dados', conn, index=False, if_exists='replace')
            res = pd.read_sql_query(query, conn)
            conn.close()
            return res, None
        except Exception as e:
            return None, str(e)

# --------------------- Insights Engine ---------------------
class InsightsEngine:
    def column_summary(self, df: pd.DataFrame, col: str) -> Dict[str, any]:
        s = {}
        ser = df[col]
        s['dtype'] = str(ser.dtype)
        s['n_missing'] = int(ser.isna().sum())
        s['pct_missing'] = float(ser.isna().mean())
        if pd.api.types.is_numeric_dtype(ser):
            s.update({
                'mean': float(ser.mean()),
                'median': float(ser.median()),
                'std': float(ser.std()),
                'skew': float(ser.skew()),
                'min': float(ser.min()),
                'max': float(ser.max())
            })
        else:
            vc = ser.value_counts().head(10)
            s['top_values'] = vc.to_dict()
            s['n_unique'] = int(ser.nunique())
        return s

    def generate_insights(self, df: pd.DataFrame, col: str) -> List[str]:
        insights = []
        ser = df[col]
        null_pct = ser.isna().mean()
        if null_pct > 0.05:
            insights.append(f"⚠️ Alta taxa de nulos: {null_pct:.1%} — considerar imputação/remocao")
        if pd.api.types.is_numeric_dtype(ser):
            skew = ser.dropna().skew()
            if abs(skew) > 1:
                insights.append(f"📈 Assimetria forte (skew={skew:.2f}). Pense em transformações log/power.")
            # IQR outliers
            q1, q3 = ser.dropna().quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ser[(ser < (q1 - 1.5*iqr)) | (ser > (q3 + 1.5*iqr))].shape[0]
            if outliers > 0:
                insights.append(f"🚨 {outliers} outliers detectados (IQR). Verificar tratamento.")
        else:
            if ser.nunique() > 50:
                insights.append(f"🔢 Alta cardinalidade: {ser.nunique()} valores — cuidado com one-hot.")
            top = ser.value_counts(normalize=True).iloc[0]
            if top > 0.75:
                top_label = ser.value_counts().index[0]
                insights.append(f"⚖️ Desbalanceamento: '{top_label}' representa {top:.1%} dos registros.")
        return insights

# --------------------- Time Series Engine ---------------------
class TimeSeriesEngine:
    def auto_arima_lite(self, ts: pd.Series, max_p=2, max_d=1, max_q=2):
        best_aic = np.inf
        best = None
        for p in range(max_p+1):
            for d in range(max_d+1):
                for q in range(max_q+1):
                    try:
                        m = ARIMA(ts, order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best = (p,d,q,m)
                    except Exception:
                        continue
        return best

    def forecast(self, df: pd.DataFrame, date_col: str, val_col: str, steps: int = 30):
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
        df2 = df2.dropna(subset=[date_col])
        ts = df2.sort_values(date_col).set_index(date_col)[val_col]
        ts = ts.asfreq(pd.infer_freq(ts.index) or 'D').fillna(method='ffill')
        best = self.auto_arima_lite(ts)
        if best is None:
            raise RuntimeError('Nenhum modelo ARIMA convergiu')
        p,d,q,m = best
        forecast_res = m.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
        ci = forecast_res.conf_int()
        return ts, mean, ci, (p,d,q)

# --------------------- AutoML Engine ---------------------
class AutoMLEngine:
    def __init__(self, random_state:int=42):
        self.random_state = random_state

    def build_preprocessor(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        steps = []
        transformers = []
        if num_cols:
            transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols))
        preproc = ColumnTransformer(transformers, remainder='drop')
        return preproc

    def train(self, df: pd.DataFrame, target: str, features: List[str], algo: str = 'Random Forest', is_reg: bool = True, use_fs: bool = False):
        X = df[features]
        y = df[target]
        preproc = self.build_preprocessor(X)
        steps = [('preproc', preproc)]
        if use_fs:
            k = min(20, len(features))
            func = f_regression if is_reg else f_classif
            steps.append(('fs', SelectKBest(score_func=func, k=k)))
        if is_reg:
            if algo == 'Random Forest': clf = RandomForestRegressor(n_jobs=-1, random_state=self.random_state)
            else: clf = HistGradientBoostingRegressor(random_state=self.random_state)
        else:
            if algo == 'Random Forest': clf = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
            else: clf = HistGradientBoostingClassifier(random_state=self.random_state)
        steps.append(('clf', clf))
        pipe = Pipeline(steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        imp = None
        try:
            res = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=self.random_state, n_jobs=-1)
            imp = pd.DataFrame({ 'feature': X_test.columns, 'importance': res.importances_mean }).sort_values('importance', ascending=False)
        except Exception:
            imp = None
        return pipe, X_test, y_test, preds, imp

# --------------------- PDF Report ---------------------
def generate_pdf(df: pd.DataFrame, kpis: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Helvetica','B',16)
    pdf.cell(0,10,'Relatorio Executivo — Enterprise Analytics', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,8,f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    # KPIs
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,40,190,30,'F')
    pdf.set_y(45)
    colw = 190/4
    keys = ['rows','cols','nulls','dups']
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    for i,t in enumerate(titles):
        pdf.set_font('Helvetica','B',11)
        pdf.cell(colw,8,t,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(7)
    pdf.set_font('Helvetica','',12)
    for i,k in enumerate(keys):
        pdf.cell(colw,8,str(kpis.get(k,'')),align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(12)
    # short stats
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Resumo Estatistico (Top variaveis numericas):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    desc = df.describe().T.reset_index().head(10)
    if not desc.empty:
        cols = ['index','mean','min','max']
        if set(cols).issubset(desc.columns):
            desc = desc[cols]
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

# --------------------- App UI ---------------------
def main():
    set_theme()
    st.sidebar.title('Enterprise Analytics')
    st.sidebar.caption('Enterprise Ultra — Refatorado')

    # file uploader (prefer uploader; fallback to local path)
    uploaded = st.sidebar.file_uploader('Carregar dataset (CSV/XLSX)', type=['csv','xlsx'])
    use_local = st.sidebar.checkbox('Usar caminho local (dev)', value=False)
    df = pd.DataFrame()
    de = DataEngine()

    if uploaded is None and use_local:
        try:
            df = de.load_data(UPLOADED_FILE_PATH)
        except Exception as e:
            st.sidebar.error('Erro ao carregar caminho local')
    elif uploaded is not None:
        df = de.load_data(uploaded)

    if df is None or df.empty:
        st.title('Enterprise Analytics — Carregue um dataset')
        st.info('Faça upload de um arquivo CSV/Excel na lateral ou marque usar caminho local.')
        return

    # initialize session state values
    if 'df_work' not in st.session_state:
        st.session_state['df_work'] = df.copy()
    if 'sql_history' not in st.session_state:
        st.session_state['sql_history'] = []

    df_work = st.session_state['df_work']

    # Sidebar filters
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    date_cols = df_work.select_dtypes(include=['datetime','object']).columns.tolist()
    # try convert obvious date columns
    guess_dates = [c for c in date_cols if 'date' in c.lower() or 'data' in c.lower()]
    for c in guess_dates:
        try:
            df_work[c] = pd.to_datetime(df_work[c])
        except Exception:
            pass

    st.sidebar.markdown('---')
    menu = st.sidebar.radio('Menu', ['Dashboard','SQL Lab','Engenharia','AutoML','TimeSeries','NLP','Clustering','Export'])

    # Dashboard
    if menu == 'Dashboard':
        st.header('Visão Executiva')
        r,c = df_work.shape
        nulls = int(df_work.isna().sum().sum())
        dups = int(df_work.duplicated().sum())
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Registros</h3><h2>{r:,}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Variáveis</h3><h2>{c}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Nulos</h3><h2>{nulls}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Duplicatas</h3><h2>{dups}</h2></div>", unsafe_allow_html=True)

        st.markdown('---')
        st.subheader('Column Explorer')
        col = st.selectbox('Coluna', df_work.columns)
        ie = InsightsEngine()
        with st.expander('Insights automáticos'):
            ins = ie.generate_insights(df_work, col)
            if ins:
                for i in ins: st.write('- ', i)
            else:
                st.write('Nenhum insight automático relevante encontrado.')
        # plots
        left, right = st.columns([2,1])
        if pd.api.types.is_numeric_dtype(df_work[col]):
            fig = px.histogram(df_work, x=col, marginal='box', template='plotly_dark')
            left.plotly_chart(fig, use_container_width=True)
            # summary
            right.write(df_work[col].describe())
        else:
            fig = px.bar(df_work[col].value_counts().head(15).reset_index(), x=col, y='count', template='plotly_dark')
            left.plotly_chart(fig, use_container_width=True)
            right.write(df_work[col].describe())

    # SQL Lab
    elif menu == 'SQL Lab':
        st.header('SQL Playground')
        query = st.text_area('SQL (tabela: dados)', 'SELECT * FROM dados LIMIT 10', height=180)
        if st.button('Executar Query'):
            res, err = de.run_query(df_work, query)
            if err:
                st.error(err)
            else:
                st.session_state['sql_history'].append(query)
                st.success(f'Returns {len(res)} rows')
                st.dataframe(res)
                st.download_button('Download CSV', res.to_csv(index=False).encode('utf-8'), 'sql_result.csv')
        if st.session_state['sql_history']:
            with st.expander('Histórico de queries'):
                for q in reversed(st.session_state['sql_history'][-10:]):
                    st.code(q, language='sql')

    # Engenharia
    elif menu == 'Engenharia':
        st.header('Engenharia de Dados')
        tab1,tab2,tab3 = st.tabs(['Estrutura','Limpeza','Encoding'])
        with tab1:
            st.subheader('Renomear/Conversão de tipos')
            col = st.selectbox('Coluna', df_work.columns)
            new = st.text_input('Novo nome', value=col)
            if st.button('Renomear'):
                df_work.rename(columns={col:new}, inplace=True)
                st.success('Coluna renomeada')
            # tipagem
            typ_col = st.selectbox('Converter tipo:', df_work.columns)
            to = st.selectbox('Para', ['Numérico','Data','Texto'])
            if st.button('Converter tipo'):
                try:
                    if to=='Numérico': df_work[typ_col] = pd.to_numeric(df_work[typ_col], errors='coerce')
                    elif to=='Data': df_work[typ_col] = pd.to_datetime(df_work[typ_col], errors='coerce')
                    else: df_work[typ_col] = df_work[typ_col].astype(str)
                    st.success('Conversão aplicada')
                except Exception as e:
                    st.error(f'Erro: {e}')
        with tab2:
            st.subheader('Nulos e Duplicatas')
            if st.button('Remover duplicatas'):
                df_work.drop_duplicates(inplace=True)
                st.success('Duplicatas removidas')
            col_null = st.selectbox('Coluna para tratar nulos', df_work.columns)
            m = st.selectbox('Método', ['Drop','Média','Mediana','Zero'])
            if st.button('Aplicar tratamento de nulos'):
                if m=='Drop': df_work.dropna(subset=[col_null], inplace=True)
                elif m=='Média': df_work[col_null].fillna(df_work[col_null].mean(), inplace=True)
                elif m=='Mediana': df_work[col_null].fillna(df_work[col_null].median(), inplace=True)
                else: df_work[col_null].fillna(0, inplace=True)
                st.success('Tratamento aplicado')
        with tab3:
            st.subheader('Encoding')
            cat_cols = df_work.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                c = st.selectbox('One-Hot coluna', cat_cols)
                if st.button('Gerar Dummies'):
                    df_work = pd.get_dummies(df_work, columns=[c], drop_first=True, dtype=int)
                    st.success('Dummies criadas')
            else:
                st.info('Nenhuma coluna categórica identificada')
        st.session_state['df_work'] = df_work

    # AutoML
    elif menu == 'AutoML':
        st.header('AutoML Pro')
        automl = AutoMLEngine()
        tgt = st.selectbox('Target', df_work.columns)
        feats = st.multiselect('Features', [c for c in df_work.columns if c!=tgt])
        use_fs = st.checkbox('Usar Feature Selection (SelectKBest)')
        if st.button('Treinar') and feats:
            is_reg = pd.api.types.is_numeric_dtype(df_work[tgt]) and df_work[tgt].nunique()>20
            algo = st.selectbox('Algoritmo',['Random Forest','Hist Gradient Boosting'])
            with st.spinner('Treinando...'):
                pipe, X_test, y_test, preds, imp = automl.train(df_work, tgt, feats, algo, is_reg, use_fs)
                if is_reg:
                    st.metric('R2', f"{r2_score(y_test,preds):.4f}")
                    st.metric('MAE', f"{mean_absolute_error(y_test,preds):.4f}")
                else:
                    st.metric('Acuracia', f"{accuracy_score(y_test,preds):.2%}")
                    st.plotly_chart(px.imshow(confusion_matrix(y_test,preds), text_auto=True, template='plotly_dark'))
                if imp is not None:
                    st.subheader('Importancia (Permutation)')
                    st.dataframe(imp.head(30))
                # export
                b = BytesIO()
                joblib.dump(pipe, b)
                st.download_button('Baixar Modelo (.joblib)', b.getvalue(), 'model.joblib')

    # TimeSeries
    elif menu == 'TimeSeries':
        st.header('Time Series Studio')
        tse = TimeSeriesEngine()
        date_col = st.selectbox('Coluna de data', df_work.columns)
        val_col = st.selectbox('Coluna de valor', df_work.select_dtypes(include=np.number).columns)
        steps = st.slider('Dias para prever', 7, 90, 30)
        if st.button('Gerar previsao'):
            try:
                ts, mean, ci, order = tse.forecast(df_work, date_col, val_col, steps)
                st.success(f'ARIMA{order}')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Hist'))
                fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name='Forecast'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f'Erro TS: {e}')

    # NLP
    elif menu == 'NLP':
        st.header('NLP Studio')
        txt_col = st.selectbox('Coluna texto', df_work.select_dtypes(include='object').columns)
        if st.button('Analisar texto'):
            vec = TfidfVectorizer(stop_words='english', max_features=30)
            X = vec.fit_transform(df_work[txt_col].astype(str))
            words = dict(zip(vec.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
            st.bar_chart(pd.Series(words).sort_values(ascending=False).head(20))

    # Clustering
    elif menu == 'Clustering':
        st.header('Clusterização')
        numeric = df_work.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect('Features', numeric)
        k = st.slider('K', 2, 15, 3)
        if st.button('Executar cluster') and features:
            X = StandardScaler().fit_transform(df_work[features].dropna())
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            pca = PCA(2).fit_transform(X)
            fig = px.scatter(x=pca[:,0], y=pca[:,1], color=labels.astype(str), template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    # Export
    elif menu == 'Export':
        st.header('Export')
        st.download_button('Baixar CSV', df_work.to_csv(index=False).encode('utf-8'), 'dados.csv')
        kpis = {'rows': len(df_work), 'cols': df_work.shape[1], 'nulls': int(df_work.isna().sum().sum()), 'dups': int(df_work.duplicated().sum())}
        if st.button('Gerar PDF Executivo'):
            pdf_bytes = generate_pdf(df_work, kpis)
            st.download_button('Download PDF', pdf_bytes, 'report.pdf')

    # persist work df
    st.session_state['df_work'] = df_work

if __name__ == '__main__':
    main()
