import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib
import re
import unicodedata
import logging
from io import BytesIO
from datetime import datetime

# --- CIENTÍFICO & ESTATÍSTICA ---
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# --- MACHINE LEARNING ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

# --- PDF ---
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --- CONFIGURAÇÃO ---
st.set_page_config(
    page_title="Enterprise Analytics Ultra", 
    layout="wide", 
    page_icon="💠", 
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)

# ==============================================================================
# 🎨 TEMA & UI (PREMIUM)
# ==============================================================================
def set_theme():
    card_bg = "#1E1E1E"
    text_c = "#FAFAFA"
    accent = "#00CC96"
    gold = "#E1C16E"
    
    css = f"""
    <style>
        .metric-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid {accent};
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 15px;
        }}
        .metric-card h3 {{ margin: 0; font-size: 0.9rem; color: #AAA; text-transform: uppercase; letter-spacing: 1px; }}
        .metric-card h2 {{ margin: 5px 0; font-size: 2rem; color: {text_c}; font-weight: 700; }}
        
        [data-baseweb="tag"] {{ background-color: {gold} !important; color: #111 !important; }}
        
        h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{ background-color: #111; border-radius: 4px 4px 0 0; }}
        .stTabs [aria-selected="true"] {{ background-color: {card_bg}; border-top: 2px solid {accent}; }}
        
        .insight-box {{
            background-color: rgba(0, 204, 150, 0.05);
            border: 1px solid {accent};
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        .insight-title {{ color: {accent}; font-weight: bold; margin-bottom: 5px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# 🧠 DATA ENGINE & SQL
# ==============================================================================
class DataEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_data(file):
        try:
            if file.name.endswith("csv"): return pd.read_csv(file)
            return pd.read_excel(file)
        except Exception as e: return None

    @staticmethod
    def run_query(df, query):
        conn = sqlite3.connect(':memory:')
        df.to_sql('dados', conn, index=False, if_exists='replace')
        try:
            result = pd.read_sql_query(query, conn)
            return result, None
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()

# ==============================================================================
# 💡 AUTO INSIGHTS ENGINE (PREMIUM)
# ==============================================================================
class InsightsEngine:
    def generate_column_insights(self, df, col):
        insights = []
        
        # Nulos
        null_pct = df[col].isna().mean()
        if null_pct > 0.05:
            insights.append(f"⚠️ **Dados Faltantes:** {null_pct:.1%} nulos. Acima do ideal (5%).")
        
        # Numéricos
        if pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].skew()
            if abs(skew) > 1:
                insights.append(f"📈 **Assimetria:** Skewness {skew:.2f}. Considere log/power transform.")
            
            # Outliers (Z-Score > 3)
            z_scores = stats.zscore(df[col].dropna())
            outliers = (np.abs(z_scores) > 3).sum()
            if outliers > 0:
                insights.append(f"🚨 **Outliers (Z-Score):** {outliers} valores extremos (>3 sigma).")
            
            # Correlação
            df_num = df.select_dtypes(include=np.number)
            if len(df_num.columns) > 1:
                corrs = df_num.corr()[col].drop(col)
                high_corr = corrs[abs(corrs) > 0.85]
                for c_name, c_val in high_corr.items():
                    insights.append(f"🔗 **Correlação Forte:** {c_val:.2f} com '{c_name}'. Risco de multicolinearidade.")
        
        # Categóricos
        else:
            n_unique = df[col].nunique()
            if n_unique > 50:
                insights.append(f"🔢 **Alta Cardinalidade:** {n_unique} categorias. One-Hot pode ser custoso.")
            
            if n_unique == 1:
                insights.append("🛑 **Variância Zero:** Coluna constante. Pode ser removida.")
                
            top_val = df[col].value_counts(normalize=True).iloc[0]
            if top_val > 0.8:
                insights.append(f"⚖️ **Desbalanceamento:** '{df[col].value_counts().index[0]}' domina {top_val:.1%} dos dados.")
                
        return insights

# ==============================================================================
# 🔮 TIME SERIES FORECAST PRO (AUTO-ARIMA)
# ==============================================================================
class TimeSeriesEngine:
    def auto_forecast(self, df, date_col, val_col, horizon=30):
        # Prep
        ts = df.set_index(pd.to_datetime(df[date_col]))[val_col].sort_index()
        ts = ts.asfreq(pd.infer_freq(ts.index) or 'D')
        ts = ts.fillna(method='ffill')
        
        # Grid Search ARIMA Lite
        best_aic = float('inf')
        best_order = (1,1,1)
        best_model = None
        
        # Grid reduzido para performance em tempo real
        for p in [0, 1, 2]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        model = ARIMA(ts, order=(p,d,q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p,d,q)
                            best_model = model
                    except: continue
                    
        if best_model is None: return None, None, None, None
        
        # Forecast
        forecast_res = best_model.get_forecast(steps=horizon)
        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
        
        return ts, pred_mean, conf_int, best_order

# ==============================================================================
# 🤖 AUTOML ENGINE (FEATURE SELECTION + MODELOS AVANÇADOS)
# ==============================================================================
class AutoMLEngine:
    def train(self, df, target, features, algo, is_reg, use_fs=False):
        X = df[features]
        y = df[target]
        
        # Feature Selection
        fs_step = []
        if use_fs:
            k = min(15, len(features))
            # Seleciona K-Best baseado em F-Score
            fs_step = [('fs', SelectKBest(score_func=f_regression if is_reg else f_classif, k=k))]
        
        # Preprocessor
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns
        
        preproc = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
        
        # Modelo
        if is_reg:
            models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "Hist Gradient Boosting": HistGradientBoostingRegressor() # LightGBM inspired
            }
        else:
            models = {
                "Random Forest": RandomForestClassifier(n_jobs=-1),
                "Hist Gradient Boosting": HistGradientBoostingClassifier()
            }
            
        # Pipeline Completo
        steps = [('pre', preproc)] + fs_step + [('clf', models[algo])]
        pipeline = Pipeline(steps)
        
        # Treino
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        # Feature Importance (Permutation - Agnóstico a modelo)
        try:
            result = permutation_importance(pipeline, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
            imp = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean}).sort_values('Importance', ascending=False)
        except: imp = None
            
        return pipeline, X_test, y_test, preds, imp

# ==============================================================================
# 📄 PDF REPORT ENGINE
# ==============================================================================
def generate_pdf_report(df, kpis):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Executivo de Analise", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 35, 190, 25, 'F')
    pdf.set_y(40)
    
    pdf.set_font("Helvetica", "B", 12)
    col_w = 190 / 4
    
    headers = ["Total Linhas", "Total Colunas", "Dados Nulos", "Duplicatas"]
    values = [f"{kpis['rows']:,}", f"{kpis['cols']}", f"{kpis['nulls']}", f"{kpis['dups']}"]
    
    for i, h in enumerate(headers):
        align = XPos.RIGHT if i < 3 else XPos.LMARGIN
        newy = YPos.TOP if i < 3 else YPos.NEXT
        pdf.cell(col_w, 8, h, align='C', new_x=align, new_y=newy)
    
    pdf.set_font("Helvetica", "", 12)
    for i, v in enumerate(values):
        align = XPos.RIGHT if i < 3 else XPos.LMARGIN
        newy = YPos.TOP if i < 3 else YPos.NEXT
        pdf.cell(col_w, 8, v, align='C', new_x=align, new_y=newy)
    
    pdf.ln(20)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Resumo Estatistico (Top Variáveis):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    desc = df.describe().T.head(10).reset_index()
    target_cols = ['index', 'mean', 'min', 'max']
    if set(target_cols).issubset(desc.columns):
        desc = desc[target_cols]
        
        pdf.set_font("Helvetica", "B", 10)
        w = [70, 40, 40, 40]
        for i, c in enumerate(desc.columns):
            pdf.cell(w[i], 8, c, 1, align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
            
        pdf.set_font("Helvetica", "", 9)
        for _, row in desc.iterrows():
            for i, item in enumerate(row):
                txt = str(item)[:25]
                if isinstance(item, float): txt = f"{item:.2f}"
                safe_txt = txt.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(w[i], 7, safe_txt, 1, align='C' if i>0 else 'L', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
            
    return pdf.output()

# ==============================================================================
# 🖥️ APP PRINCIPAL
# ==============================================================================
def main():
    set_theme()
    
    # --- SIDEBAR ---
    st.sidebar.markdown("""
    <div style="background-color:#00CC96; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
        <h3 style="color:white; margin:0; font-weight:bold;">Enterprise Ultra</h3>
        <p style="color:#EEE; font-size:0.8rem;">Analytics v21.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("📂 Carregar Dataset", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.title("Enterprise Analytics Ultra")
        st.markdown("""
        ### A Ferramenta Definitiva.
        
        **Módulos Premium:**
        * **🔍 Column Explorer:** Raio-X completo de variáveis com Insights Automáticos.
        * **🔮 Forecast Pro:** Previsão de Séries Temporais com Auto-ARIMA e Intervalo de Confiança.
        * **🤖 AutoML Pro:** Seleção automática de features e modelos Gradient Boosting.
        * **💾 SQL Premium:** Histórico de queries e ambiente SQL completo.
        
        👈 **Carregue um arquivo para começar.**
        """)
        return

    de = DataEngine()
    if 'df_raw' not in st.session_state or st.session_state.get('fname') != uploaded_file.name:
        df = de.load_data(uploaded_file)
        if df is not None:
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy()
            st.session_state['fname'] = uploaded_file.name
            st.session_state['sql_history'] = []
            st.toast("Dataset carregado com sucesso!", icon="✅")
    
    df_full = st.session_state['df_work']
    
    # Filtros Globais
    st.sidebar.header("🔍 Filtros Globais")
    date_cols = df_full.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    if not date_cols:
        for c in df_full.select_dtypes(include='object').columns:
            if 'date' in c.lower() or 'data' in c.lower():
                try:
                    df_full[c] = pd.to_datetime(df_full[c])
                    date_cols.append(c)
                except: pass
                
    if date_cols:
        dt_col = st.sidebar.selectbox("Data:", ["Nenhum"] + date_cols)
        if dt_col != "Nenhum":
            min_d, max_d = df_full[dt_col].min(), df_full[dt_col].max()
            d_range = st.sidebar.date_input("Período:", [min_d, max_d])
            if len(d_range) == 2:
                df_full = df_full[(df_full[dt_col].dt.date >= d_range[0]) & (df_full[dt_col].dt.date <= d_range[1])]

    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio("Navegação:", [
        "📊 Dashboard & Explorer", 
        "💾 SQL Lab Premium",
        "🛠️ Engenharia de Dados",
        "🤖 AutoML Pro",
        "🔮 Time Series Forecast",
        "🧠 NLP (Texto)",
        "🌀 Clustering",
        "📤 Relatórios & Export"
    ])

    # --- 1. DASHBOARD & EXPLORER PREMIUM ---
    if menu == "📊 Dashboard & Explorer":
        st.title("Visão Executiva & Exploratória")
        
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"""<div class="metric-card"><h3>Registros</h3><h2>{len(df_full):,}</h2></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="metric-card"><h3>Variáveis</h3><h2>{df_full.shape[1]}</h2></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="metric-card"><h3>Nulos</h3><h2>{df_full.isna().sum().sum()}</h2></div>""", unsafe_allow_html=True)
        with c4: st.markdown(f"""<div class="metric-card"><h3>Duplicatas</h3><h2>{df_full.duplicated().sum()}</h2></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # COLUMN EXPLORER PREMIUM
        st.subheader("🔍 Column Explorer Premium")
        
        col_exp = st.selectbox("Selecione uma Coluna para Raio-X:", df_full.columns)
        
        # Auto Insights
        ie = InsightsEngine()
        insights = ie.generate_column_insights(df_full, col_exp)
        
        if insights:
            st.markdown('<div class="insight-box"><div class="insight-title">💡 Auto Insights</div>', unsafe_allow_html=True)
            for msg in insights:
                st.markdown(f"- {msg}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        c_l, c_r = st.columns([2, 1])
        with c_l:
            if pd.api.types.is_numeric_dtype(df_full[col_exp]):
                fig = px.histogram(df_full, x=col_exp, marginal="box", title=f"Distribuição: {col_exp}", template="plotly_dark")
                fig.update_traces(marker_color="#00CC96")
                st.plotly_chart(fig, use_container_width=True)
                
                # QQ Plot
                try:
                    qq_x = stats.probplot(df_full[col_exp].dropna(), dist="norm")[0][0]
                    qq_y = stats.probplot(df_full[col_exp].dropna(), dist="norm")[0][1]
                    qq_fig = px.scatter(x=qq_x, y=qq_y, labels={'x':'Teórico', 'y':'Real'}, title="QQ-Plot (Normalidade)", template="plotly_dark")
                    st.plotly_chart(qq_fig, use_container_width=True)
                except: pass
            else:
                top = df_full[col_exp].value_counts().head(10).reset_index()
                top.columns = [col_exp, 'Contagem']
                fig = px.bar(top, x='Contagem', y=col_exp, orientation='h', title=f"Top 10: {col_exp}", template="plotly_dark")
                fig.update_traces(marker_color="#00CC96")
                st.plotly_chart(fig, use_container_width=True)
                
        with c_r:
            st.markdown("##### Estatísticas Detalhadas")
            st.dataframe(df_full[col_exp].describe(), use_container_width=True)

    # --- 2. SQL LAB PREMIUM ---
    elif menu == "💾 SQL Lab Premium":
        st.header("SQL Playground Pro")
        
        c_sql, c_res = st.columns([1, 2])
        with c_sql:
            query = st.text_area("Query (Tabela: 'dados'):", "SELECT * FROM dados LIMIT 10", height=150)
            if st.button("Executar Query"):
                res, err = de.run_query(df_full, query)
                if err: st.error(f"Erro: {err}")
                else: 
                    st.session_state['sql_res'] = res
                    if 'sql_history' not in st.session_state: st.session_state['sql_history'] = []
                    st.session_state['sql_history'].append(query)
            
            if 'sql_history' in st.session_state and st.session_state['sql_history']:
                with st.expander("Histórico de Queries"):
                    for q in reversed(st.session_state['sql_history'][-5:]):
                        st.code(q, language='sql')

        with c_res:
            if 'sql_res' in st.session_state:
                st.success(f"{len(st.session_state['sql_res'])} linhas retornadas.")
                st.dataframe(st.session_state['sql_res'], use_container_width=True)
                csv_sql = st.session_state['sql_res'].to_csv(index=False).encode('utf-8')
                st.download_button("Baixar Resultado (CSV)", csv_sql, "sql_result.csv", "text/csv")
            else:
                st.info("Execute uma query para ver os resultados.")

    # --- 3. ENGENHARIA ---
    elif menu == "🛠️ Engenharia de Dados":
        st.header("Engenharia de Dados")
        t1, t2, t3 = st.tabs(["Estrutura", "Limpeza", "Encoding"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                ren = st.selectbox("Renomear:", df_full.columns)
                new_n = st.text_input("Novo nome:", value=ren)
                if st.button("Renomear"):
                    df_full.rename(columns={ren: new_n}, inplace=True)
                    st.session_state['df_work'] = df_full
                    st.rerun()
            with c2:
                typ = st.selectbox("Converter Tipo:", df_full.columns)
                to_t = st.selectbox("Para:", ["Numérico", "Texto", "Data"])
                if st.button("Converter"):
                    try:
                        if to_t=="Numérico": df_full[typ] = pd.to_numeric(df_full[typ], errors='coerce')
                        elif to_t=="Data": df_full[typ] = pd.to_datetime(df_full[typ], errors='coerce')
                        else: df_full[typ] = df_full[typ].astype(str)
                        st.session_state['df_work'] = df_full
                        st.rerun()
                    except: st.error("Falha na conversão.")
        
        with t2:
            if st.button("Remover Duplicatas"):
                df_full.drop_duplicates(inplace=True)
                st.session_state['df_work'] = df_full
                st.rerun()
            
            c_null = st.selectbox("Tratar Nulos:", df_full.columns)
            met = st.selectbox("Método:", ["Drop", "Média", "Zero"])
            if st.button("Aplicar"):
                if met=="Drop": df_full.dropna(subset=[c_null], inplace=True)
                elif met=="Média": df_full[c_null].fillna(df_full[c_null].mean(), inplace=True)
                else: df_full[c_null].fillna(0, inplace=True)
                st.session_state['df_work'] = df_full
                st.rerun()

        with t3:
            c_dum = st.selectbox("One-Hot (Dummies):", df_full.select_dtypes(include='object').columns)
            if st.button("Criar Dummies"):
                df_full = pd.get_dummies(df_full, columns=[c_dum], drop_first=True, dtype=int)
                st.session_state['df_work'] = df_full
                st.rerun()

    # --- 4. AUTOML PRO ---
    elif menu == "🤖 AutoML Pro":
        st.header("AutoML Pipeline")
        automl = AutoMLEngine()
        
        tgt = st.selectbox("Target:", df_full.columns)
        fts = st.multiselect("Features:", [c for c in df_full.columns if c!=tgt])
        
        use_fs = st.checkbox("Feature Selection Automático (SelectKBest)")
        
        if st.button("Treinar Modelo") and fts:
            is_reg = pd.api.types.is_numeric_dtype(df_full[tgt]) and df_full[tgt].nunique() > 20
            algo = st.selectbox("Algoritmo:", ["Random Forest", "Hist Gradient Boosting (LightGBM)"])
            
            with st.spinner("Treinando Pipeline..."):
                mod, Xt, yt, pred, imp = automl.train(df_full, tgt, fts, algo, is_reg, use_fs)
                
                c1, c2 = st.columns(2)
                if is_reg:
                    c1.metric("R2 Score", f"{r2_score(yt, pred):.4f}")
                    c2.metric("MAE", f"{mean_absolute_error(yt, pred):.4f}")
                    st.plotly_chart(px.scatter(x=yt, y=pred, title="Real vs Previsto", template="plotly_dark"))
                else:
                    c1.metric("Acurácia", f"{accuracy_score(yt, pred):.2%}")
                    st.plotly_chart(px.imshow(confusion_matrix(yt, pred), text_auto=True, template="plotly_dark"))
                
                if imp is not None:
                    st.subheader("Feature Importance")
                    st.bar_chart(imp.set_index('Feature'))
                
                bio = BytesIO()
                joblib.dump(mod, bio)
                st.download_button("Baixar Modelo (.joblib)", bio.getvalue(), "modelo.joblib")

    # --- 5. TIME SERIES FORECAST PRO ---
    elif menu == "🔮 Time Series Forecast":
        st.header("Forecast Pro (Auto-ARIMA)")
        ts_eng = TimeSeriesEngine()
        
        c_dt = st.selectbox("Data:", df_full.columns)
        c_val = st.selectbox("Valor:", df_full.select_dtypes(include=np.number).columns)
        hz = st.slider("Horizonte (Dias):", 7, 60, 30)
        
        if st.button("Gerar Previsão"):
            try:
                with st.spinner("Otimizando modelo ARIMA..."):
                    ts, pred, conf, order = ts_eng.auto_forecast(df_full, c_dt, c_val, hz)
                
                if ts is not None:
                    st.success(f"Melhor Modelo: ARIMA{order}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Histórico"))
                    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Previsão", line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,0], showlegend=False, line=dict(width=0)))
                    fig.add_trace(go.Scatter(x=conf.index, y=conf.iloc[:,1], name="Intervalo Confiança", fill='tonexty', line=dict(width=0), fillcolor='rgba(255,0,0,0.2)'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Falha na convergência do modelo ARIMA.")
            except Exception as e: st.error(f"Erro: {e}")

    # --- 6. NLP ---
    elif menu == "🧠 NLP (Texto)":
        st.header("Text Analytics")
        c_txt = st.selectbox("Texto:", df_full.select_dtypes(include='object').columns)
        if st.button("Analisar"):
            vec = TfidfVectorizer(stop_words='english', max_features=20)
            X = vec.fit_transform(df_full[c_txt].astype(str))
            words = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
            st.bar_chart(pd.Series(words).sort_values(ascending=False))

    # --- 7. CLUSTERING ---
    elif menu == "🌀 Clustering":
        st.header("Clusterização")
        fs = st.multiselect("Features:", df_full.select_dtypes(include=np.number).columns)
        k = st.slider("K:", 2, 10, 3)
        if st.button("Executar") and fs:
            X = StandardScaler().fit_transform(df_full[fs].dropna())
            cl = KMeans(n_clusters=k).fit_predict(X)
            pca = PCA(2).fit_transform(X)
            fig = px.scatter(x=pca[:,0], y=pca[:,1], color=cl.astype(str), title="PCA Clusters", template="plotly_dark")
            st.plotly_chart(fig)

    # --- 8. EXPORT ---
    elif menu == "📤 Relatórios & Export":
        st.header("Exportação")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Baixar CSV", df_full.to_csv(index=False).encode('utf-8'), "dados.csv", "text/csv")
        with c2:
            if st.button("Gerar PDF Executivo"):
                kpis = {"rows": len(df_full), "cols": df_full.shape[1], "nulls": df_full.isna().sum().sum(), "dups": df_full.duplicated().sum()}
                pdf = generate_pdf_report(df_full, kpis)
                st.download_button("Baixar PDF", pdf, "report.pdf", "application/pdf")

if __name__ == "__main__":
    main()