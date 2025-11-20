import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import re
import unicodedata
import logging
from io import BytesIO
from datetime import datetime

# --- CIENTÍFICO & ESTATÍSTICA ---
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram

# --- SKLEARN (O CANIVETE SUÍÇO) ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance

# --- PDF ---
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Data Studio X - Ultimate", layout="wide", page_icon="💠", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# 🎨 UI ENGINE
# ==============================================================================
def set_theme():
    st.markdown("""
    <style>
        .metric-card {
            background-color: #1E1E1E; border: 1px solid #333; padding: 15px; 
            border-radius: 8px; text-align: center; margin-bottom: 10px;
        }
        .metric-card h3 { color: #AAA; font-size: 0.9rem; margin: 0; }
        .metric-card h2 { color: #FFF; font-size: 1.6rem; margin: 5px 0; font-weight: bold; }
        .metric-card .delta { color: #4CAF50; font-size: 0.8rem; }
        h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
        .stTabs [aria-selected="true"] { background-color: #262730; border-bottom: 2px solid #4F8BF9; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 🧠 PROFILING ENGINE (EDA AUTOMÁTICO)
# ==============================================================================
class ProfilingEngine:
    """Gera relatórios automáticos de qualidade e estatística."""
    
    def generate_report(self, df):
        st.subheader("📊 Relatório de Perfilamento de Dados")
        
        # 1. Visão Geral
        c1, c2, c3, c4 = st.columns(4)
        n_dups = df.duplicated().sum()
        n_cols_const = [c for c in df.columns if df[c].nunique() <= 1]
        
        with c1: st.markdown(f"""<div class="metric-card"><h3>Variáveis</h3><h2>{df.shape[1]}</h2></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="metric-card"><h3>Observações</h3><h2>{df.shape[0]}</h2></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="metric-card"><h3>Células Vazias</h3><h2>{df.isna().sum().sum()}</h2></div>""", unsafe_allow_html=True)
        with c4: st.markdown(f"""<div class="metric-card"><h3>Duplicatas</h3><h2>{n_dups}</h2></div>""", unsafe_allow_html=True)

        # 2. Alertas de Qualidade
        with st.expander("🚨 Alertas de Qualidade", expanded=True):
            if n_dups > 0: st.error(f"**Duplicatas:** Existem {n_dups} linhas idênticas.")
            if n_cols_const: st.warning(f"**Variância Zero:** As colunas {n_cols_const} possuem um único valor.")
            
            # Cardinalidade Alta
            high_card = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() > 50]
            if high_card: st.warning(f"**Alta Cardinalidade:** Colunas categóricas com muitos valores únicos: {high_card}")
            
            # Correlação Alta
            df_num = df.select_dtypes(include=np.number)
            if not df_num.empty and df_num.shape[1] > 1:
                corr_matrix = df_num.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
                if high_corr: st.info(f"**Multicolinearidade:** Variáveis altamente correlacionadas (>0.95): {high_corr}")

        # 3. Estatísticas Detalhadas
        t1, t2 = st.tabs(["Numérico", "Categórico"])
        with t1:
            if not df_num.empty:
                desc = df_num.describe().T
                desc['skew'] = df_num.skew()
                desc['kurtosis'] = df_num.kurtosis()
                desc['missing_pct'] = df_num.isna().mean() * 100
                st.dataframe(desc.style.background_gradient(cmap='Blues'), use_container_width=True)
        with t2:
            df_cat = df.select_dtypes(exclude=np.number)
            if not df_cat.empty:
                st.dataframe(df_cat.describe().T, use_container_width=True)

        # 4. Correlações Heatmap
        if not df_num.empty and df_num.shape[1] > 1:
            st.subheader("🔥 Mapa de Calor de Correlação")
            fig = px.imshow(df_num.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 🛠️ FEATURE ENGINEERING PRO
# ==============================================================================
class FeatureEngineeringEngine:
    
    def apply_transformation(self, df, col, method):
        try:
            # Reshape para 2D array que o sklearn exige
            data = df[[col]].values
            
            if method == "Log (Log1p)":
                return np.log1p(data)
            elif method == "StandardScaler (Z-Score)":
                return StandardScaler().fit_transform(data)
            elif method == "MinMax (0-1)":
                return MinMaxScaler().fit_transform(data)
            elif method == "RobustScaler (Outliers)":
                return RobustScaler().fit_transform(data)
            elif method == "PowerTransformer (Yeo-Johnson)":
                return PowerTransformer(method='yeo-johnson').fit_transform(data)
            elif method == "QuantileTransformer (Normal)":
                return QuantileTransformer(output_distribution='normal').fit_transform(data)
            elif method == "Binning (qcut - 4)":
                return pd.qcut(df[col], q=4, labels=False).values.reshape(-1, 1)
            else:
                return data
        except Exception as e:
            st.error(f"Erro na transformação: {e}")
            return df[[col]].values

# ==============================================================================
# 🔮 TIME SERIES PRO (ARIMA + FORECAST GRID)
# ==============================================================================
class TimeSeriesPro:
    
    def auto_arima_grid(self, series, p_values=[0,1,2], d_values=[0,1], q_values=[0,1,2]):
        """Simulação leve de Auto-ARIMA usando GridSearch manual"""
        best_aic = float("inf")
        best_order = None
        best_model = None
        
        # Loop simples para encontrar melhor ordem
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(series, order=(p,d,q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p,d,q)
                            best_model = model
                    except: continue
        return best_model, best_order

    def forecast_pro(self, df, date_col, val_col, horizons=[7, 30]):
        df = df.sort_values(date_col)
        ts = df.set_index(date_col)[val_col].asfreq(pd.infer_freq(df[date_col]) or 'D')
        ts = ts.fillna(method='ffill') # Tratar buracos
        
        st.subheader("Análise de Decomposição")
        res = seasonal_decompose(ts, model='additive', extrapolate_trend='freq')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=res.observed, name='Original'))
        fig.add_trace(go.Scatter(x=ts.index, y=res.trend, name='Tendência', line=dict(width=4)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔮 Previsão Automática (Auto-ARIMA Lite)")
        
        # Train/Test split visual
        train_size = int(len(ts) * 0.85)
        train, test = ts[:train_size], ts[train_size:]
        
        with st.spinner("Buscando melhores parâmetros ARIMA..."):
            model, order = self.auto_arima_grid(train)
            
        st.success(f"Melhor Modelo Encontrado: ARIMA{order} (AIC: {model.aic:.1f})")
        
        # Forecast
        forecast_res = model.get_forecast(len(test) + max(horizons))
        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
        
        # Plot
        fig_f = go.Figure()
        # Histórico
        fig_f.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Histórico Real', line=dict(color='gray')))
        # Previsão Teste
        fig_f.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, name='Previsão Modelo', line=dict(color='red')))
        # Intervalo de Confiança
        fig_f.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:,0], line=dict(width=0), showlegend=False))
        fig_f.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:,1], fill='tonexty', line=dict(width=0), fillcolor='rgba(255,0,0,0.2)', name='Intervalo Confiança'))
        
        st.plotly_chart(fig_f, use_container_width=True)

# ==============================================================================
# 🤖 AUTOML AVANÇADO (COM EXPLAINABILITY)
# ==============================================================================
class AutoMLPro:
    
    def train_model(self, X, y, algo_name, is_reg):
        # 1. Preprocessor Inteligente
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()) # RobustScaler lida melhor com outliers
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_pipe, X.select_dtypes(include=np.number).columns),
            ('cat', cat_pipe, X.select_dtypes(exclude=np.number).columns)
        ])
        
        # 2. Seleção de Modelo (Incluindo HistGradientBoosting = LightGBM nativo)
        if is_reg:
            models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "Gradient Boosting (Sklearn)": GradientBoostingRegressor(),
                "Hist Gradient Boosting (LightGBM style)": HistGradientBoostingRegressor(),
                "SVM (SVR)": SVR(),
                "Ridge Regression": Ridge()
            }
        else:
            models = {
                "Random Forest": RandomForestClassifier(n_jobs=-1),
                "Gradient Boosting (Sklearn)": GradientBoostingClassifier(),
                "Hist Gradient Boosting (LightGBM style)": HistGradientBoostingClassifier(),
                "SVM (SVC)": SVC(probability=True),
                "KNN": KNeighborsClassifier()
            }
            
        clf = models[algo_name]
        pipeline = Pipeline([('pre', preprocessor), ('model', clf)])
        
        # 3. Treino
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        # 4. Explainability (Permutation Importance)
        # Funciona para qualquer modelo, sem precisar do pacote SHAP
        with st.spinner("Calculando Importância das Variáveis..."):
            result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False).head(15)
        
        return pipeline, preds, y_test, importance_df

# ==============================================================================
# 🌀 CLUSTERING PRO
# ==============================================================================
class ClusteringPro:
    
    def auto_clustering(self, df, features, n_clusters=3, method='K-Means'):
        X = df[features].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        
        if method == 'K-Means':
            model = KMeans(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            # Elbow Method Visual
            inertias = []
            K = range(1, 10)
            for k in K:
                inertias.append(KMeans(n_clusters=k).fit(X_scaled).inertia_)
            
            fig_elbow = px.line(x=list(K), y=inertias, markers=True, title='Método Elbow (Cotovelo)', labels={'x':'K', 'y':'Inércia'})
            
        elif method == 'DBSCAN':
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            score = -1 if len(set(labels)) < 2 else silhouette_score(X_scaled, labels)
            fig_elbow = None
            
        elif method == 'Agglomerative (Hierarchical)':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            # Dendrograma (Simulado com linkage do scipy)
            Z = linkage(X_scaled[:50], 'ward') # Amostra para performance
            fig_elbow = ff.create_dendrogram(X_scaled[:50])
            fig_elbow.update_layout(title="Dendrograma (Amostra)")

        # PCA 3D
        pca = PCA(n_components=3).fit_transform(X_scaled)
        fig_3d = px.scatter_3d(x=pca[:,0], y=pca[:,1], z=pca[:,2], color=labels.astype(str), title=f"Clusters 3D (Silhouette: {score:.3f})")
        
        return labels, fig_3d, fig_elbow

# ==============================================================================
# 📄 PDF EXPORT ENGINE
# ==============================================================================
def create_pdf(df, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Data Studio X", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "", 10)
    for line in report_text.split('\n'):
        safe_line = str(line).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, safe_line)
        
    return bytes(pdf.output())

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================
def main():
    set_theme()
    
    # --- SIDEBAR ---
    st.sidebar.markdown("""
    <div style="background-color:#4F8BF9; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
        <h2 style="color:white; margin:0; font-size: 1.4rem;">Data Studio<br><span style="font-size:0.8rem; opacity:0.8">ULTIMATE EDITION v12</span></h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("📂 Carregar Dados", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.title("Bem-vindo ao Data Studio")
        st.markdown("### A Plataforma Definitiva de Inteligência de Dados")
        st.info("Carregue um arquivo CSV ou Excel para desbloquear os módulos.")
        return

    # Load & Session
    @st.cache_data(show_spinner=False)
    def get_data(file):
        if file.name.endswith('.csv'): return pd.read_csv(file)
        return pd.read_excel(file)

    if 'df_raw' not in st.session_state or st.session_state.get('fname') != uploaded_file.name:
        st.session_state['df_raw'] = get_data(uploaded_file)
        st.session_state['df_work'] = st.session_state['df_raw'].copy()
        st.session_state['fname'] = uploaded_file.name
        st.session_state['history'] = []
        st.toast("Dados carregados com sucesso!", icon="🚀")

    df_work = st.session_state['df_work']
    
    # Navigation
    menu = st.sidebar.radio("Workstation:", [
        "🏠 Dashboard & Profiler",
        "🧼 Data Cleaning & Quality",
        "🧱 Feature Engineering PRO",
        "⚗️ Chemometrics & Stats",
        "🤖 AutoML & Explainability",
        "🔮 Time Series Studio",
        "🧠 NLP Studio",
        "🌀 Clustering PRO",
        "📤 Relatórios & Export"
    ])
    
    st.sidebar.info(f"Memória: {df_work.memory_usage(deep=True).sum()/1024**2:.1f} MB")

    # --- 1. DASHBOARD & PROFILER ---
    if menu == "🏠 Dashboard & Profiler":
        st.title("Visão 360º dos Dados")
        profiler = ProfilingEngine()
        profiler.generate_report(df_work)
        
    # --- 2. CLEANING & QUALITY ---
    elif menu == "🧼 Data Cleaning & Quality":
        st.header("Centro de Qualidade de Dados")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tratamento de Duplicatas")
            if st.button("Remover Duplicatas Exatas"):
                df_work.drop_duplicates(inplace=True)
                st.session_state['df_work'] = df_work
                st.success("Duplicatas removidas!")
                st.rerun()
                
        with c2:
            st.subheader("Tratamento de Nulos")
            cols_null = df_work.columns[df_work.isna().any()].tolist()
            if cols_null:
                sel_col = st.selectbox("Coluna Alvo:", cols_null)
                method = st.selectbox("Método:", ["Drop Rows", "Média", "Mediana", "Moda", "Constante (0)", "Forward Fill"])
                if st.button("Aplicar Correção"):
                    if method == "Drop Rows": df_work.dropna(subset=[sel_col], inplace=True)
                    elif method == "Média": df_work[sel_col].fillna(df_work[sel_col].mean(), inplace=True)
                    elif method == "Mediana": df_work[sel_col].fillna(df_work[sel_col].median(), inplace=True)
                    elif method == "Moda": df_work[sel_col].fillna(df_work[sel_col].mode()[0], inplace=True)
                    elif method == "Constante (0)": df_work[sel_col].fillna(0, inplace=True)
                    elif method == "Forward Fill": df_work[sel_col].fillna(method='ffill', inplace=True)
                    st.session_state['df_work'] = df_work
                    st.rerun()
            else:
                st.success("Sem dados faltantes.")

    # --- 3. FEATURE ENGINEERING PRO ---
    elif menu == "🧱 Feature Engineering PRO":
        st.header("Engenharia de Atributos Avançada")
        fe = FeatureEngineeringEngine()
        
        st.subheader("Transformação de Distribuição")
        col_trans = st.selectbox("Coluna Numérica:", df_work.select_dtypes(include=np.number).columns)
        method_trans = st.selectbox("Técnica:", [
            "Log (Log1p)", "StandardScaler (Z-Score)", "MinMax (0-1)", 
            "RobustScaler (Outliers)", "PowerTransformer (Yeo-Johnson)", 
            "QuantileTransformer (Normal)", "Binning (qcut - 4)"
        ])
        
        col1, col2 = st.columns(2)
        with col1: 
            st.caption("Original")
            st.plotly_chart(px.histogram(df_work, x=col_trans), use_container_width=True)
            
        if st.button("Aplicar Transformação"):
            new_vals = fe.apply_transformation(df_work, col_trans, method_trans)
            new_col_name = f"{col_trans}_{method_trans.split()[0]}"
            df_work[new_col_name] = new_vals
            st.session_state['df_work'] = df_work
            
            with col2:
                st.caption("Transformado")
                st.plotly_chart(px.histogram(x=new_vals.flatten()), use_container_width=True)
            st.success(f"Coluna criada: {new_col_name}")

    # --- 4. CHEMOMETRICS ---
    elif menu == "⚗️ Chemometrics & Stats":
        st.header("Quimiometria & Estatística Experimental")
        t1, t2 = st.tabs(["PCA & PLS", "Testes Estatísticos"])
        
        with t1:
            st.subheader("Modelagem Multivariada")
            model_type = st.radio("Método:", ["PCA (Exploratório)", "PLS (Regressão)"])
            feat_cols = st.multiselect("Variáveis X:", df_work.select_dtypes(include=np.number).columns)
            
            if model_type == "PCA (Exploratório)" and len(feat_cols) > 1:
                if st.button("Gerar PCA"):
                    pca = PCA(n_components=2)
                    comps = pca.fit_transform(StandardScaler().fit_transform(df_work[feat_cols].dropna()))
                    fig = px.scatter(x=comps[:,0], y=comps[:,1], title=f"PCA Score Plot (Var: {sum(pca.explained_variance_ratio_):.2%})", labels={'x':'PC1', 'y':'PC2'})
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif model_type == "PLS (Regressão)":
                target_col = st.selectbox("Target Y:", df_work.select_dtypes(include=np.number).columns)
                if st.button("Calcular PLS") and len(feat_cols) > 0:
                    pls = PLSRegression(n_components=2)
                    X = df_work[feat_cols].dropna()
                    y = df_work.loc[X.index, target_col]
                    pls.fit(X, y)
                    y_pred = pls.predict(X)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("R2 Score", f"{r2_score(y, y_pred):.4f}")
                    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
                    
                    fig = px.scatter(x=y, y=y_pred.flatten(), labels={'x':'Real', 'y':'Previsto'}, title="Real vs Previsto (PLS)")
                    fig.add_shape(type="line", line=dict(dash="dash", color="red"), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                    st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.subheader("Testes de Hipótese")
            test_type = st.selectbox("Teste:", ["Teste T (2 amostras)", "ANOVA (1 fator)", "Shapiro-Wilk (Normalidade)"])
            col_val = st.selectbox("Variável Numérica:", df_work.select_dtypes(include=np.number).columns)
            
            if test_type == "Teste T (2 amostras)":
                col_grp = st.selectbox("Grupo:", df_work.columns)
                if st.button("Rodar Teste T"):
                    grps = df_work[col_grp].unique()
                    if len(grps) == 2:
                        g1 = df_work[df_work[col_grp]==grps[0]][col_val]
                        g2 = df_work[df_work[col_grp]==grps[1]][col_val]
                        s, p = stats.ttest_ind(g1.dropna(), g2.dropna())
                        st.info(f"P-Valor: {p:.5f} | {'Significativo' if p<0.05 else 'Não Significativo'}")
                    else: st.error("Necessário exatamente 2 grupos.")
            elif test_type == "ANOVA (1 fator)":
                col_grp = st.selectbox("Fator:", df_work.columns)
                if st.button("Rodar ANOVA"):
                    model = ols(f"{col_val} ~ C({col_grp})", data=df_work).fit()
                    st.write(sm.stats.anova_lm(model, typ=2))

    # --- 5. AUTOML ---
    elif menu == "🤖 AutoML & Explainability":
        st.header("AutoML Avançado")
        automl = AutoMLPro()
        
        target = st.selectbox("Target:", df_work.columns)
        feats = st.multiselect("Features:", [c for c in df_work.columns if c != target])
        
        if st.button("🚀 Iniciar Treinamento") and feats:
            is_reg = pd.api.types.is_numeric_dtype(df_work[target]) and df_work[target].nunique() > 20
            algo = "Hist Gradient Boosting (LightGBM style)" # Default inteligente
            
            pipe, preds, y_test, imp_df = automl.train_model(df_work[feats], df_work[target], algo, is_reg)
            
            # Métricas
            c1, c2, c3 = st.columns(3)
            if is_reg:
                c1.metric("R2 Score", f"{r2_score(y_test, preds):.4f}")
                c2.metric("MAE", f"{mean_absolute_error(y_test, preds):.4f}")
            else:
                c1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                c2.metric("F1 Macro", f"{f1_score(y_test, preds, average='macro'):.2f}")

            # Visualização
            t1, t2 = st.tabs(["Resultados", "Explainability (Importância)"])
            with t1:
                if is_reg:
                    fig = px.scatter(x=y_test, y=preds, title="Real vs Previsto", labels={'x':'Real', 'y':'Previsto'})
                    fig.add_shape(type="line", line=dict(dash="dash", color="red"), x0=y_test.min(), y0=y_test.max(), x1=y_test.min(), y1=y_test.max())
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    cm = confusion_matrix(y_test, preds)
                    st.plotly_chart(px.imshow(cm, text_auto=True, title="Matriz de Confusão"), use_container_width=True)
            
            with t2:
                st.subheader("Feature Importance (Permutation)")
                st.bar_chart(imp_df.set_index('Feature'))
                st.caption("Calculado via Permutation Importance (Agnóstico a modelo)")

    # --- 6. TIME SERIES ---
    elif menu == "🔮 Time Series Studio":
        st.header("Séries Temporais Profissionais")
        tsp = TimeSeriesPro()
        
        date_col = st.selectbox("Data:", df_work.columns)
        val_col = st.selectbox("Valor:", df_work.select_dtypes(include=np.number).columns)
        
        if st.button("Executar Análise e Forecast"):
            try:
                # Converter para datetime se precisar
                df_ts = df_work.copy()
                df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                tsp.forecast_pro(df_ts, date_col, val_col)
            except Exception as e:
                st.error(f"Erro na análise: {e}. Verifique se a coluna de data está correta.")

    # --- 7. NLP ---
    elif menu == "🧠 NLP Studio":
        st.header("NLP & Text Mining")
        col_txt = st.selectbox("Texto:", df_work.select_dtypes(include='object').columns)
        
        if st.button("Processar Texto"):
            # Limpeza
            clean = df_work[col_txt].apply(lambda x: str(x).lower())
            
            # TF-IDF
            vec = TfidfVectorizer(stop_words='english', max_features=100)
            X = vec.fit_transform(clean)
            
            # Top Words
            sum_words = X.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]
            
            wf_df = pd.DataFrame(words_freq, columns=['Word', 'Count'])
            st.subheader("Top Palavras (TF-IDF)")
            st.bar_chart(wf_df.set_index('Word'))

    # --- 8. CLUSTERING ---
    elif menu == "🌀 Clustering PRO":
        st.header("Segmentação Avançada")
        cp = ClusteringPro()
        
        feats = st.multiselect("Features:", df_work.select_dtypes(include=np.number).columns)
        method = st.selectbox("Algoritmo:", ["K-Means", "DBSCAN", "Agglomerative (Hierarchical)"])
        
        if st.button("Clusterizar") and feats:
            labels, fig3d, figElbow = cp.auto_clustering(df_work, feats, method=method)
            
            c1, c2 = st.columns([3, 1])
            with c1: st.plotly_chart(fig3d, use_container_width=True)
            with c2:
                if figElbow: st.plotly_chart(figElbow, use_container_width=True)
            
            df_work['Cluster'] = labels
            st.success("Clusterização concluída e salva no dataset.")

    # --- 9. EXPORT ---
    elif menu == "📤 Relatórios & Export":
        st.header("Exportação Profissional")
        
        txt_report = f"""
        RELATORIO DATA STUDIO X
        Data: {datetime.now()}
        Registros: {df_work.shape[0]}
        Colunas: {df_work.shape[1]}
        
        Resumo Estatístico:
        {df_work.describe().T.to_string()}
        """
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Baixar Dataset Atual (CSV)", df_work.to_csv(index=False).encode('utf-8'), "data_studio_export.csv", "text/csv")
        with c2:
            pdf_data = create_pdf(df_work, txt_report)
            st.download_button("📄 Baixar Relatório PDF", pdf_data, "relatorio.pdf", "application/pdf")

if __name__ == "__main__":
    main()