import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import joblib
import traceback

# Scikit-Learn Suite
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, roc_auc_score, silhouette_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. CONFIGURAÇÃO E ESTILOS ---
st.set_page_config(
    page_title="DS Workbench Prometheus v19",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# CSS Otimizado
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; background: -webkit-linear-gradient(120deg, #f97316, #ea580c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-bottom: 1.2rem; font-weight: 500; }
    .stButton>button { border-radius: 6px; font-weight: 600; border: 1px solid #334155; }
    .stButton>button:hover { border-color: #f97316; color: #f97316; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #f97316; }
    .console-error { background-color: #450a0a; color: #fca5a5; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONTROLLER & STATE MANAGEMENT ---

class StateManager:
    """Gerencia estado, histórico e integridade do app."""
    
    def __init__(self):
        if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
        if 'history' not in st.session_state: st.session_state['history'] = []
        if 'max_history' not in st.session_state: st.session_state['max_history'] = 5

    def get_df(self):
        return st.session_state['df']

    def set_df(self, new_df, save_history=True):
        if save_history and not st.session_state['df'].empty:
            st.session_state['history'].append(st.session_state['df'].copy())
            if len(st.session_state['history']) > st.session_state['max_history']:
                st.session_state['history'].pop(0)
        st.session_state['df'] = new_df

    def undo(self):
        if st.session_state['history']:
            last_df = st.session_state['history'].pop()
            st.session_state['df'] = last_df
            st.toast("Ação desfeita com sucesso.", icon="↩️")
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning("Histórico vazio.")

app = StateManager()

# --- 3. MÓDULOS ---

class IngestionModule:
    @staticmethod
    def render():
        st.markdown("<div class='main-header'>1. Data Ingestion</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                st.markdown("### 📥 Upload")
                uploaded = st.file_uploader("CSV, Excel, Parquet, JSON", type=['csv', 'xlsx', 'parquet', 'json'])
                
                large_mode = st.checkbox("Modo Arquivo Grande (Amostra 50k)", help="Ative para ler apenas as primeiras 50 mil linhas de CSVs gigantes.")
                
                if uploaded:
                    try:
                        with st.spinner("Processando dados..."):
                            if uploaded.name.endswith('.csv'):
                                if large_mode:
                                    df = pd.read_csv(uploaded, nrows=50000)
                                    st.info("Modo Amostra: Carregadas as primeiras 50.000 linhas.")
                                else:
                                    df = pd.read_csv(uploaded)
                            elif uploaded.name.endswith('.xlsx'): df = pd.read_excel(uploaded)
                            elif uploaded.name.endswith('.parquet'): df = pd.read_parquet(uploaded)
                            elif uploaded.name.endswith('.json'): df = pd.read_json(uploaded)
                            
                            # Auto-cast datas
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    try: df[col] = pd.to_datetime(df[col])
                                    except: pass
                            
                            app.set_df(df)
                            st.success(f"Dataset carregado: {df.shape[0]} linhas")
                            
                    except Exception as e: st.error(f"Erro crítico: {e}")

                if st.button("🎲 Carregar Titanic (Demo)"):
                    app.set_df(pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"))
                    st.rerun()

        with c2:
            df = app.get_df()
            if not df.empty:
                with st.container(border=True):
                    st.markdown("### 📊 Overview")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Linhas", df.shape[0])
                    k2.metric("Colunas", df.shape[1])
                    k3.metric("Duplicatas", df.duplicated().sum())
                    mem = df.memory_usage(deep=True).sum() / 1024**2
                    k4.metric("Memória", f"{mem:.1f} MB")
                    
                    t1, t2 = st.tabs(["Amostra", "Estrutura"])
                    with t1: st.dataframe(df.head(50), use_container_width=True, height=300)
                    with t2: 
                        buf = io.StringIO()
                        df.info(buf=buf)
                        st.text(buf.getvalue())

class RefineryModule:
    @staticmethod
    def render():
        st.markdown("<div class='main-header'>2. Data Refinery</div>", unsafe_allow_html=True)
        df = app.get_df()
        if df.empty: st.warning("Sem dados."); return

        if st.button("↩️ Desfazer Alteração", key="undo_btn"): app.undo()

        t1, t2, t3 = st.tabs(["Limpeza", "Feature Engineering", "Encoding"])

        with t1:
            c1, c2 = st.columns(2)
            with c1:
                with st.form("nulls_form"):
                    st.subheader("Valores Nulos")
                    cols = st.multiselect("Colunas", df.columns)
                    method = st.selectbox("Ação", ["Preencher Média", "Preencher Mediana", "Preencher Moda", "Preencher 0/NA", "Dropar Linhas"])
                    if st.form_submit_button("Aplicar"):
                        new_df = df.copy()
                        if "Dropar" in method:
                            new_df = new_df.dropna(subset=cols)
                        else:
                            for c in cols:
                                if "Média" in method and pd.api.types.is_numeric_dtype(new_df[c]): val = new_df[c].mean()
                                elif "Mediana" in method and pd.api.types.is_numeric_dtype(new_df[c]): val = new_df[c].median()
                                elif "Moda" in method: val = new_df[c].mode()[0]
                                else: val = 0 if pd.api.types.is_numeric_dtype(new_df[c]) else "N/A"
                                new_df[c] = new_df[c].fillna(val)
                        app.set_df(new_df)
                        st.success("Nulos tratados.")
                        st.rerun()
            
            with c2:
                with st.form("filter_form"):
                    st.subheader("Filtros")
                    query = st.text_input("Query Pandas (ex: `Age > 20`)")
                    if st.form_submit_button("Filtrar"):
                        try: app.set_df(df.query(query)); st.rerun()
                        except Exception as e: st.error(f"Erro query: {e}")

        with t2:
            with st.form("calc_form"):
                st.subheader("Criar Variável (Seguro)")
                name = st.text_input("Nome Nova Coluna")
                expr = st.text_input("Fórmula (ex: `Fare / Age`)")
                if st.form_submit_button("Calcular"):
                    if name in df.columns:
                        st.error(f"A coluna '{name}' já existe! Use outro nome.")
                    else:
                        try:
                            new_df = df.copy()
                            new_df.eval(f"{name} = {expr}", inplace=True)
                            app.set_df(new_df)
                            st.success(f"Coluna {name} criada.")
                            st.rerun()
                        except Exception as e: st.error(f"Erro matemática: {e}")

        with t3:
            with st.form("onehot_form"):
                st.subheader("One-Hot Encoding")
                col_target = st.selectbox("Coluna Categórica", df.select_dtypes(include='object').columns)
                drop_first = st.checkbox("Drop First (Evitar Dummy Trap)", value=True)
                if st.form_submit_button("Aplicar One-Hot"):
                    new_df = pd.get_dummies(df, columns=[col_target], drop_first=drop_first)
                    app.set_df(new_df)
                    st.success(f"Encoding aplicado em {col_target}")
                    st.rerun()

class VisualModule:
    @staticmethod
    def render():
        st.markdown("<div class='main-header'>3. Visual Lab</div>", unsafe_allow_html=True)
        df = app.get_df()
        if df.empty: st.warning("Carregue dados."); return

        with st.sidebar:
            st.markdown("### 🎨 Plot Config")
            chart = st.selectbox("Tipo", ["Scatter", "Line", "Bar", "Histogram", "Box", "Heatmap"])
            x = st.selectbox("Eixo X", df.columns)
            y = st.selectbox("Eixo Y", [None] + list(df.columns)) if chart != "Histogram" else None
            color = st.selectbox("Cor", [None] + list(df.columns))
            facet = st.selectbox("Facet", [None] + list(df.columns))

        # Lógica de Amostragem e WebGL
        limit = 5000
        plot_df = df
        msg_sample = ""
        
        if len(df) > limit:
            msg_sample = f"⚠️ Visualizando amostra de {limit} pontos para performance."
            plot_df = df.sample(limit)
            
        st.caption(msg_sample)

        try:
            fig = None
            if chart == "Scatter":
                # Ativa WebGL se muitos dados
                render_mode = 'webgl' if len(plot_df) > 2000 else 'svg'
                trend = st.sidebar.selectbox("Trendline", [None, "ols", "lowess"])
                fig = px.scatter(plot_df, x=x, y=y, color=color, facet_col=facet, trendline=trend, 
                               render_mode=render_mode, template="plotly_dark", title=f"Scatter ({render_mode})")

            elif chart == "Bar":
                fig = px.bar(plot_df, x=x, y=y, color=color, facet_col=facet, template="plotly_dark")
            elif chart == "Histogram":
                fig = px.histogram(plot_df, x=x, color=color, facet_col=facet, template="plotly_dark")
            elif chart == "Box":
                fig = px.box(plot_df, x=x, y=y, color=color, template="plotly_dark")
            elif chart == "Heatmap":
                cols = st.multiselect("Colunas", df.select_dtypes(include=np.number).columns)
                if cols:
                    corr = df[cols].corr()
                    fig = px.imshow(corr, text_auto=True, template="plotly_dark", color_continuous_scale='RdBu_r')

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Erro Plot: {e}")

class AutoMLModule:
    @staticmethod
    def render():
        st.markdown("<div class='main-header'>4. AutoML Prometheus</div>", unsafe_allow_html=True)
        df = app.get_df()
        if df.empty: st.warning("Carregue dados."); return

        c1, c2 = st.columns([1, 3])
        with c1:
            with st.container(border=True):
                st.markdown("### ⚙️ Setup")
                target = st.selectbox("Target", df.columns)
                features = st.multiselect("Features", [c for c in df.columns if c != target])
                
                is_reg = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20
                task = "Regressão" if is_reg else "Classificação"
                if "Cluster" in target: task = "Clustering"
                st.info(f"Task: **{task}**")
                
                algo = st.selectbox("Modelo", ["Random Forest", "Linear/Logistic"])
                cv_k = st.slider("K-Folds CV", 2, 10, 5)
                run = st.button("🚀 Treinar", type="primary")

        with c2:
            if run and features:
                AutoMLModule.execute(df, features, target, task, algo, cv_k)

    @staticmethod
    def execute(df, features, target, task, algo, cv_k):
        try:
            X, y = df[features], df[target]
            
            # Pipeline Setup
            nums = X.select_dtypes(include=np.number).columns
            cats = X.select_dtypes(include=['object', 'category']).columns
            
            preprocessor = ColumnTransformer([
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), nums),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats)
            ])
            
            if task == "Regressão":
                model = RandomForestRegressor(random_state=42) if "Forest" in algo else LinearRegression()
                metric = 'r2'
            else:
                model = RandomForestClassifier(random_state=42) if "Forest" in algo else LogisticRegression(max_iter=2000)
                metric = 'accuracy'
                
            pipe = Pipeline([('pre', preprocessor), ('model', model)])
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.status("Treinando Modelo...", expanded=True):
                st.write("⚙️ Ajustando pré-processamento...")
                pipe.fit(X_train, y_train)
                
                st.write(f"⚔️ Calculando Cross-Validation ({cv_k}-folds)...")
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_k, scoring=metric)
                
                st.write("✅ Finalizando...")
                y_pred = pipe.predict(X_test)
                
            # Resultados Modularizados
            st.divider()
            if task == "Regressão":
                AutoMLModule._report_regression(y_test, y_pred, cv_scores)
            else:
                # Probabilidade para ROC (se suportado)
                y_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
                AutoMLModule._report_classification(y_test, y_pred, y_prob, cv_scores, pipe.classes_)

            # Download
            buf = io.BytesIO()
            joblib.dump(pipe, buf)
            buf.seek(0)
            st.download_button("💾 Baixar Modelo (.joblib)", buf, "model.joblib")

        except Exception as e:
            st.error("Falha no Pipeline:")
            st.markdown(f"<div class='console-error'>{traceback.format_exc()}</div>", unsafe_allow_html=True)

    @staticmethod
    def _report_regression(y_true, y_pred, cv_scores):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (Teste)", f"{r2:.3f}")
        c2.metric("CV Médio", f"{cv_scores.mean():.3f}")
        c3.metric("RMSE", f"{rmse:.2f}")
        c4.metric("MAE", f"{mae:.2f}")
        
        df_res = pd.DataFrame({'Real': y_true, 'Previsto': y_pred})
        fig = px.scatter(df_res, x='Real', y='Previsto', title="Real vs Previsto", template="plotly_dark")
        fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _report_classification(y_true, y_pred, y_prob, cv_scores, classes):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC-AUC Robusto (Binário ou Multi)
        roc = 0.0
        if y_prob is not None:
            try:
                if len(classes) == 2:
                    roc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except: pass

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acurácia", f"{acc:.2%}")
        c2.metric("CV Médio", f"{cv_scores.mean():.2%}")
        c3.metric("F1-Score", f"{f1:.2f}")
        c4.metric("ROC-AUC", f"{roc:.2f}")

        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Matriz de Confusão", color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)

class ConsoleModule:
    @staticmethod
    def render():
        st.markdown("<div class='main-header'>5. Python Console</div>", unsafe_allow_html=True)
        st.warning("⚠️ Sandbox: Comandos de sistema bloqueados.")
        
        code = st.text_area("Script", height=200, value="# df['Log_Fare'] = np.log1p(df['Fare'])\n# st.write(df.head())")
        
        if st.button("Executar"):
            forbidden = ["os.", "sys.", "subprocess", "open(", "eval(", "exec("]
            if any(x in code for x in forbidden):
                st.error("🚫 Comando bloqueado.")
                return

            try:
                local_env = {'pd': pd, 'np': np, 'st': st, 'px': px, 'df': app.get_df().copy()}
                exec(code, {}, local_env)
                
                if 'df' in local_env and not local_env['df'].equals(app.get_df()):
                    app.set_df(local_env['df'])
                    st.toast("DataFrame atualizado!", icon="🐍")
            except Exception:
                st.markdown(f"<div class='console-error'>{traceback.format_exc()}</div>", unsafe_allow_html=True)

# --- 4. MAIN ---

def main():
    with st.sidebar:
        st.title("Prometheus v19")
        st.markdown("---")
        page = st.radio("Navegação", ["Ingestion", "Refinery", "Visual Lab", "AutoML", "Console"])
        st.markdown("---")
        if not app.get_df().empty:
            st.download_button("💾 Baixar CSV", app.get_df().to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

    if page == "Ingestion": IngestionModule.render()
    elif page == "Refinery": RefineryModule.render()
    elif page == "Visual Lab": VisualModule.render()
    elif page == "AutoML": AutoMLModule.render()
    elif page == "Console": ConsoleModule.render()

if __name__ == "__main__":
    main()