import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import base64
import re
import unicodedata
import logging
from io import BytesIO

# --- BIBLIOTECAS CIENTÍFICAS ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# --- SKLEARN PIPELINES & PREPROCESSING (MODERNO) ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURAÇÃO DE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Data Studio Pro", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(14, 17, 23, 0.05);
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    /* Ajuste para tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES & CACHING ---

@st.cache_data(show_spinner=False)
def load_data(file):
    """Carrega dados com cache para performance."""
    try:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        logger.info(f"Dataset carregado: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Erro fatal ao ler arquivo: {e}")
        return None

def to_bytes(obj):
    """Serializa modelos usando joblib (padrão profissional)."""
    bio = BytesIO()
    joblib.dump(obj, bio)
    bio.seek(0)
    return bio.read()

def dsa_limpa_texto(texto):
    """Limpeza de texto robusta para NLP."""
    if not isinstance(texto, str): return ""
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z\s]', '', texto.lower()).strip()

# --- BRAIN: FUNÇÕES DE AUTOML AVANÇADO ---

def build_preprocessor(X):
    """Cria um ColumnTransformer inteligente que trata nulos e categorias automaticamente."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline Numérico: Imputa média -> Normaliza
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline Categórico: Imputa moda -> OneHot
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    
    return preprocessor

def train_advanced_model(X, y, model_type='Random Forest', is_regression=True):
    """Treina modelo com Pipeline completo e RandomizedSearchCV."""
    
    preprocessor = build_preprocessor(X)
    
    if is_regression:
        if model_type == 'Random Forest':
            clf = RandomForestRegressor(n_jobs=-1)
            param_dist = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
        else:
            clf = GradientBoostingRegressor()
            param_dist = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
    else:
        if model_type == 'Random Forest':
            clf = RandomForestClassifier(n_jobs=-1)
            param_dist = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
        else:
            clf = GradientBoostingClassifier()
            param_dist = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    
    # Busca de Hiperparâmetros (AutoML Real)
    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=5, cv=3, n_jobs=-1, verbose=1)
    search.fit(X, y)
    
    return search.best_estimator_, search.best_score_

# --- SIDEBAR & NAVEGAÇÃO ---
with st.sidebar:
    st.title("🧬 Data Studio")
    st.caption("v9.0 Professional Edition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Carregar Dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Lógica de Sessão Segura
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df_raw'] = df
                st.session_state['df_work'] = df.copy()
                st.session_state['file_name'] = uploaded_file.name
                st.toast("Dados carregados com sucesso!", icon="✅")
        
        if 'df_work' in st.session_state:
            df_work = st.session_state['df_work']
            st.info(f"Linhas: {df_work.shape[0]} | Colunas: {df_work.shape[1]}")
            
            if st.button("🔄 Resetar Dados"):
                st.session_state['df_work'] = st.session_state['df_raw'].copy()
                st.rerun()
            
            st.markdown("---")
            menu = st.radio("Módulos:", [
                "🏠 Dashboard & Resumo",
                "🛠️ Engenharia Manual",     
                "🔬 Explorador Visual",
                "⚗️ Laboratório Estatístico", 
                "🤖 AutoML (Pipelines)",
                "🧠 NLP Studio",
                "🌀 Clustering",
                "🎨 Visualizador Pro"
            ])
    else:
        menu = "Home"

# --- HOME ---
if not uploaded_file:
    st.title("Data Studio Pro v9.0")
    st.markdown("""
    ### A Workstation Definitiva de Ciência de Dados.
    
    Esta ferramenta une a flexibilidade da manipulação manual com a robustez de pipelines profissionais de Machine Learning.
    
    **Funcionalidades Principais:**
    1.  **Engenharia Manual:** Limpeza, renomeação e tratamento de nulos para exportação.
    2.  **Estatística (Chemometrics):** ANOVA, OLS, Testes T (Statsmodels/Scipy).
    3.  **AutoML Profissional:** Pipelines com `ColumnTransformer` e `RandomizedSearchCV`.
    4.  **NLP & Clustering:** Ferramentas especializadas para texto e grupos.
    
    👈 **Carregue seu arquivo para iniciar.**
    """)
    st.stop()

# ==============================================================================
# 1. DASHBOARD & RESUMO (RESTAURADO)
# ==============================================================================
if menu == "🏠 Dashboard & Resumo":
    st.header("Visão Geral dos Dados")
    
    # Métricas de Topo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", df_work.shape[0])
    c2.metric("Colunas", df_work.shape[1])
    c3.metric("Células Vazias", df_work.isna().sum().sum())
    c4.metric("Duplicatas", df_work.duplicated().sum())
    
    col_L, col_R = st.columns([2, 1])
    
    with col_L:
        st.subheader("Amostra de Dados")
        st.dataframe(df_work.head(50), use_container_width=True, height=400)
        
    with col_R:
        st.subheader("Tipos de Variáveis")
        dtypes = df_work.dtypes.value_counts().reset_index()
        dtypes.columns = ['Tipo', 'Qtd']
        dtypes['Tipo'] = dtypes['Tipo'].astype(str)
        fig = px.pie(dtypes, names='Tipo', values='Qtd', hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Estatísticas Descritivas (Summary)")
    st.markdown("Resumo estatístico completo das variáveis numéricas e categóricas.")
    st.dataframe(df_work.describe(include='all'), use_container_width=True)

# ==============================================================================
# 2. ENGENHARIA MANUAL (COMPLETO)
# ==============================================================================
elif menu == "🛠️ Engenharia Manual":
    st.header("Engenharia de Dados & Limpeza")
    st.caption("Manipule seus dados manualmente antes da modelagem ou para exportação.")
    
    t1, t2, t3 = st.tabs(["Estrutura & Tipos", "Limpeza de Nulos", "Transformações"])
    
    # --- T1: Estrutura ---
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gerenciar Colunas")
            # Renomear
            col_ren = st.selectbox("Renomear Coluna:", df_work.columns)
            new_name = st.text_input("Novo nome:", value=col_ren)
            if st.button("Aplicar Renomeação"):
                df_work.rename(columns={col_ren: new_name}, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
            
            # Excluir
            cols_del = st.multiselect("Excluir Colunas:", df_work.columns)
            if st.button("🗑️ Deletar Selecionadas"):
                df_work.drop(columns=cols_del, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
                
        with c2:
            st.subheader("Tipagem de Dados")
            col_type = st.selectbox("Coluna:", df_work.columns, key="type_col")
            to_type = st.selectbox("Converter para:", ["Numérico", "Texto", "Data"])
            
            if st.button("Converter Tipo"):
                try:
                    if to_type == "Numérico":
                        df_work[col_type] = pd.to_numeric(df_work[col_type], errors='coerce')
                    elif to_type == "Data":
                        df_work[col_type] = pd.to_datetime(df_work[col_type], errors='coerce')
                    else:
                        df_work[col_type] = df_work[col_type].astype(str)
                    st.session_state['df_work'] = df_work
                    st.success("Convertido!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

    # --- T2: Limpeza ---
    with t2:
        c_dup, c_null = st.columns(2)
        with c_dup:
            st.subheader("Duplicatas")
            if st.button("Remover Duplicatas"):
                old = len(df_work)
                df_work.drop_duplicates(inplace=True)
                st.session_state['df_work'] = df_work
                st.success(f"Removidas {old - len(df_work)} linhas.")
                st.rerun()
        
        with c_null:
            st.subheader("Tratar Nulos (Manual)")
            cols_null = df_work.columns[df_work.isna().any()].tolist()
            if cols_null:
                target_null = st.selectbox("Coluna com Nulos:", cols_null)
                method = st.selectbox("Ação:", ["Preencher com 0", "Média", "Mediana", "Moda", "Remover Linhas"])
                
                if st.button("Aplicar Tratamento"):
                    if method == "Remover Linhas":
                        df_work.dropna(subset=[target_null], inplace=True)
                    elif method == "Preencher com 0":
                        df_work[target_null].fillna(0, inplace=True)
                    elif method == "Média" and pd.api.types.is_numeric_dtype(df_work[target_null]):
                        df_work[target_null].fillna(df_work[target_null].mean(), inplace=True)
                    elif method == "Mediana" and pd.api.types.is_numeric_dtype(df_work[target_null]):
                        df_work[target_null].fillna(df_work[target_null].median(), inplace=True)
                    elif method == "Moda":
                        df_work[target_null].fillna(df_work[target_null].mode()[0], inplace=True)
                    
                    st.session_state['df_work'] = df_work
                    st.rerun()
            else:
                st.success("Sem nulos no dataset atual.")

    # --- T3: Transformações ---
    with t3:
        st.subheader("Manipulação Avançada")
        mode = st.radio("Modo:", ["One-Hot Encoding (Dummies)", "Mapeamento Manual (Labels)"])
        
        if mode == "One-Hot Encoding (Dummies)":
            col_dum = st.selectbox("Coluna Categórica:", df_work.select_dtypes(include='object').columns)
            if st.button("Gerar Dummies"):
                df_work = pd.get_dummies(df_work, columns=[col_dum], drop_first=True, dtype=int)
                st.session_state['df_work'] = df_work
                st.rerun()
                
        else:
            col_map = st.selectbox("Coluna para Mapear:", df_work.columns)
            uniques = df_work[col_map].unique()
            if len(uniques) < 50:
                mapping = {}
                st.write("Defina os valores:")
                cols_ui = st.columns(3)
                for i, val in enumerate(uniques):
                    with cols_ui[i%3]:
                        new_v = st.text_input(f"'{val}' vira:", key=f"m_{val}")
                        if new_v: mapping[val] = new_v
                
                if st.button("Aplicar Mapa"):
                    # Tenta converter valores para float se possível
                    final_map = {}
                    for k, v in mapping.items():
                        try: final_map[k] = float(v)
                        except: final_map[k] = v
                    
                    df_work[col_map] = df_work[col_map].map(final_map).fillna(df_work[col_map])
                    st.session_state['df_work'] = df_work
                    st.rerun()
            else:
                st.warning("Muitos valores únicos para mapear manualmente.")

    st.divider()
    # Exportar CSV Manipulado
    csv = df_work.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar CSV Processado", csv, "dados_processados.csv", "text/csv")

# ==============================================================================
# 3. EXPLORADOR VISUAL
# ==============================================================================
elif menu == "🔬 Explorador Visual":
    st.header("Exploração Gráfica")
    t1, t2 = st.tabs(["Histogramas & Barras", "Correlações"])
    
    with t1:
        col_vis = st.selectbox("Variável:", df_work.columns)
        if pd.api.types.is_numeric_dtype(df_work[col_vis]):
            fig = px.histogram(df_work, x=col_vis, marginal="box", title=f"Distribuição: {col_vis}")
        else:
            fig = px.bar(df_work[col_vis].value_counts().head(20), title=f"Top Categorias: {col_vis}")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            corr = df_num.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.warning("Colunas numéricas insuficientes para correlação.")

# ==============================================================================
# 4. ESTATÍSTICA (CHEMOMETRICS) - MANTIDO
# ==============================================================================
elif menu == "⚗️ Laboratório Estatístico":
    st.header("Estatística & Planejamento (Statsmodels)")
    
    t_ols, t_anova, t_ttest = st.tabs(["Regressão OLS", "ANOVA", "Teste T"])
    
    with t_ols:
        st.subheader("Regressão Linear Múltipla")
        y_ols = st.selectbox("Y (Resposta):", df_work.select_dtypes(include='number').columns)
        x_ols = st.multiselect("X (Preditores):", df_work.select_dtypes(include='number').columns)
        
        if st.button("Calcular OLS") and x_ols:
            try:
                # Statsmodels lida mal com Nulos, dropamos temporariamente para o calculo
                df_ols = df_work[[y_ols] + x_ols].dropna()
                X = sm.add_constant(df_ols[x_ols])
                model = sm.OLS(df_ols[y_ols], X).fit()
                st.text(model.summary())
                
                # Plot Resíduos
                fig = px.scatter(x=model.fittedvalues, y=model.resid, labels={'x':'Ajustado', 'y':'Resíduo'})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro OLS: {e}")

    with t_anova:
        st.subheader("ANOVA (One-Way / N-Way)")
        y_an = st.selectbox("Y:", df_work.select_dtypes(include='number').columns, key="an_y")
        x_an = st.multiselect("Fatores:", df_work.columns, key="an_x")
        
        if st.button("Calcular ANOVA") and x_an:
            try:
                # Fórmula estilo R
                safe_cols = {col: f"C({col})" if df_work[col].dtype == 'object' else col for col in x_an}
                formula_terms = [safe_cols[col] for col in x_an]
                f = f"{y_an} ~ {' + '.join(formula_terms)}"
                
                model = ols(f, data=df_work.dropna(subset=[y_an] + x_an)).fit()
                table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(table.style.format("{:.4f}"))
            except Exception as e:
                st.error(f"Erro ANOVA: {e}. Tente renomear colunas para remover espaços.")

    with t_ttest:
        st.subheader("Teste T de Student")
        grp = st.selectbox("Coluna Grupo (2 níveis):", df_work.columns)
        val = st.selectbox("Variável Numérica:", df_work.select_dtypes(include='number').columns)
        
        if st.button("Executar Teste T"):
            groups = df_work[grp].dropna().unique()
            if len(groups) == 2:
                g1 = df_work[df_work[grp] == groups[0]][val].dropna()
                g2 = df_work[df_work[grp] == groups[1]][val].dropna()
                stat, p = stats.ttest_ind(g1, g2)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Média {groups[0]}", f"{g1.mean():.2f}")
                c2.metric(f"Média {groups[1]}", f"{g2.mean():.2f}")
                st.metric("P-Valor", f"{p:.5f}")
                
                if p < 0.05: st.success("Diferença Estatística Significativa!")
                else: st.info("Sem diferença significativa.")
            else:
                st.error("A coluna de grupo deve ter exatamente 2 valores únicos.")

# ==============================================================================
# 5. AUTOML PROFISSIONAL (PIPELINES + SEARCH)
# ==============================================================================
elif menu == "🤖 AutoML (Pipelines)":
    st.header("AutoML com Pipelines Profissionais")
    st.markdown("Utiliza `ColumnTransformer` para tratar nulos/categorias automaticamente e `RandomizedSearchCV` para otimização.")
    
    target = st.selectbox("Target (Alvo):", df_work.columns)
    features = st.multiselect("Features:", [c for c in df_work.columns if c != target])
    
    if not features:
        st.info("Selecione features para treinar.")
    else:
        model_arch = st.selectbox("Algoritmo:", ["Random Forest", "Gradient Boosting"])
        
        if st.button("🚀 Iniciar Treinamento Robusto"):
            try:
                with st.spinner("Otimizando Pipeline de Machine Learning..."):
                    X = df_work[features]
                    y = df_work[target]
                    
                    # Detectar problema
                    is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Treinar usando a função avançada
                    model, score = train_advanced_model(X_train, y_train, model_arch, is_reg)
                    
                    # Avaliar
                    preds = model.predict(X_test)
                    
                    st.success("Treinamento Concluído!")
                    st.write(f"Melhor Score (Validação Cruzada): {score:.4f}")
                    
                    # Métricas Teste
                    c1, c2 = st.columns(2)
                    if is_reg:
                        c1.metric("R2 (Teste)", f"{r2_score(y_test, preds):.2%}")
                        c2.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")
                        st.plotly_chart(px.scatter(x=y_test, y=preds, labels={'x':'Real', 'y':'Previsto'}, title="Real vs Previsto"))
                    else:
                        c1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                        st.text("Relatório de Classificação:")
                        st.text(classification_report(y_test, preds))
                        st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True, title="Matriz Confusão"))
                    
                    # Download Joblib (Profissional)
                    model_bytes = to_bytes(model)
                    st.download_button("📥 Baixar Pipeline Completo (.joblib)", data=model_bytes, file_name="pipeline_model.joblib")
                    
            except Exception as e:
                st.error(f"Erro no treinamento: {e}")

# ==============================================================================
# 6. NLP STUDIO
# ==============================================================================
elif menu == "🧠 NLP Studio":
    st.header("Processamento de Texto (NLP)")
    col_txt = st.selectbox("Coluna de Texto:", df_work.select_dtypes(include='object').columns)
    col_tgt = st.selectbox("Target (Classe):", df_work.columns, index=1 if len(df_work.columns)>1 else 0)
    
    if st.button("Treinar Pipeline NLP"):
        with st.spinner("Vetorizando e Treinando..."):
            # Limpeza On-the-fly
            texts = df_work[col_txt].apply(dsa_limpa_texto)
            targets = df_work[col_tgt]
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
                ('clf', LogisticRegression(max_iter=1000))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(texts, targets, test_size=0.2)
            pipeline.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            st.success(f"Modelo NLP Treinado! Acurácia: {acc:.2%}")
            
            # Salvar sessão e download
            st.session_state['nlp_model'] = pipeline
            st.download_button("📥 Baixar Modelo NLP", data=to_bytes(pipeline), file_name="nlp_model.joblib")
            
    if 'nlp_model' in st.session_state:
        txt = st.text_input("Teste seu modelo NLP:")
        if txt:
            clean = dsa_limpa_texto(txt)
            pred = st.session_state['nlp_model'].predict([clean])[0]
            st.info(f"Classificação: **{pred}**")

# ==============================================================================
# 7. CLUSTERING & VISUALIZADOR (MANTIDOS)
# ==============================================================================
elif menu == "🌀 Clustering":
    st.header("Clustering (K-Means)")
    feats = st.multiselect("Features Numéricas:", df_work.select_dtypes(include='number').columns)
    k = st.slider("Clusters (K):", 2, 10, 3)
    
    if st.button("Clusterizar") and feats:
        X = df_work[feats].dropna()
        # Pipeline de Clustering
        pipe_cl = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=k))
        ])
        clusters = pipe_cl.fit_predict(X)
        
        # PCA para visualização
        pca = PCA(n_components=2).fit_transform(Pipeline([('s', StandardScaler())]).fit_transform(X))
        
        fig = px.scatter(x=pca[:,0], y=pca[:,1], color=clusters.astype(str), title="PCA dos Clusters")
        st.plotly_chart(fig)
        
        df_work.loc[X.index, 'Cluster'] = clusters
        st.success("Clusters adicionados ao dataset!")

elif menu == "🎨 Visualizador Pro":
    st.header("Gráficos Customizáveis")
    c_conf, c_plot = st.columns([1, 3])
    with c_conf:
        tipo = st.selectbox("Tipo:", ["Scatter", "Barra", "Linha", "Hist", "Box", "3D"])
        x = st.selectbox("Eixo X:", df_work.columns)
        y = st.selectbox("Eixo Y/Z:", df_work.columns, index=1)
        cor = st.selectbox("Cor:", [None] + list(df_work.columns))
        tema = st.selectbox("Tema:", ["plotly", "plotly_dark", "ggplot2", "seaborn"])
        
    with c_plot:
        try:
            if tipo == "Scatter": fig = px.scatter(df_work, x=x, y=y, color=cor, template=tema)
            elif tipo == "Barra": fig = px.bar(df_work, x=x, y=y, color=cor, template=tema)
            elif tipo == "Linha": fig = px.line(df_work, x=x, y=y, color=cor, template=tema)
            elif tipo == "Hist": fig = px.histogram(df_work, x=x, color=cor, template=tema)
            elif tipo == "Box": fig = px.box(df_work, x=x, y=y, color=cor, template=tema)
            elif tipo == "3D": 
                if pd.api.types.is_numeric_dtype(df_work[cor]):
                    fig = px.scatter_3d(df_work, x=x, y=y, z=cor, color=cor, template=tema)
                else:
                    st.warning("Para 3D, use cor numérica.")
                    fig = px.scatter(df_work, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro gráfico: {e}")