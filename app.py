import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import base64
import re
import unicodedata
from io import BytesIO

# --- BIBLIOTECAS CIENTÍFICAS ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# --- MACHINE LEARNING ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Data Studio Ultimate", 
    layout="wide", 
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO (NATIVO + AJUSTES) ---
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-radius: 8px;
        padding: 10px;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith("csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None

def download_file(object_to_download, filename, label="Baixar Arquivo"):
    if isinstance(object_to_download, pd.DataFrame):
        data = object_to_download.to_csv(index=False).encode('utf-8')
        mime = "text/csv"
    else:
        data = pickle.dumps(object_to_download)
        data = base64.b64encode(data).decode()
        href = f'<a href="data:file/output_model;base64,{data}" download="{filename}">📥 {label}</a>'
        st.markdown(href, unsafe_allow_html=True)
        return

    st.download_button(label=f"📥 {label}", data=data, file_name=filename, mime=mime)

def dsa_limpa_texto(texto):
    if not isinstance(texto, str): return ""
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z\s]', '', texto.lower()).strip()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Data Studio")
    st.caption("v8.0 Definitive Edition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Dataset (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Gestão de Estado
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df_raw'] = df
                st.session_state['df_work'] = df.copy()
                st.session_state['file_name'] = uploaded_file.name
                st.toast("Dataset carregado!", icon="✅")
        
        if 'df_work' in st.session_state:
            df_work = st.session_state['df_work']
            st.info(f"L: {df_work.shape[0]} | C: {df_work.shape[1]}")
            
            if st.button("🔄 Restaurar Original"):
                st.session_state['df_work'] = st.session_state['df_raw'].copy()
                st.rerun()
            
            st.markdown("---")
            menu = st.radio("Fluxo de Trabalho:", [
                "🏠 Dashboard",
                "🛠️ Engenharia de Dados",     # Restaurado Completo
                "🔬 Explorador & Correlação",
                "⚗️ Estatística & DOE",        # Mantido v7
                "🤖 AutoML (Tabular)",
                "🧠 NLP Studio (Texto)",
                "🌀 Clustering",
                "🎨 Visualizador Pro"
            ])
    else:
        menu = "Home"

# --- HOME ---
if not uploaded_file:
    st.title("Data Studio Ultimate v8.0")
    st.markdown("""
    ### A Ferramenta Definitiva para Cientistas de Dados.
    
    Esta versão consolida todas as funcionalidades em uma interface robusta.
    
    1.  **Engenharia Completa:** Limpeza, Tipagem, Nulos, Mapeamento Manual.
    2.  **Estatística (Chemometrics):** ANOVA, OLS, Testes de Hipótese.
    3.  **Machine Learning:** AutoML, NLP e Clustering.
    4.  **Visualização:** Gráficos customizáveis para publicação.
    
    👈 **Carregue seu arquivo para começar.**
    """)
    st.stop()

# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if menu == "🏠 Dashboard":
    st.header("Visão Geral")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Registros", df_work.shape[0])
    c2.metric("Total Colunas", df_work.shape[1])
    c3.metric("Dados Faltantes", df_work.isna().sum().sum())
    
    st.subheader("Amostra de Dados")
    st.dataframe(df_work.head(50), use_container_width=True)
    
    st.subheader("Tipos de Dados")
    dtypes = df_work.dtypes.value_counts().reset_index()
    dtypes.columns = ['Tipo', 'Contagem']
    dtypes['Tipo'] = dtypes['Tipo'].astype(str)
    st.plotly_chart(px.pie(dtypes, names='Tipo', values='Contagem', hole=0.4), use_container_width=True)

# ==============================================================================
# 2. ENGENHARIA DE DADOS (COMPLETO RESTAURADO)
# ==============================================================================
elif menu == "🛠️ Engenharia de Dados":
    st.header("Engenharia de Dados & Limpeza")
    
    tab_struct, tab_clean, tab_transf = st.tabs(["1. Estrutura & Tipos", "2. Limpeza & Nulos", "3. Transformação"])
    
    # --- TAB 1: ESTRUTURA ---
    with tab_struct:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Renomear Colunas")
            col_rename = st.selectbox("Coluna:", df_work.columns)
            new_name = st.text_input("Novo Nome:", value=col_rename)
            if st.button("Aplicar Renomeação"):
                df_work.rename(columns={col_rename: new_name}, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
                
            st.divider()
            st.subheader("Excluir Colunas")
            cols_drop = st.multiselect("Selecione para apagar:", df_work.columns)
            if st.button("🗑️ Excluir Colunas Selecionadas"):
                df_work.drop(columns=cols_drop, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
                
        with c2:
            st.subheader("Converter Tipos de Dados")
            col_convert = st.selectbox("Selecione a Coluna:", df_work.columns)
            target_type = st.selectbox("Para qual tipo?", ["Numérico (Float)", "Texto (String)", "Data (DateTime)"])
            
            if st.button("Converter Tipo"):
                try:
                    if "Numérico" in target_type:
                        df_work[col_convert] = pd.to_numeric(df_work[col_convert], errors='coerce')
                    elif "Texto" in target_type:
                        df_work[col_convert] = df_work[col_convert].astype(str)
                    elif "Data" in target_type:
                        df_work[col_convert] = pd.to_datetime(df_work[col_convert], errors='coerce')
                    st.session_state['df_work'] = df_work
                    st.success(f"Coluna {col_convert} convertida!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro na conversão: {e}")

    # --- TAB 2: LIMPEZA ---
    with tab_clean:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Remover Duplicatas")
            if st.button("Detectar e Remover Duplicatas"):
                antes = len(df_work)
                df_work.drop_duplicates(inplace=True)
                depois = len(df_work)
                st.session_state['df_work'] = df_work
                st.success(f"Removidas {antes - depois} linhas duplicadas.")
                st.rerun()
                
        with c2:
            st.subheader("Tratamento Avançado de Nulos")
            cols_null = df_work.columns[df_work.isna().any()].tolist()
            
            if cols_null:
                col_target = st.selectbox("Coluna com Nulos:", cols_null)
                method = st.selectbox("Método de Tratamento:", 
                    ["Preencher com 0", "Preencher com Média", "Preencher com Mediana", "Preencher com Moda", "Remover Linhas", "Valor Personalizado"])
                
                custom_val = None
                if method == "Valor Personalizado":
                    custom_val = st.text_input("Digite o valor:")
                
                if st.button("Aplicar Tratamento de Nulos"):
                    if method == "Remover Linhas":
                        df_work.dropna(subset=[col_target], inplace=True)
                    elif method == "Preencher com 0":
                        df_work[col_target].fillna(0, inplace=True)
                    elif method == "Preencher com Média" and pd.api.types.is_numeric_dtype(df_work[col_target]):
                        df_work[col_target].fillna(df_work[col_target].mean(), inplace=True)
                    elif method == "Preencher com Mediana" and pd.api.types.is_numeric_dtype(df_work[col_target]):
                        df_work[col_target].fillna(df_work[col_target].median(), inplace=True)
                    elif method == "Preencher com Moda":
                        df_work[col_target].fillna(df_work[col_target].mode()[0], inplace=True)
                    elif method == "Valor Personalizado" and custom_val:
                        df_work[col_target].fillna(custom_val, inplace=True)
                    
                    st.session_state['df_work'] = df_work
                    st.success("Tratamento aplicado!")
                    st.rerun()
            else:
                st.success("O Dataset não possui valores nulos neste momento.")

    # --- TAB 3: TRANSFORMAÇÃO ---
    with tab_transf:
        st.subheader("Feature Engineering")
        
        type_transf = st.radio("Ferramenta:", ["One-Hot Encoding (Dummies)", "Normalização (StandardScaler)", "Mapeamento Manual"])
        
        if type_transf == "One-Hot Encoding (Dummies)":
            col_dummy = st.selectbox("Coluna Categórica:", df_work.select_dtypes(include='object').columns)
            if st.button("Gerar Dummies"):
                df_work = pd.get_dummies(df_work, columns=[col_dummy], drop_first=True, dtype=int)
                st.session_state['df_work'] = df_work
                st.rerun()
                
        elif type_transf == "Normalização (StandardScaler)":
            cols_norm = st.multiselect("Colunas para Normalizar:", df_work.select_dtypes(include='number').columns)
            if st.button("Aplicar Normalização") and cols_norm:
                scaler = StandardScaler()
                df_work[cols_norm] = scaler.fit_transform(df_work[cols_norm])
                st.session_state['df_work'] = df_work
                st.rerun()
                
        elif type_transf == "Mapeamento Manual":
            col_map = st.selectbox("Coluna para Mapear:", df_work.columns)
            unique_vals = df_work[col_map].unique()
            if len(unique_vals) < 50:
                st.write("Defina os valores:")
                mapping = {}
                cols_ui = st.columns(2)
                for i, val in enumerate(unique_vals):
                    with cols_ui[i%2]:
                        new_val = st.text_input(f"Valor para '{val}':", key=f"map_{val}")
                        if new_val: mapping[val] = new_val
                
                if st.button("Aplicar Mapeamento"):
                    # Tenta converter para numero se possivel
                    final_map = {}
                    for k, v in mapping.items():
                        try: final_map[k] = float(v)
                        except: final_map[k] = v
                    
                    df_work[col_map] = df_work[col_map].map(final_map).fillna(df_work[col_map])
                    st.session_state['df_work'] = df_work
                    st.rerun()
            else:
                st.warning("Muitos valores únicos para mapeamento manual.")
    
    st.divider()
    download_file(df_work, "dados_engenharia.csv", "Baixar Dados Atuais (CSV)")

# ==============================================================================
# 3. EXPLORADOR
# ==============================================================================
elif menu == "🔬 Explorador & Correlação":
    st.header("Análise Exploratória")
    t1, t2 = st.tabs(["Distribuições", "Correlações"])
    
    with t1:
        col_sel = st.selectbox("Variável:", df_work.columns)
        if pd.api.types.is_numeric_dtype(df_work[col_sel]):
            fig = px.histogram(df_work, x=col_sel, marginal="box", title=f"Hist: {col_sel}")
        else:
            fig = px.bar(df_work[col_sel].value_counts().head(20), title=f"Contagem: {col_sel}")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            fig = px.imshow(df_num.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True, height=700)
        else:
            st.warning("Colunas numéricas insuficientes.")

# ==============================================================================
# 4. ESTATÍSTICA & DOE
# ==============================================================================
elif menu == "⚗️ Estatística & DOE":
    st.header("Laboratório Estatístico")
    st.caption("Statsmodels & Scipy Integration")
    
    t_reg, t_anova, t_test = st.tabs(["Regressão OLS", "ANOVA", "Testes T"])
    
    with t_reg:
        target = st.selectbox("Y (Resposta):", df_work.select_dtypes(include='number').columns)
        feats = st.multiselect("X (Preditores):", df_work.select_dtypes(include='number').columns)
        
        if st.button("Calcular OLS") and feats:
            X = sm.add_constant(df_work[feats])
            model = sm.OLS(df_work[target], X).fit()
            st.text(model.summary())
            
            # Plot Resíduos
            fig = px.scatter(x=model.fittedvalues, y=model.resid, labels={'x':'Ajustado', 'y':'Resíduo'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    with t_anova:
        y_an = st.selectbox("Y (Resposta):", df_work.select_dtypes(include='number').columns, key="ay")
        x_an = st.multiselect("Fatores:", df_work.columns, key="ax")
        
        if st.button("Calcular ANOVA") and x_an:
            f = f"{y_an} ~ {' + '.join(x_an)}"
            model = ols(f, data=df_work).fit()
            table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(table.style.format("{:.4f}"))

    with t_test:
        col_grp = st.selectbox("Grupo (2 níveis):", df_work.columns)
        col_val = st.selectbox("Valor:", df_work.select_dtypes(include='number').columns)
        if st.button("Teste T"):
            grps = df_work[col_grp].unique()
            if len(grps) == 2:
                g1 = df_work[df_work[col_grp]==grps[0]][col_val]
                g2 = df_work[df_work[col_grp]==grps[1]][col_val]
                stat, p = stats.ttest_ind(g1, g2)
                st.metric("P-Valor", f"{p:.5f}")
                if p < 0.05: st.success("Diferença Significativa!")
                else: st.info("Sem diferença significativa.")
                st.plotly_chart(px.box(df_work, x=col_grp, y=col_val, color=col_grp))
            else:
                st.error("A coluna de grupo precisa ter exatamente 2 valores únicos.")

# ==============================================================================
# 5. AUTO ML
# ==============================================================================
elif menu == "🤖 AutoML (Tabular)":
    st.header("AutoML Supervisionado")
    target = st.selectbox("Alvo:", df_work.columns)
    features = st.multiselect("Features:", [c for c in df_work.columns if c != target])
    
    if st.button("Treinar Modelo"):
        X = df_work[features]
        y = df_work[target]
        
        # Check texto
        if X.select_dtypes(include=['object']).shape[1] > 0:
            st.error("Erro: Features com texto. Use a aba Engenharia para tratar.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            model = RandomForestRegressor() if is_reg else RandomForestClassifier()
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            if is_reg:
                c1.metric("R2", f"{r2_score(y_test, preds):.2%}")
                c2.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")
                st.plotly_chart(px.scatter(x=y_test, y=preds, labels={'x':'Real','y':'Previsto'}, title="Real vs Previsto"))
            else:
                c1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))
            
            download_file(model, "modelo_treinado.pkl", "Baixar Modelo (.pkl)")

# ==============================================================================
# 6. NLP STUDIO
# ==============================================================================
elif menu == "🧠 NLP Studio (Texto)":
    st.header("Processamento de Linguagem Natural")
    
    col_txt = st.selectbox("Coluna Texto:", df_work.select_dtypes(include='object').columns)
    col_tgt = st.selectbox("Coluna Alvo:", df_work.columns, index=1)
    
    if st.button("Treinar NLP"):
        df_nlp = df_work.copy()
        df_nlp['clean'] = df_nlp[col_txt].apply(dsa_limpa_texto)
        
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000)),
            ('clf', LogisticRegression())
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(df_nlp['clean'], df_nlp[col_tgt], test_size=0.2)
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        
        st.success(f"Acurácia: {acc:.2%}")
        st.session_state['nlp_model'] = pipe
        download_file(pipe, "nlp_model.pkl", "Baixar Modelo NLP")
        
    if 'nlp_model' in st.session_state:
        txt = st.text_input("Teste ao vivo:")
        if txt:
            pred = st.session_state['nlp_model'].predict([dsa_limpa_texto(txt)])[0]
            st.info(f"Previsão: {pred}")

# ==============================================================================
# 7. CLUSTERING
# ==============================================================================
elif menu == "🌀 Clustering":
    st.header("Não Supervisionado (K-Means)")
    feats = st.multiselect("Colunas Numéricas:", df_work.select_dtypes(include='number').columns)
    k = st.slider("K Clusters:", 2, 10, 3)
    
    if st.button("Executar Clustering") and feats:
        X = StandardScaler().fit_transform(df_work[feats].dropna())
        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(X)
        
        pca = PCA(n_components=2).fit_transform(X)
        fig = px.scatter(x=pca[:,0], y=pca[:,1], color=clusters.astype(str), title="PCA Clusters")
        st.plotly_chart(fig)
        
        df_work['Cluster'] = clusters
        download_file(df_work, "dados_clusterizados.csv", "Baixar CSV com Clusters")

# ==============================================================================
# 8. VISUALIZADOR PRO
# ==============================================================================
elif menu == "🎨 Visualizador Pro":
    st.header("Construtor de Gráficos")
    
    c_conf, c_plot = st.columns([1, 3])
    with c_conf:
        tipo = st.selectbox("Tipo:", ["Scatter", "Barra", "Linha", "Hist", "Box", "3D Scatter"])
        x = st.selectbox("Eixo X:", df_work.columns)
        y = st.selectbox("Eixo Y/Z:", df_work.columns, index=1)
        cor = st.selectbox("Cor:", [None] + list(df_work.columns))
        
        tema = st.selectbox("Tema:", ["plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"])
        titulo = st.text_input("Título:", f"{tipo} de {y} por {x}")
        
    with c_plot:
        try:
            if tipo == "Scatter": fig = px.scatter(df_work, x=x, y=y, color=cor, template=tema, title=titulo)
            elif tipo == "Barra": fig = px.bar(df_work, x=x, y=y, color=cor, template=tema, title=titulo)
            elif tipo == "Linha": fig = px.line(df_work, x=x, y=y, color=cor, template=tema, title=titulo)
            elif tipo == "Hist": fig = px.histogram(df_work, x=x, color=cor, template=tema, title=titulo)
            elif tipo == "Box": fig = px.box(df_work, x=x, y=y, color=cor, template=tema, title=titulo)
            elif tipo == "3D Scatter": 
                if pd.api.types.is_numeric_dtype(df_work[cor]):
                     fig = px.scatter_3d(df_work, x=x, y=y, z=cor, color=cor, template=tema, title=titulo)
                else:
                     st.warning("Para 3D, a Cor (Z) deve ser numérica.")
                     fig = px.scatter(df_work, x=x, y=y, title="Fallback 2D")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")