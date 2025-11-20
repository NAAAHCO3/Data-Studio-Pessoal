import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import base64
import re
import unicodedata
import logging
from io import BytesIO
from datetime import datetime

# --- BIBLIOTECAS DO APP EXEMPLO (PDF & DADOS) ---
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --- BIBLIOTECAS CIENTÍFICAS ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# --- MACHINE LEARNING MODERNO ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Data Studio X", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# BLOCO 1: ESTILIZAÇÃO (TEMA DSA)
# ==============================================================================
def set_custom_theme():
    """Injeta o CSS profissional baseado no app da DSA."""
    card_bg_color = "#262730"
    text_color = "#FAFAFA"
    gold_color = "#E1C16E" 
    dark_text = "#1E1E1E"
    
    css = f"""
    <style>
        /* KPI Cards */
        .metric-card {{
            background-color: {card_bg_color};
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #444;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 10px;
        }}
        .metric-card h3 {{ margin: 0; font-size: 1.1rem; color: #AAA; font-weight: normal; }}
        .metric-card h2 {{ margin: 10px 0 0 0; font-size: 2rem; color: {text_color}; font-weight: bold; }}
        .metric-card .delta {{ font-size: 0.9rem; color: #4CAF50; margin-top: 5px; }}
        
        /* Ajuste Multiselect Tags */
        [data-baseweb="tag"] {{
            background-color: {gold_color} !important;
            color: {dark_text} !important;
        }}
        
        /* Tabelas */
        div[data-testid="stDataFrame"] {{ border: 1px solid #444; }}
        
        /* Headers */
        h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# BLOCO 2: FUNÇÕES UTILITÁRIAS & PDF
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_data(file):
    try:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None

def to_bytes(obj):
    bio = BytesIO()
    joblib.dump(obj, bio)
    bio.seek(0)
    return bio.read()

def dsa_limpa_texto(texto):
    if not isinstance(texto, str): return ""
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z\s]', '', texto.lower()).strip()

def generate_pdf_report(df, title="Relatório Analítico"):
    """Gera um PDF genérico com resumo dos dados atuais."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Título
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Total de Registros: {len(df)} | Colunas: {len(df.columns)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Resumo Estatístico (Top 5 colunas numéricas para caber)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Resumo Estatístico (Principais Variáveis):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    num_df = df.select_dtypes(include='number').describe().T.head(10).reset_index()
    num_df = num_df[['index', 'mean', 'min', 'max']]
    num_df.columns = ['Variavel', 'Media', 'Min', 'Max']
    
    # Tabela
    pdf.set_font("Helvetica", "B", 9)
    col_w = [50, 40, 40, 40]
    for i, h in enumerate(num_df.columns):
        pdf.cell(col_w[i], 8, h, 1, align='C')
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 9)
    for _, row in num_df.iterrows():
        for i, val in enumerate(row):
            # Tratamento seguro de string/float
            if isinstance(val, float): txt = f"{val:.2f}"
            else: txt = str(val)
            # Encode latin-1 safe
            safe_txt = txt.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(col_w[i], 7, safe_txt, 1, align='C')
        pdf.ln()
        
    return bytes(pdf.output())

# ==============================================================================
# BLOCO 3: MÓDULOS DE MACHINE LEARNING (PIPELINES)
# ==============================================================================

def train_ml_pipeline(X, y, algo_type, is_reg):
    """Pipeline Profissional com ColumnTransformer e RandomizedSearchCV."""
    
    # Preprocessor Automático
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns
    
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    # Modelo
    if is_reg:
        model = RandomForestRegressor() if algo_type == 'Random Forest' else GradientBoostingRegressor()
    else:
        model = RandomForestClassifier() if algo_type == 'Random Forest' else GradientBoostingClassifier()
        
    final_pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    
    # Otimização leve (pode expandir)
    param_grid = {'clf__n_estimators': [50, 100, 200]} 
    search = RandomizedSearchCV(final_pipe, param_grid, n_iter=3, cv=3, verbose=0)
    search.fit(X, y)
    
    return search.best_estimator_, search.best_score_

# ==============================================================================
# BLOCO 4: APLICAÇÃO PRINCIPAL
# ==============================================================================

def main():
    set_custom_theme()
    
    # --- SIDEBAR ---
    st.sidebar.markdown("""
    <div style="background-color:#00CC96; padding: 10px; border-radius: 5px; text-align: center;">
        <h3 style="color:white; margin:0; font-weight:bold;">Data Studio X</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("📂 Carregar Dados (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Sessão Persistente
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df_raw'] = df
                st.session_state['df_work'] = df.copy()
                st.session_state['file_name'] = uploaded_file.name
        
        if 'df_work' in st.session_state:
            df_work = st.session_state['df_work']
            
            st.sidebar.info(f"📊 {df_work.shape[0]} Linhas | {df_work.shape[1]} Colunas")
            if st.sidebar.button("🔄 Resetar Tudo"):
                st.session_state['df_work'] = st.session_state['df_raw'].copy()
                st.rerun()
                
            st.sidebar.markdown("---")
            menu = st.sidebar.radio("Navegação:", [
                "🏠 Dashboard", 
                "🛠️ Engenharia (Limpeza)", 
                "⚗️ Estatística (Chemometrics)", 
                "🤖 AutoML (Preditivo)", 
                "🧠 NLP Studio", 
                "🌀 Clustering",
                "🎨 Visualizador Pro"
            ])
            
            # --- EXPORTAÇÃO NA SIDEBAR ---
            st.sidebar.markdown("---")
            with st.sidebar.expander("📥 Exportação Rápida"):
                csv = df_work.to_csv(index=False).encode('utf-8')
                st.download_button("Salvar CSV", csv, "dados_editados.csv", "text/csv")
                
                if st.button("Gerar Relatório PDF"):
                    pdf_data = generate_pdf_report(df_work)
                    st.download_button("Baixar PDF", pdf_data, "relatorio_analitico.pdf", "application/pdf")

    else:
        menu = "Home"

    # --- CONTEÚDO DAS PÁGINAS ---
    
    if menu == "Home":
        st.title("Data Studio X - Versão Definitiva")
        st.markdown("### A Workstation Completa para Cientistas de Dados e Químicos.")
        st.info("Carregue um arquivo na barra lateral para começar.")

    # --- 1. DASHBOARD (ESTILO DSA) ---
    elif menu == "🏠 Dashboard":
        st.title("Visão Geral & KPIs")
        
        # Cálculos
        n_rows = df_work.shape[0]
        n_cols = df_work.shape[1]
        n_miss = df_work.isna().sum().sum()
        mem_usage = df_work.memory_usage(deep=True).sum() / 1024**2
        
        # KPI Cards HTML
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><h3>Registros</h3><h2>{n_rows:,}</h2><div class="delta">Linhas totais</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><h3>Variáveis</h3><h2>{n_cols}</h2><div class="delta">Colunas</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><h3>Dados Faltantes</h3><h2>{n_miss}</h2><div class="delta" style="color:orange">Células vazias</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><h3>Memória</h3><h2>{mem_usage:.1f} MB</h2><div class="delta">Uso de RAM</div></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        t1, t2 = st.tabs(["🔍 Amostra de Dados", "📊 Tipos e Estrutura"])
        with t1: st.dataframe(df_work.head(100), use_container_width=True)
        with t2: st.write(df_work.dtypes.astype(str))

    # --- 2. ENGENHARIA (COMPLETA/RESTAURADA) ---
    elif menu == "🛠️ Engenharia (Limpeza)":
        st.header("Engenharia de Dados Manual")
        t1, t2, t3 = st.tabs(["Estrutura", "Limpeza", "Transformação"])
        
        with t1:
            c1, c2 = st.columns(2)
            with c1: # Renomear
                ren_col = st.selectbox("Renomear:", df_work.columns)
                new_name = st.text_input("Novo nome:", value=ren_col)
                if st.button("Aplicar Nome"):
                    df_work.rename(columns={ren_col: new_name}, inplace=True)
                    st.session_state['df_work'] = df_work
                    st.rerun()
            with c2: # Tipos
                type_col = st.selectbox("Mudar Tipo:", df_work.columns)
                to_type = st.selectbox("Para:", ["Numérico", "Texto", "Data"])
                if st.button("Converter"):
                    try:
                        if to_type=="Numérico": df_work[type_col] = pd.to_numeric(df_work[type_col], errors='coerce')
                        elif to_type=="Data": df_work[type_col] = pd.to_datetime(df_work[type_col], errors='coerce')
                        else: df_work[type_col] = df_work[type_col].astype(str)
                        st.session_state['df_work'] = df_work
                        st.success("Convertido!")
                        st.rerun()
                    except Exception as e: st.error(e)
        
        with t2: # Nulos e Duplicatas
            if st.button("Remover Duplicatas"):
                df_work.drop_duplicates(inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
            
            null_col = st.selectbox("Tratar Nulos em:", df_work.columns)
            method = st.selectbox("Método:", ["0", "Média", "Mediana", "Remover Linhas"])
            if st.button("Aplicar Tratamento"):
                if method=="Remover Linhas": df_work.dropna(subset=[null_col], inplace=True)
                elif method=="0": df_work[null_col].fillna(0, inplace=True)
                elif method=="Média": df_work[null_col].fillna(df_work[null_col].mean(), inplace=True)
                elif method=="Mediana": df_work[null_col].fillna(df_work[null_col].median(), inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()

        with t3: # Mapeamento e Dummies
            op = st.radio("Opção:", ["Mapeamento Manual", "One-Hot Encoding"])
            if op == "Mapeamento Manual":
                map_col = st.selectbox("Coluna:", df_work.columns)
                uniqs = df_work[map_col].unique()
                if len(uniqs) < 30:
                    mapping = {}
                    cols = st.columns(3)
                    for i, u in enumerate(uniqs):
                        with cols[i%3]:
                            mapping[u] = st.text_input(f"{u} ->", key=f"m_{u}")
                    if st.button("Aplicar Mapa"):
                        final_map = {k: float(v) if v.replace('.','',1).isdigit() else v for k,v in mapping.items() if v}
                        df_work[map_col] = df_work[map_col].map(final_map).fillna(df_work[map_col])
                        st.session_state['df_work'] = df_work
                        st.rerun()
            else:
                dum_col = st.selectbox("Categorica:", df_work.select_dtypes(include='object').columns)
                if st.button("Gerar Dummies"):
                    df_work = pd.get_dummies(df_work, columns=[dum_col], drop_first=True, dtype=int)
                    st.session_state['df_work'] = df_work
                    st.rerun()

    # --- 3. ESTATÍSTICA (CHEMOMETRICS) ---
    elif menu == "⚗️ Estatística (Chemometrics)":
        st.header("Laboratório Estatístico (Statsmodels)")
        t1, t2, t3 = st.tabs(["Regressão OLS", "ANOVA", "Testes T"])
        
        with t1: # OLS
            y = st.selectbox("Y (Resposta):", df_work.select_dtypes(include=np.number).columns)
            x = st.multiselect("X (Preditores):", df_work.select_dtypes(include=np.number).columns)
            if st.button("Calcular Regressão") and x:
                X_ = sm.add_constant(df_work[x].dropna())
                y_ = df_work.loc[X_.index, y]
                model = sm.OLS(y_, X_).fit()
                st.code(model.summary())
                
        with t2: # ANOVA
            y_an = st.selectbox("Y:", df_work.select_dtypes(include=np.number).columns, key='any')
            x_an = st.multiselect("Fatores:", df_work.columns, key='anx')
            if st.button("Calcular ANOVA") and x_an:
                form = f"{y_an} ~ {' + '.join(x_an)}"
                mod = ols(form, data=df_work).fit()
                st.table(sm.stats.anova_lm(mod, typ=2))
        
        with t3: # T-Test
            grp = st.selectbox("Grupo (2 niveis):", df_work.columns)
            val = st.selectbox("Valor:", df_work.select_dtypes(include=np.number).columns)
            if st.button("Teste T"):
                cats = df_work[grp].unique()
                if len(cats)==2:
                    g1 = df_work[df_work[grp]==cats[0]][val]
                    g2 = df_work[df_work[grp]==cats[1]][val]
                    s, p = stats.ttest_ind(g1.dropna(), g2.dropna())
                    st.metric("P-Valor", f"{p:.5f}")
                    if p<0.05: st.success("Diferença Significativa!")
                else: st.error("Necessário coluna com exatamente 2 grupos.")

    # --- 4. AUTOML (PIPELINES) ---
    elif menu == "🤖 AutoML (Preditivo)":
        st.header("Machine Learning Pipeline")
        tgt = st.selectbox("Target:", df_work.columns)
        feats = st.multiselect("Features:", [c for c in df_work.columns if c!=tgt])
        
        if st.button("Treinar Modelo") and feats:
            X = df_work[feats]
            y = df_work[tgt]
            
            is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            algo = st.selectbox("Algoritmo", ["Random Forest", "Gradient Boosting"])
            
            with st.spinner("Otimizando Pipeline..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model, score = train_ml_pipeline(X_train, y_train, algo, is_reg)
                
                preds = model.predict(X_test)
                st.success(f"Score Validação Cruzada: {score:.4f}")
                
                c1, c2 = st.columns(2)
                if is_reg:
                    c1.metric("R2 Teste", f"{r2_score(y_test, preds):.2f}")
                    st.plotly_chart(px.scatter(x=y_test, y=preds, title="Real x Previsto"))
                else:
                    c1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                    st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))
                
                st.download_button("Baixar Modelo (.joblib)", to_bytes(model), "modelo.joblib")

    # --- 5. NLP STUDIO ---
    elif menu == "🧠 NLP Studio":
        st.header("Processamento de Texto")
        col_txt = st.selectbox("Texto:", df_work.select_dtypes(include='object').columns)
        col_lbl = st.selectbox("Label:", df_work.columns)
        
        if st.button("Treinar NLP"):
            with st.spinner("Vetorizando..."):
                texts = df_work[col_txt].apply(dsa_limpa_texto)
                pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=2000)), ('clf', LogisticRegression())])
                pipe.fit(texts, df_work[col_lbl])
                st.success("Modelo Treinado!")
                st.session_state['nlp'] = pipe
        
        if 'nlp' in st.session_state:
            txt = st.text_input("Testar frase:")
            if txt: st.info(f"Previsão: {st.session_state['nlp'].predict([dsa_limpa_texto(txt)])[0]}")

    # --- 6. CLUSTERING ---
    elif menu == "🌀 Clustering":
        st.header("K-Means Clustering")
        feats = st.multiselect("Colunas Numéricas:", df_work.select_dtypes(include=np.number).columns)
        k = st.slider("K:", 2, 10, 3)
        
        if st.button("Clusterizar") and feats:
            X = SimpleImputer().fit_transform(df_work[feats])
            X_scl = StandardScaler().fit_transform(X)
            cl = KMeans(n_clusters=k).fit_predict(X_scl)
            
            pca = PCA(2).fit_transform(X_scl)
            fig = px.scatter(x=pca[:,0], y=pca[:,1], color=cl.astype(str), title="Clusters (PCA)")
            st.plotly_chart(fig)
            df_work['Cluster'] = cl

    # --- 7. VISUALIZADOR ---
    elif menu == "🎨 Visualizador Pro":
        st.header("Gráficos Customizados")
        c1, c2 = st.columns([1,3])
        with c1:
            tipo = st.selectbox("Tipo", ["Scatter", "Bar", "Line", "Hist", "Box", "3D"])
            x = st.selectbox("X", df_work.columns)
            y = st.selectbox("Y", df_work.columns)
            cor = st.selectbox("Cor", [None]+list(df_work.columns))
            tema = st.selectbox("Tema", ["plotly", "plotly_dark", "ggplot2"])
        with c2:
            try:
                if tipo=="Scatter": fig = px.scatter(df_work, x=x, y=y, color=cor, template=tema)
                elif tipo=="Bar": fig = px.bar(df_work, x=x, y=y, color=cor, template=tema)
                elif tipo=="Line": fig = px.line(df_work, x=x, y=y, color=cor, template=tema)
                elif tipo=="Hist": fig = px.histogram(df_work, x=x, color=cor, template=tema)
                elif tipo=="Box": fig = px.box(df_work, x=x, y=y, color=cor, template=tema)
                elif tipo=="3D": fig = px.scatter_3d(df_work, x=x, y=y, z=cor, color=cor, template=tema)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(e)

if __name__ == "__main__":
    main()