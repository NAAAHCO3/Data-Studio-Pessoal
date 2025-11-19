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

# --- BIBLIOTECAS CIENTÍFICAS (NOVAS) ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Bibliotecas de NLP (Texto)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Bibliotecas de Clustering (Não Supervisionado)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- CONFIGURAÇÃO GERAL ---
st.set_page_config(
    page_title="Data Studio v7 Chemometrics", 
    layout="wide", 
    page_icon="⚗️",
    initial_sidebar_state="expanded"
)

# --- CSS MINIMALISTA ---
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES UTILITÁRIAS ---
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

def download_model(model, filename="model.pkl"):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{filename}">📥 Baixar Modelo Treinado (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar CSV Atualizado",
        data=csv,
        file_name="dados_processados.csv",
        mime="text/csv"
    )

def dsa_limpa_texto(texto):
    if not isinstance(texto, str): return ""
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z\s]', '', texto.lower()).strip()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚗️ Data Studio")
    st.caption("v7.0 Chemometrics Edition")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Carregar Dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df_raw'] = df
                st.session_state['df_work'] = df.copy()
                st.session_state['file_name'] = uploaded_file.name
                st.toast("Dados carregados!", icon="✅")
        
        if 'df_work' in st.session_state:
            df_work = st.session_state['df_work']
            st.info(f"Linhas: {df_work.shape[0]} | Colunas: {df_work.shape[1]}")
            
            if st.button("🔄 Restaurar Original"):
                st.session_state['df_work'] = st.session_state['df_raw'].copy()
                st.rerun()
            
            st.markdown("---")
            # Menu Reorganizado
            menu = st.radio("Módulos:", 
                [
                    "🏠 Dashboard", 
                    "🔬 Explorador & Correlações", 
                    "🛠️ Engenharia", 
                    "⚗️ Estatística & DOE (Novo!)", 
                    "🤖 AutoML (Preditivo)", 
                    "🌀 Clustering (Grupos)", 
                    "🎨 Visualizador Pro (Relatórios)"
                ],
                index=3
            )
    else:
        menu = "Home"

# --- HOME ---
if not uploaded_file:
    st.title("Data Studio v7.0 - Chemometrics Edition")
    st.markdown("""
    ### Workstation para Ciência de Dados e Estatística Experimental.
    
    **Novidades da v7.0:**
    * ⚗️ **ANOVA & DOE:** Tabelas de variância completas (estilo Minitab) e análise de superfície.
    * 📈 **Statsmodels:** Relatórios de regressão OLS detalhados (P-valor, R-quadrado ajustado, Confiança).
    * 🧪 **Testes T:** Comparação de médias para experimentos.
    * 🎨 **Gráficos de Publicação:** Personalize cores, títulos e eixos para seus relatórios.
    
    👈 **Carregue seus dados experimentais para começar.**
    """)
    st.stop()

# --- 1. DASHBOARD ---
if menu == "🏠 Dashboard":
    st.header("Visão Geral")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", df_work.shape[0])
    c2.metric("Colunas", df_work.shape[1])
    c3.metric("Nulos", df_work.isna().sum().sum())
    st.dataframe(df_work.head(100), use_container_width=True)

# --- 2. EXPLORADOR ---
elif menu == "🔬 Explorador & Correlações":
    st.header("Exploração de Dados")
    t1, t2 = st.tabs(["Distribuições", "Matriz de Correlação"])
    
    with t1:
        col_sel = st.selectbox("Coluna:", df_work.columns)
        if pd.api.types.is_numeric_dtype(df_work[col_sel]):
            fig = px.histogram(df_work, x=col_sel, marginal="box", title=f"Distribuição: {col_sel}")
        else:
            fig = px.bar(df_work[col_sel].value_counts().head(20), title=f"Contagem: {col_sel}")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        st.markdown("### Correlação de Pearson")
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            fig = px.imshow(df_num.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.warning("Colunas numéricas insuficientes.")

# --- 3. ENGENHARIA ---
elif menu == "🛠️ Engenharia":
    st.header("Engenharia de Dados")
    t1, t2 = st.tabs(["Limpeza Básica", "Transformações"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remover Duplicatas"):
                df_work.drop_duplicates(inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
        with c2:
            col_nan = st.selectbox("Tratar Nulos:", df_work.columns)
            if st.button("Preencher (Zero/Vazio)"):
                val = 0 if pd.api.types.is_numeric_dtype(df_work[col_nan]) else ""
                df_work[col_nan].fillna(val, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
    with t2:
        col_dummy = st.selectbox("One-Hot Encode:", df_work.select_dtypes(include='object').columns)
        if st.button("Aplicar Dummies"):
            df_work = pd.get_dummies(df_work, columns=[col_dummy], drop_first=True, dtype=int)
            st.session_state['df_work'] = df_work
            st.rerun()
    download_csv(df_work)

# --- 4. ESTATÍSTICA & DOE (NOVO!) ---
elif menu == "⚗️ Estatística & DOE (Novo!)":
    st.header("Laboratório Estatístico (Statsmodels)")
    st.markdown("Ferramentas rigorosas para planejamento experimental e validação analítica.")
    
    stat_tabs = st.tabs(["📉 Regressão Linear (OLS)", "📊 ANOVA (Variância)", "🧪 Testes de Hipótese"])
    
    # --- ABA OLS (REGRESSÃO DETALHADA) ---
    with stat_tabs[0]:
        st.subheader("Regressão Linear Múltipla (Relatório Completo)")
        st.caption("Ideal para curvas de calibração e análise de coeficientes.")
        
        target_ols = st.selectbox("Variável Resposta (Y):", df_work.select_dtypes(include='number').columns, key='ols_y')
        features_ols = st.multiselect("Variáveis Preditivas (X):", df_work.select_dtypes(include='number').columns, key='ols_x')
        
        if st.button("Calcular Regressão OLS"):
            if not features_ols:
                st.error("Selecione as variáveis X.")
            else:
                try:
                    # Prepara dados
                    X = df_work[features_ols]
                    X = sm.add_constant(X) # Adiciona intercepto (constante)
                    y = df_work[target_ols]
                    
                    # Ajusta modelo
                    model = sm.OLS(y, X).fit()
                    
                    # Exibe Summary estilo Minitab/SAS
                    st.text(model.summary())
                    
                    # Gráfico de Resíduos
                    st.markdown("#### Diagnóstico de Resíduos")
                    residuals = model.resid
                    fitted = model.fittedvalues
                    
                    fig_resid = px.scatter(x=fitted, y=residuals, labels={'x':'Valor Ajustado', 'y':'Resíduo'}, title="Resíduos vs Ajustado")
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_resid, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erro no cálculo: {e}")

    # --- ABA ANOVA ---
    with stat_tabs[1]:
        st.subheader("Análise de Variância (ANOVA)")
        st.caption("Verifique a significância de fatores categóricos ou numéricos.")
        
        response_anova = st.selectbox("Resposta (Y):", df_work.select_dtypes(include='number').columns, key='anova_y')
        factors_anova = st.multiselect("Fatores (X):", df_work.columns, key='anova_x')
        
        if st.button("Gerar Tabela ANOVA"):
            if not factors_anova:
                st.error("Selecione pelo menos um fator.")
            else:
                try:
                    # Monta fórmula estilo R: "Y ~ X1 + X2"
                    formula_str = f"{response_anova} ~ {' + '.join(factors_anova)}"
                    st.info(f"Modelo: {formula_str}")
                    
                    model = ols(formula_str, data=df_work).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    st.dataframe(anova_table.style.format("{:.4f}"))
                    
                    if anova_table['PR(>F)'].min() < 0.05:
                        st.success("Pelo menos um fator é estatisticamente significativo (p < 0.05).")
                    else:
                        st.warning("Nenhum fator apresentou significância estatística.")
                        
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique se nomes de colunas têm espaços ou caracteres especiais.")

    # --- ABA TESTES T ---
    with stat_tabs[2]:
        st.subheader("Teste T de Student (2 Amostras)")
        
        col_group = st.selectbox("Coluna de Agrupamento (Categórica, 2 níveis):", df_work.select_dtypes(include=['object', 'int']).columns)
        col_val = st.selectbox("Variável Numérica:", df_work.select_dtypes(include='number').columns, key='ttest_y')
        
        if col_group:
            groups = df_work[col_group].unique()
            if len(groups) == 2:
                group1 = df_work[df_work[col_group] == groups[0]][col_val]
                group2 = df_work[df_work[col_group] == groups[1]][col_val]
                
                if st.button("Executar Teste T"):
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Média Grupo 1", f"{group1.mean():.2f}")
                    c2.metric("Média Grupo 2", f"{group2.mean():.2f}")
                    
                    st.markdown("---")
                    st.metric("P-Valor", f"{p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success("Diferença Significativa! (Rejeita-se H0)")
                    else:
                        st.info("Sem diferença estatisticamente significativa. (Não se rejeita H0)")
                    
                    fig_box = px.box(df_work, x=col_group, y=col_val, color=col_group, title="Comparação de Distribuições")
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning(f"A coluna de agrupamento deve ter exatamente 2 níveis. Encontrados: {len(groups)}")

# --- 5. AUTO ML ---
elif menu == "🤖 AutoML (Preditivo)":
    st.header("AutoML (Scikit-Learn)")
    target = st.selectbox("Alvo:", df_work.columns)
    features = st.multiselect("Features:", [c for c in df_work.columns if c != target])
    
    if st.button("Treinar"):
        X = df_work[features]
        y = df_work[target]
        
        if X.select_dtypes(include=['object']).shape[1] > 0:
            st.error("Converta textos em Engenharia primeiro.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            is_reg = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            model = RandomForestRegressor() if is_reg else RandomForestClassifier()
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if is_reg:
                st.metric("R2 Score", f"{r2_score(y_test, preds):.2%}")
                st.plotly_chart(px.scatter(x=y_test, y=preds, labels={'x':'Real', 'y':'Previsto'}, title="Real vs Previsto"))
            else:
                st.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                st.plotly_chart(px.imshow(confusion_matrix(y_test, preds), text_auto=True))
            
            download_model(model)

# --- 6. CLUSTERING ---
elif menu == "🌀 Clustering (Grupos)":
    st.header("Clustering (K-Means & PCA)")
    feats = st.multiselect("Features:", df_work.select_dtypes(include='number').columns)
    if feats:
        k = st.slider("Clusters (k):", 2, 10, 3)
        if st.button("Clusterizar"):
            X = StandardScaler().fit_transform(df_work[feats].dropna())
            clusters = KMeans(n_clusters=k).fit_predict(X)
            
            pca = PCA(n_components=2).fit_transform(X)
            fig = px.scatter(x=pca[:,0], y=pca[:,1], color=clusters.astype(str), title="PCA - Visualização dos Grupos")
            st.plotly_chart(fig)
            
            df_work['Cluster'] = clusters
            download_csv(df_work)

# --- 7. VISUALIZADOR PRO (NOVO!) ---
elif menu == "🎨 Visualizador Pro (Relatórios)":
    st.header("Criador de Gráficos para Relatórios")
    
    # Layout de 2 Colunas: Controle e Gráfico
    col_ctrl, col_plot = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("1. Dados")
        tipo = st.selectbox("Tipo:", ["Dispersão (Scatter)", "Barra", "Linha", "Histograma", "Boxplot", "Superfície 3D"])
        x_ax = st.selectbox("Eixo X:", df_work.columns)
        y_ax = st.selectbox("Eixo Y / Z:", df_work.columns, index=1)
        col_ax = st.selectbox("Legenda (Cor):", [None] + list(df_work.columns))
        
        st.markdown("---")
        st.subheader("2. Estilo (Personalizar)")
        
        custom_title = st.text_input("Título do Gráfico:", f"{tipo} de {y_ax} por {x_ax}")
        custom_x_label = st.text_input("Rótulo Eixo X:", x_ax)
        custom_y_label = st.text_input("Rótulo Eixo Y:", y_ax)
        
        color_scheme = st.selectbox("Paleta de Cores:", ["Plotly", "Viridis", "Cividis", "Inferno", "Blues", "Reds"])
        template_theme = st.selectbox("Tema de Fundo:", ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"])
        
        width_plot = st.slider("Largura:", 400, 1200, 800)
        height_plot = st.slider("Altura:", 300, 1000, 600)

    with col_plot:
        try:
            # Lógica de Plotagem
            if tipo == "Dispersão (Scatter)":
                fig = px.scatter(df_work, x=x_ax, y=y_ax, color=col_ax, title=custom_title, template=template_theme, color_continuous_scale=color_scheme)
            
            elif tipo == "Barra":
                fig = px.bar(df_work, x=x_ax, y=y_ax, color=col_ax, title=custom_title, template=template_theme, color_continuous_scale=color_scheme)
            
            elif tipo == "Linha":
                fig = px.line(df_work, x=x_ax, y=y_ax, color=col_ax, title=custom_title, template=template_theme)
            
            elif tipo == "Histograma":
                fig = px.histogram(df_work, x=x_ax, color=col_ax, title=custom_title, template=template_theme)
            
            elif tipo == "Boxplot":
                fig = px.box(df_work, x=x_ax, y=y_ax, color=col_ax, title=custom_title, template=template_theme)
            
            elif tipo == "Superfície 3D":
                # Para superfície, precisamos de 3 eixos numéricos ou interpolação
                if col_ax and pd.api.types.is_numeric_dtype(df_work[col_ax]):
                    # Scatter 3D é mais seguro para dados brutos experimentais
                    fig = px.scatter_3d(df_work, x=x_ax, y=y_ax, z=col_ax, color=col_ax, title=custom_title, template=template_theme, color_continuous_scale=color_scheme)
                else:
                    st.warning("Para 3D, selecione uma coluna numérica no campo 'Legenda (Cor)' para servir como eixo Z.")
                    fig = px.scatter(df_work, x=x_ax, y=y_ax, title="Fallback 2D")

            # Atualizações de Estilo Fino
            fig.update_layout(
                xaxis_title=custom_x_label,
                yaxis_title=custom_y_label,
                width=width_plot,
                height=height_plot,
                font=dict(size=14)
            )
            
            # Exibir
            st.plotly_chart(fig)
            
            # Botão de Download de Imagem Estática (Para relatórios)
            # Nota: Streamlit permite download via menu do Plotly, mas podemos adicionar logica extra se necessário
            st.info("💡 Dica: Clique na câmera 📷 no topo do gráfico para baixar como PNG transparente.")
            
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")