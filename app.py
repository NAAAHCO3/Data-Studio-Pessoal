"""
Enterprise Analytics — BI Edition (UI Overhaul v3.0)
Author: Gemini Advanced
Version: 3.0 (Modern UI + Fix Rerun + Interactive Onboarding)

Changes v3.0:
- FIX: Replaced deprecated `st.experimental_rerun()` with `st.rerun()`.
- UI: Added CSS for "Card" styling and gradients.
- UX: "Empty State" landing page with big action buttons.
- FEAT: Immediate "Quick Dashboard" upon loading data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
from datetime import datetime
from typing import List, Dict, Optional

# ML & Stats Imports
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

# PDF Support
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Logging
logging.basicConfig(level=logging.INFO)

# ---------------------------
# CONFIG & STYLES
# ---------------------------
st.set_page_config(
    page_title="Data Studio Pro", 
    layout="wide", 
    page_icon="✨", 
    initial_sidebar_state="expanded"
)

# Modern CSS Styles
st.markdown("""
<style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards Styles */
    .st-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .dark-theme .st-card {
        background-color: #1f2937;
        border-color: #374151;
        color: white;
    }

    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }

    /* Header Gradients */
    .gradient-text {
        background: linear-gradient(45deg, #2563eb, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
    
    /* Steps */
    .step-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        padding: 10px;
        background: #f0f9ff;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
    }
    .step-icon {
        font-size: 1.5rem;
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# UTILITIES
# ---------------------------
def try_read_csv(file_obj, encoding_list=("utf-8", "latin1", "cp1252")):
    for enc in encoding_list:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            continue
    file_obj.seek(0)
    return pd.read_csv(file_obj, engine="python", encoding_errors="ignore")

def safe_read(file):
    try:
        if hasattr(file, "read"):
            name = getattr(file, "name", "")
            if name.lower().endswith(".csv"):
                return try_read_csv(file)
            else:
                file.seek(0)
                return pd.read_excel(file, engine="openpyxl")
        else:
            # Local path fallback
            if str(file).endswith(".csv"):
                return pd.read_csv(file)
            return pd.read_excel(file)
    except Exception as e:
        raise e

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

# PDF Report Generator
def generate_pdf_report(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Relatório Executivo", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # KPI Grid
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Resumo dos Dados", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    
    col_width = 45
    pdf.cell(col_width, 10, f"Linhas: {kpis['rows']}", 1)
    pdf.cell(col_width, 10, f"Colunas: {kpis['cols']}", 1)
    pdf.cell(col_width, 10, f"Nulos: {kpis['nulls']}", 1)
    pdf.cell(col_width, 10, f"Duplicatas: {kpis['dups']}", 1)
    pdf.ln(15)

    # Text Stats
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Estatísticas Principais (Numéricas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    desc = df.describe().T.head(10).reset_index()
    if not desc.empty:
        pdf.set_font("Helvetica", "", 8)
        for _, row in desc.iterrows():
            line = f"{row['index'][:15]}: Mean={row['mean']:.2f} | Min={row['min']:.2f} | Max={row['max']:.2f}"
            pdf.cell(0, 6, line, ln=True)
    
    pdf.ln(10)
    # Charts Info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Anexos Visuais", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for ch in charts:
        pdf.bullet = "-"
        pdf.multi_cell(0, 8, f"• Gráfico: {ch.get('title', 'Sem Título')} ({ch.get('type','')})\n  Nota: {ch.get('note','')}")
        pdf.ln(2)
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------
# SESSION STATE
# ---------------------------
if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
if 'df_raw' not in st.session_state: st.session_state['df_raw'] = pd.DataFrame()
if 'report_charts' not in st.session_state: st.session_state['report_charts'] = []
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = "🏠 Início"

# ---------------------------
# UI COMPONENTS
# ---------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 📊 Data Studio Pro")
        
        # File Uploader in Sidebar
        uploaded = st.file_uploader("📁 Carregar Arquivo", type=['csv', 'xlsx'])
        
        if uploaded:
            try:
                # Load only if different
                if st.session_state.get('last_file') != uploaded.name:
                    df = safe_read(uploaded)
                    df = clean_colnames(df)
                    st.session_state['df'] = df
                    st.session_state['df_raw'] = df.copy()
                    st.session_state['last_file'] = uploaded.name
                    st.toast("Arquivo carregado com sucesso!", icon="✅")
            except Exception as e:
                st.error(f"Erro: {e}")

        st.markdown("---")
        
        # Navigation
        options = ["🏠 Início", "🛠️ Data Studio (ETL)", "📈 Visual Studio", "🤖 ML & IA", "📑 Relatório & Export"]
        page = st.radio("Navegação", options)
        
        st.markdown("---")
        if st.button("🗑️ Resetar Tudo"):
            st.session_state['df'] = pd.DataFrame()
            st.session_state['report_charts'] = []
            st.rerun()
            
        return page

def render_landing_page():
    st.markdown('<h1 class="gradient-text">Bem-vindo ao Data Studio</h1>', unsafe_allow_html=True)
    st.markdown("### Sua plataforma central para transformar dados em inteligência.")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        # Check if data exists
        if st.session_state['df'].empty:
            st.info("👈 Comece carregando um arquivo na barra lateral ou use dados de exemplo abaixo.")
            
            if st.button("🚀 Carregar Dados de Exemplo (Demo)", type="primary", use_container_width=True):
                # Create dummy data
                data = {
                    'data_venda': pd.date_range(start='1/1/2023', periods=100),
                    'categoria': np.random.choice(['Eletrônicos', 'Roupas', 'Casa', 'Jardim'], 100),
                    'vendas': np.random.randint(100, 5000, 100),
                    'custo': np.random.randint(50, 2000, 100),
                    'satisfacao_cliente': np.random.choice(['Alta', 'Média', 'Baixa'], 100),
                    'comentarios': np.random.choice(['Ótimo produto', 'Demorou a chegar', 'Recomendo', 'Ruim'], 100)
                }
                df = pd.DataFrame(data)
                df['lucro'] = df['vendas'] - df['custo']
                st.session_state['df'] = df
                st.session_state['df_raw'] = df.copy()
                st.rerun()
        
        else:
            # Quick Summary Dashboard if data exists
            df = st.session_state['df']
            st.success(f"Dataset Ativo: {df.shape[0]} linhas | {df.shape[1]} colunas")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Linhas", df.shape[0])
            k2.metric("Variáveis Numéricas", len(df.select_dtypes(include=np.number).columns))
            k3.metric("Variáveis Texto", len(df.select_dtypes(include='object').columns))
            
            st.markdown("#### 👀 Espiada nos Dados")
            st.dataframe(df.head(10), use_container_width=True)

# ---------------------------
# MODULES
# ---------------------------

def page_data_studio():
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados carregados."); return

    st.header("🛠️ Data Studio (Preparação)")
    
    t1, t2, t3 = st.tabs(["➕ Criar & Calcular", "🧹 Limpeza & Filtros", "⚙️ Preparação IA"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Calculadora de Colunas")
            col_a = st.selectbox("Coluna A", df.select_dtypes(include=np.number).columns, key='ca')
            op = st.selectbox("Operação", ["+", "-", "*", "/"])
            use_val = st.checkbox("Usar valor fixo?")
            val_b = 0
            col_b = None
            if use_val:
                val_b = st.number_input("Valor B", value=1.0)
            else:
                col_b = st.selectbox("Coluna B", df.select_dtypes(include=np.number).columns, key='cb')
            
            name = st.text_input("Nome Nova Coluna", "resultado_calc")
            
            if st.button("Calcular"):
                try:
                    if use_val:
                        if op=="+": df[name] = df[col_a] + val_b
                        if op=="-": df[name] = df[col_a] - val_b
                        if op=="*": df[name] = df[col_a] * val_b
                        if op=="/": df[name] = df[col_a] / val_b
                    else:
                        if op=="+": df[name] = df[col_a] + df[col_b]
                        if op=="-": df[name] = df[col_a] - df[col_b]
                        if op=="*": df[name] = df[col_a] * df[col_b]
                        if op=="/": df[name] = df[col_a] / df[col_b]
                    st.session_state['df'] = df
                    st.success("Coluna Criada!")
                    st.rerun()
                except Exception as e: st.error(e)
        
        with c2:
            st.subheader("Extrair de Data")
            date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64) or 'data' in c or 'date' in c]
            if date_cols:
                dcol = st.selectbox("Coluna Data", date_cols)
                comp = st.selectbox("Extrair", ["Ano", "Mês", "Dia", "Dia da Semana"])
                if st.button("Extrair"):
                    try:
                        if not np.issubdtype(df[dcol].dtype, np.datetime64):
                            df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
                        
                        new_n = f"{dcol}_{comp}"
                        if comp == "Ano": df[new_n] = df[dcol].dt.year
                        elif comp == "Mês": df[new_n] = df[dcol].dt.month
                        elif comp == "Dia": df[new_n] = df[dcol].dt.day
                        elif comp == "Dia da Semana": df[new_n] = df[dcol].dt.day_name()
                        st.session_state['df'] = df
                        st.success("Extraído!")
                        st.rerun()
                    except Exception as e: st.error(e)
            else:
                st.info("Nenhuma coluna de data detectada.")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tratar Nulos")
            act = st.radio("Ação", ["Remover Linhas", "Preencher com 0", "Preencher com Média"])
            if st.button("Aplicar Correção"):
                if "Remover" in act: df = df.dropna()
                elif "0" in act: df = df.fillna(0)
                elif "Média" in act: 
                    nums = df.select_dtypes(include=np.number).columns
                    df[nums] = df[nums].fillna(df[nums].mean())
                st.session_state['df'] = df
                st.success("Feito!")
                st.rerun()
        
        with c2:
            st.subheader("Filtros Rápidos")
            fcol = st.selectbox("Filtrar por", df.columns)
            vals = df[fcol].unique()
            if len(vals) < 50:
                sel = st.multiselect("Manter valores:", vals, default=vals)
                if st.button("Aplicar Filtro"):
                    st.session_state['df'] = df[df[fcol].isin(sel)]
                    st.rerun()
            else:
                st.info("Muitos valores únicos. Filtro de texto não disponível nesta versão.")

    with t3:
        st.subheader("Preparação para IA (Machine Learning)")
        st.info("Esses passos são essenciais se você for treinar modelos.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Dummies (Texto -> Número)**")
            cats = df.select_dtypes(include=['object', 'category']).columns
            if len(cats) > 0:
                target_d = st.multiselect("Colunas de Texto", cats)
                if st.button("Gerar Dummies"):
                    df = pd.get_dummies(df, columns=target_d, drop_first=True, dtype=int)
                    st.session_state['df'] = df
                    st.success("Transformado!")
                    st.rerun()
            else:
                st.caption("Sem colunas de texto.")

        with c2:
            st.markdown("**2. Escalar (Normalizar)**")
            nums = df.select_dtypes(include=np.number).columns
            target_s = st.multiselect("Colunas Numéricas", nums)
            if st.button("Aplicar Standard Scaler"):
                scaler = StandardScaler()
                df[target_s] = scaler.fit_transform(df[target_s])
                st.session_state['df'] = df
                st.success("Escalonado!")
                st.rerun()

def page_visual_studio():
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return
    
    st.header("📈 Visual Studio")
    
    l, r = st.columns([1, 2])
    
    with l:
        with st.container(border=True):
            st.subheader("Configuração")
            chart_type = st.selectbox("Tipo", ["Barras", "Linha", "Scatter", "Pizza", "Histograma", "Boxplot", "Heatmap"])
            
            x_col = st.selectbox("Eixo X", df.columns)
            y_col = st.selectbox("Eixo Y", [None] + list(df.select_dtypes(include=np.number).columns))
            color = st.selectbox("Cor (Agrupamento)", [None] + list(df.columns))
            
            st.markdown("---")
            title = st.text_input("Título do Gráfico", f"{chart_type} - {x_col}")
            
            if st.button("🎨 Gerar Gráfico", type="primary", use_container_width=True):
                try:
                    if chart_type == "Barras":
                        fig = px.bar(df, x=x_col, y=y_col, color=color, title=title, template="plotly_white")
                    elif chart_type == "Linha":
                        fig = px.line(df, x=x_col, y=y_col, color=color, title=title, template="plotly_white")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title, template="plotly_white")
                    elif chart_type == "Histograma":
                        fig = px.histogram(df, x=x_col, color=color, title=title, template="plotly_white")
                    elif chart_type == "Boxplot":
                        fig = px.box(df, x=x_col, y=y_col, color=color, title=title, template="plotly_white")
                    elif chart_type == "Pizza":
                        fig = px.pie(df, names=x_col, values=y_col, title=title)
                    elif chart_type == "Heatmap":
                        fig = px.imshow(df.select_dtypes(include=np.number).corr(), text_auto=True, title=title)
                    
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {'title': title, 'type': chart_type}
                except Exception as e:
                    st.error(f"Erro ao plotar: {e}")

    with r:
        if st.session_state.get('last_fig'):
            st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
            
            with st.expander("Opções do Gráfico"):
                note = st.text_area("Anotação para Relatório")
                if st.button("➕ Adicionar ao Relatório Final"):
                    st.session_state['report_charts'].append({
                        "fig": st.session_state['last_fig'],
                        "title": st.session_state['last_meta']['title'],
                        "type": st.session_state['last_meta']['type'],
                        "note": note
                    })
                    st.success("Salvo no relatório!")
        else:
            st.info("Configure o gráfico à esquerda e clique em 'Gerar' para visualizar aqui.")

def page_ml():
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    st.header("🤖 Assistente de Machine Learning")
    st.markdown("Crie modelos preditivos de forma simplificada.")
    
    mode = st.radio("Objetivo", ["Previsão Numérica (Regressão)", "Classificação (Categorias)", "Agrupamento (Clusterização)"], horizontal=True)
    
    if "Agrupamento" in mode:
        st.subheader("K-Means Clustering")
        feats = st.multiselect("Colunas para Agrupar", df.select_dtypes(include=np.number).columns)
        k = st.slider("Número de Grupos", 2, 8, 3)
        
        if st.button("Executar Agrupamento") and feats:
            X = df[feats].dropna()
            model = KMeans(n_clusters=k)
            # Pipeline simples
            X_scaled = StandardScaler().fit_transform(X)
            clusters = model.fit_predict(X_scaled)
            
            df['Cluster_Gerado'] = clusters
            st.session_state['df'] = df
            st.success("Agrupamento concluído! Coluna 'Cluster_Gerado' adicionada.")
            
            # Viz
            pca = PCA(2)
            res = pca.fit_transform(X_scaled)
            fig = px.scatter(x=res[:,0], y=res[:,1], color=clusters.astype(str), title="Visualização dos Clusters (PCA)")
            st.plotly_chart(fig, use_container_width=True)

    else:
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("Coluna Alvo (O que prever?)", df.columns)
        with c2:
            features = st.multiselect("Variáveis (O que usar?)", [c for c in df.columns if c != target])
        
        if st.button("Treinar Modelo") and features:
            X = df[features]
            y = df[target]
            
            # Auto Pipeline
            num_cols = X.select_dtypes(include=np.number).columns
            cat_cols = X.select_dtypes(include=['object','category']).columns
            
            pre = ColumnTransformer([
                ('num', SimpleImputer(strategy='median'), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ])
            
            if "Regressão" in mode:
                model = Pipeline([('pre', pre), ('algo', RandomForestRegressor())])
                metric = "R²"
            else:
                model = Pipeline([('pre', pre), ('algo', RandomForestClassifier())])
                metric = "Acurácia"
                
            # Train/Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            st.balloons()
            st.metric(f"Performance do Modelo ({metric})", f"{score:.2f}")
            st.success("Modelo treinado com sucesso!")

def page_export():
    st.header("📑 Relatório & Exportação")
    
    df = st.session_state['df']
    charts = st.session_state.get('report_charts', [])
    
    t1, t2 = st.tabs(["Visualizar Relatório", "Downloads"])
    
    with t1:
        if not charts:
            st.info("Nenhum gráfico adicionado ao relatório ainda. Vá ao Visual Studio.")
        else:
            if st.button("Limpar Relatório"):
                st.session_state['report_charts'] = []
                st.rerun()
            
            for i, item in enumerate(charts):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.plotly_chart(item['fig'], use_container_width=True)
                    with c2:
                        st.markdown(f"**{item['title']}**")
                        st.caption(item['note'])
                        if st.button(f"Remover", key=f"del_{i}"):
                            st.session_state['report_charts'].pop(i)
                            st.rerun()

    with t2:
        kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📄 PDF Executivo")
            if st.button("Gerar PDF"):
                try:
                    pdf_data = generate_pdf_report(df, charts, kpis)
                    st.download_button("📥 Baixar PDF", pdf_data, "relatorio_dados.pdf", "application/pdf")
                except Exception as e:
                    st.error(f"Erro na geração do PDF: {e}")
        
        with c2:
            st.markdown("### 💾 Dados Processados")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV", csv, "dados_processados.csv", "text/csv")
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Baixar Excel", buffer.getvalue(), "dados_processados.xlsx")

# ---------------------------
# MAIN ROUTER
# ---------------------------
def main():
    page = render_sidebar()
    
    if page == "🏠 Início":
        render_landing_page()
    elif page == "🛠️ Data Studio (ETL)":
        page_data_studio()
    elif page == "📈 Visual Studio":
        page_visual_studio()
    elif page == "🤖 ML & IA":
        page_ml()
    elif page == "📑 Relatório & Export":
        page_export()

if __name__ == "__main__":
    main()