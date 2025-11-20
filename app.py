# Data Insight Pro - Ferramenta de BI e Análise de Dados
# Baseado na estrutura robusta solicitada, adaptado para dados dinâmicos (CSV/Excel).

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO

# --- Configuração da Página (Deve ser a primeira chamada) ---
st.set_page_config(
    page_title="Data Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilização Customizada (Baseada no seu exemplo) ---
def apply_custom_style():
    st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #FAFAFA;
        }
        /* Estilo para cards de métricas */
        div.css-1r6slb0.e1tzin5v2 {
            background-color: #262730;
            border: 1px solid #444;
            padding: 10px;
            border-radius: 5px;
        }
        /* Ajuste de sidebar */
        section[data-testid="stSidebar"] {
            background-color: #262730;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Funções Auxiliares ---

@st.cache_data
def load_data(file):
    """Carrega dados de CSV ou Excel com cache para performance."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def convert_df_to_csv(df):
    """Converte DF para CSV para download."""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Converte DF para Excel para download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# --- Aplicação Principal ---

def main():
    apply_custom_style()
    
    # --- Sidebar: Upload e Configurações ---
    st.sidebar.markdown("## 📂 Carregar Dados")
    uploaded_file = st.sidebar.file_uploader("Arraste seu arquivo CSV ou Excel", type=["csv", "xlsx"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data Insight Pro v1.0\nFerramenta completa de BI.")

    # Tela de Boas-vindas se não houver arquivo
    if uploaded_file is None:
        st.title("📊 Data Insight Pro")
        st.subheader("Sua Central de Inteligência de Dados")
        st.markdown("""
        Bem-vindo à sua ferramenta robusta de análise. Para começar:
        
        1.  Utilize a barra lateral para fazer **Upload** do seu dataset (CSV ou Excel).
        2.  Navegue pelas abas para **Limpar**, **Analisar** e **Visualizar** seus dados.
        
        Esta ferramenta inclui:
        * Tratamento de valores nulos e duplicatas.
        * Estatísticas descritivas automáticas.
        * Criação de gráficos dinâmicos (Plotly).
        * Filtros interativos.
        """)
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        return

    # --- Carregamento dos Dados ---
    df_original = load_data(uploaded_file)
    
    if df_original is not None:
        # Usamos session_state para manter o dataframe processado na memória durante interações
        if 'df_cleaned' not in st.session_state:
            st.session_state.df_cleaned = df_original.copy()

        # Título Principal
        st.title(f"Análise: {uploaded_file.name}")
        
        # Abas de Navegação
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Visão Geral & Dados", 
            "🧹 Tratamento & Limpeza", 
            "📈 Visualização & BI", 
            "🧠 Análise Avançada"
        ])

        # --- ABA 1: VISÃO GERAL ---
        with tab1:
            st.header("Visão Geral do Dataset")
            
            # KPIs Rápidos
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Linhas", st.session_state.df_cleaned.shape[0])
            c2.metric("Colunas", st.session_state.df_cleaned.shape[1])
            c3.metric("Duplicatas", st.session_state.df_cleaned.duplicated().sum())
            c4.metric("Valores Nulos", st.session_state.df_cleaned.isnull().sum().sum())
            
            st.markdown("### Amostra dos Dados")
            st.dataframe(st.session_state.df_cleaned.head(10), use_container_width=True)
            
            st.markdown("### Tipos de Dados e Estrutura")
            buffer = pd.DataFrame(st.session_state.df_cleaned.dtypes, columns=['Tipo de Dado']).astype(str)
            st.dataframe(buffer.T, use_container_width=True)

        # --- ABA 2: TRATAMENTO DE DADOS ---
        with tab2:
            st.header("Limpeza e Manipulação")
            
            col_clean1, col_clean2 = st.columns(2)
            
            with col_clean1:
                st.subheader("Remoção de Dados")
                if st.button("🗑️ Remover Linhas Duplicadas"):
                    rows_before = st.session_state.df_cleaned.shape[0]
                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop_duplicates()
                    rows_after = st.session_state.df_cleaned.shape[0]
                    st.success(f"Removidas {rows_before - rows_after} linhas duplicadas!")
                    st.rerun()

            with col_clean2:
                st.subheader("Tratamento de Nulos")
                null_action = st.selectbox("Como lidar com valores vazios?", 
                                         ["Selecione...", "Remover linhas com N/A", "Preencher com 0", "Preencher com a Média (Numéricos)"])
                
                if st.button("Aplicar Tratamento"):
                    if null_action == "Remover linhas com N/A":
                        st.session_state.df_cleaned = st.session_state.df_cleaned.dropna()
                        st.success("Linhas com valores nulos removidas.")
                    elif null_action == "Preencher com 0":
                        st.session_state.df_cleaned = st.session_state.df_cleaned.fillna(0)
                        st.success("Nulos preenchidos com 0.")
                    elif null_action == "Preencher com a Média (Numéricos)":
                        num_cols = st.session_state.df_cleaned.select_dtypes(include=np.number).columns
                        st.session_state.df_cleaned[num_cols] = st.session_state.df_cleaned[num_cols].fillna(st.session_state.df_cleaned[num_cols].mean())
                        st.success("Nulos numéricos preenchidos com a média.")
                    st.rerun()
            
            st.markdown("---")
            st.subheader("Visualizar Dados Atuais (Pós-Tratamento)")
            st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)

        # --- ABA 3: VISUALIZAÇÃO & BI ---
        with tab3:
            st.header("Construtor de Gráficos")
            
            # Layout de controles
            c_chart1, c_chart2, c_chart3 = st.columns(3)
            
            with c_chart1:
                chart_type = st.selectbox("Tipo de Gráfico", ["Barra", "Linha", "Dispersão (Scatter)", "Histograma", "Pizza", "Boxplot"])
            
            all_columns = st.session_state.df_cleaned.columns.tolist()
            
            with c_chart2:
                x_axis = st.selectbox("Eixo X (Categoria/Tempo)", all_columns)
            
            with c_chart3:
                # Para histograma e pizza, Y pode ser opcional ou contagem
                y_axis = st.selectbox("Eixo Y (Valores)", all_columns, index=1 if len(all_columns) > 1 else 0)
            
            # Opções Extras
            with st.expander("🎨 Opções Avançadas (Cor, Agrupamento)"):
                color_col = st.selectbox("Agrupar por Cor (Legenda)", ["Nenhum"] + all_columns)
                color_opt = None if color_col == "Nenhum" else color_col
            
            # Geração dos Gráficos
            st.markdown("---")
            
            try:
                if chart_type == "Barra":
                    fig = px.bar(st.session_state.df_cleaned, x=x_axis, y=y_axis, color=color_opt, template="plotly_dark", barmode='group')
                elif chart_type == "Linha":
                    fig = px.line(st.session_state.df_cleaned, x=x_axis, y=y_axis, color=color_opt, template="plotly_dark")
                elif chart_type == "Dispersão (Scatter)":
                    fig = px.scatter(st.session_state.df_cleaned, x=x_axis, y=y_axis, color=color_opt, template="plotly_dark", size=y_axis if pd.api.types.is_numeric_dtype(st.session_state.df_cleaned[y_axis]) else None)
                elif chart_type == "Histograma":
                    fig = px.histogram(st.session_state.df_cleaned, x=x_axis, color=color_opt, template="plotly_dark")
                elif chart_type == "Pizza":
                    fig = px.pie(st.session_state.df_cleaned, names=x_axis, values=y_axis, template="plotly_dark")
                elif chart_type == "Boxplot":
                    fig = px.box(st.session_state.df_cleaned, x=x_axis, y=y_axis, color=color_opt, template="plotly_dark")
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Não foi possível gerar o gráfico com as colunas selecionadas. Erro: {e}")

        # --- ABA 4: ANÁLISE AVANÇADA & EXPORTAÇÃO ---
        with tab4:
            st.header("Estatísticas e Exportação")
            
            c_adv1, c_adv2 = st.columns([2, 1])
            
            with c_adv1:
                st.subheader("Estatísticas Descritivas")
                st.dataframe(st.session_state.df_cleaned.describe(), use_container_width=True)
                
                st.subheader("Matriz de Correlação (Numérica)")
                try:
                    numeric_df = st.session_state.df_cleaned.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        corr = numeric_df.corr()
                        fig_corr = px.imshow(corr, text_auto=True, template="plotly_dark", aspect="auto")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Sem colunas numéricas suficientes para correlação.")
                except:
                    st.info("Não foi possível calcular correlação.")

            with c_adv2:
                st.subheader("📥 Exportar Dados Tratados")
                st.write("Baixe o dataset após as limpezas realizadas.")
                
                # Botão CSV
                csv_data = convert_df_to_csv(st.session_state.df_cleaned)
                st.download_button(
                    label="Baixar CSV",
                    data=csv_data,
                    file_name="dados_tratados.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Botão Excel
                excel_data = convert_df_to_excel(st.session_state.df_cleaned)
                st.download_button(
                    label="Baixar Excel (.xlsx)",
                    data=excel_data,
                    file_name="dados_tratados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()