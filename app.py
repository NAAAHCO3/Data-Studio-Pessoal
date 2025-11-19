import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Data Studio Pro", layout="wide", page_icon="📊")
st.title("📊 Data Studio Pro - Web Edition")
st.markdown("---")

# --- 1. CARREGAMENTO DE DADOS (SIDEBAR) ---
st.sidebar.header("📂 Importar Dados")
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo (CSV ou Excel)", type=["csv", "xlsx"])

# Função para carregar dados (com cache para performance)
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

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    # Carrega o DF inicial apenas uma vez e salva no estado da sessão
    if 'df_raw' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df_raw'] = df
            st.session_state['df_work'] = df.copy() # Cria cópia de trabalho
            st.session_state['file_name'] = uploaded_file.name
    
    # Botão para Reiniciar (caso erre na limpeza)
    if st.sidebar.button("🔄 Restaurar Original"):
        st.session_state['df_work'] = st.session_state['df_raw'].copy()
        st.rerun()

    # Define o DataFrame de trabalho atual
    df_work = st.session_state['df_work']

    # Feedback lateral
    st.sidebar.success(f"Dados: {df_work.shape[0]} linhas, {df_work.shape[1]} colunas")
    
    # --- MENU DE NAVEGAÇÃO ---
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Ferramentas:", ["🔍 Visão Geral", "🧹 Limpeza Avançada", "📈 Gráficos Interativos", "💾 Exportar"])

    # --- ABA 1: VISÃO GERAL ---
    if menu == "🔍 Visão Geral":
        st.subheader("Radiografia dos Dados")
        
        # Métricas no topo
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de Linhas", df_work.shape[0])
        c2.metric("Total de Colunas", df_work.shape[1])
        c3.metric("Duplicatas", df_work.duplicated().sum())
        c4.metric("Células Vazias", df_work.isna().sum().sum())

        # Tabela e Estatísticas
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("##### Amostra dos Dados")
            st.dataframe(df_work.head(10), use_container_width=True)
        with col_right:
            st.markdown("##### Resumo Estatístico")
            st.dataframe(df_work.describe(), use_container_width=True)

    # --- ABA 2: LIMPEZA ---
    elif menu == "🧹 Limpeza Avançada":
        st.subheader("Tratamento de Dados")
        
        col1, col2 = st.columns(2)
        
        # Bloco 1: Duplicatas
        with col1:
            st.info("🗑️ Remoção de Duplicatas")
            if df_work.duplicated().sum() > 0:
                if st.button("Remover Duplicatas"):
                    antes = df_work.shape[0]
                    df_work = df_work.drop_duplicates()
                    st.session_state['df_work'] = df_work
                    st.success(f"Removidas {antes - df_work.shape[0]} linhas duplicadas!")
                    st.rerun()
            else:
                st.success("Sem duplicatas encontradas.")

        # Bloco 2: Valores Nulos
        with col2:
            st.info("ax️ Tratamento de Nulos")
            cols_com_na = df_work.columns[df_work.isna().any()].tolist()
            
            if cols_com_na:
                col_sel = st.selectbox("Escolha a Coluna:", cols_com_na)
                metodo = st.radio("Ação:", ["Preencher com 0", "Preencher com Média", "Excluir Linhas"])
                
                if st.button("Aplicar Correção"):
                    if metodo == "Preencher com 0":
                        df_work[col_sel] = df_work[col_sel].fillna(0)
                    elif metodo == "Preencher com Média":
                        if pd.api.types.is_numeric_dtype(df_work[col_sel]):
                            df_work[col_sel] = df_work[col_sel].fillna(df_work[col_sel].mean())
                        else:
                            st.warning("Não é possível calcular média de texto.")
                    elif metodo == "Excluir Linhas":
                        df_work = df_work.dropna(subset=[col_sel])
                    
                    st.session_state['df_work'] = df_work
                    st.success("Correção aplicada com sucesso!")
                    st.rerun()
            else:
                st.success("Sem valores nulos no dataset.")

    # --- ABA 3: GRÁFICOS ---
    elif menu == "📈 Gráficos Interativos":
        st.subheader("Construtor de Gráficos")
        
        all_cols = df_work.columns.tolist()
        
        # Controles do Gráfico
        c1, c2, c3, c4 = st.columns(4)
        tipo = c1.selectbox("Tipo", ["Dispersão", "Linha", "Barra", "Histograma", "Boxplot", "Pizza"])
        x_axis = c2.selectbox("Eixo X", all_cols)
        y_axis = c3.selectbox("Eixo Y", all_cols, index=1 if len(all_cols) > 1 else 0)
        color_axis = c4.selectbox("Cor (Legenda)", [None] + all_cols)
        
        # Botão de Gerar
        if st.button("Gerar Visualização"):
            try:
                if tipo == "Dispersão":
                    fig = px.scatter(df_work, x=x_axis, y=y_axis, color=color_axis)
                elif tipo == "Linha":
                    fig = px.line(df_work, x=x_axis, y=y_axis, color=color_axis)
                elif tipo == "Barra":
                    fig = px.bar(df_work, x=x_axis, y=y_axis, color=color_axis)
                elif tipo == "Histograma":
                    fig = px.histogram(df_work, x=x_axis, color=color_axis)
                elif tipo == "Boxplot":
                    fig = px.box(df_work, x=x_axis, y=y_axis, color=color_axis)
                elif tipo == "Pizza":
                    fig = px.pie(df_work, names=x_axis, values=y_axis)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Não foi possível gerar este gráfico com os dados selecionados. Erro: {e}")

    # --- ABA 4: EXPORTAR ---
    elif menu == "💾 Exportar":
        st.subheader("Download dos Dados Processados")
        st.write("Baixe seu arquivo após realizar as limpezas e filtros.")
        
        # Converte DF para CSV em memória
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_work)

        st.download_button(
            label="📥 Baixar CSV Tratado",
            data=csv,
            file_name="dados_tratados_final.csv",
            mime="text/csv",
        )

else:
    # TELA INICIAL (QUANDO NÃO HÁ ARQUIVO)
    st.markdown("""
    ### 👋 Bem-vindo ao Data Studio Web!
    
    Esta ferramenta segura roda direto no seu navegador.
    
    **Para começar:**
    1. Abra a barra lateral (👈).
    2. Carregue um arquivo `.csv` ou `.xlsx`.
    3. Explore, limpe e visualize seus dados.
    """)