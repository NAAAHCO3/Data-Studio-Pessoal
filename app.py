import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURAÇÃO GERAL ---
st.set_page_config(
    page_title="Data Studio Universal", 
    layout="wide", 
    page_icon="🛠️",
    initial_sidebar_state="expanded"
)

# --- CSS MINIMALISTA (Apenas ajustes estruturais, sem forçar cores) ---
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1); /* Fundo sutil adaptável */
        border-radius: 8px;
        padding: 15px;
    }
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

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="model_trained.pkl">📥 Baixar Modelo Treinado (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar CSV Atualizado",
        data=csv,
        file_name="dados_processados.csv",
        mime="text/csv"
    )

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Data Studio")
    st.caption("v4.0 Universal Workstation")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Carregar Dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Inicialização do Estado
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
            
            if st.button("🔄 Restaurar Dados Originais"):
                st.session_state['df_work'] = st.session_state['df_raw'].copy()
                st.rerun()
            
            st.markdown("---")
            menu = st.radio("Módulos:", 
                ["🏠 Dashboard", "🔬 Explorador", "🛠️ Engenharia de Dados", "🤖 AutoML", "📊 Visualizador"],
                index=2 # Começa na engenharia para facilitar testes
            )
    else:
        menu = "Home"

# --- HOME ---
if not uploaded_file:
    st.title("Data Studio Universal")
    st.markdown("""
    ### A ferramenta definitiva para seus projetos de dados.
    
    Esta versão foca em **flexibilidade total**. Não assumimos nada sobre seus dados; você controla as transformações.
    
    **Novidades da v4.0:**
    * 🎨 **Tema Nativo:** Visual limpo que respeita suas configurações de sistema.
    * 🎛️ **Mapeamento Manual:** Decida exatamente qual valor vira 0, 1, 2...
    * 📝 **Edição Total:** Renomeie colunas e mude tipos de dados.
    * ⚙️ **Hiperparâmetros:** Controle fino sobre os modelos de IA.
    
    👈 **Carregue um arquivo para começar.**
    """)
    st.stop()

# --- MÓDULO 1: DASHBOARD ---
if menu == "🏠 Dashboard":
    st.header("Visão Geral")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Registros", f"{df_work.shape[0]:,}")
    c2.metric("Total Colunas", df_work.shape[1])
    c3.metric("Células Vazias", df_work.isna().sum().sum())
    
    st.markdown("### Pré-visualização e Edição Rápida")
    st.markdown("Você pode editar valores diretamente nesta tabela (experimental).")
    edited_df = st.data_editor(df_work.head(1000), num_rows="dynamic")
    
    # Se quiser salvar edições manuais da grade
    if st.button("Salvar Edições Manuais da Tabela acima"):
        # Nota: data_editor retorna apenas o que é mostrado, cuidado com datasets gigantes
        st.warning("A edição direta é limitada às primeiras 1000 linhas por performance. Use a aba 'Engenharia' para mudanças em massa.")

# --- MÓDULO 2: EXPLORADOR ---
elif menu == "🔬 Explorador":
    st.header("Análise Exploratória")
    
    t1, t2 = st.tabs(["Distribuições", "Correlações"])
    
    with t1:
        col_sel = st.selectbox("Selecione Coluna:", df_work.columns)
        c1, c2 = st.columns([2, 1])
        
        with c1:
            if pd.api.types.is_numeric_dtype(df_work[col_sel]):
                fig = px.histogram(df_work, x=col_sel, marginal="box", title=f"Histograma: {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df_work[col_sel].value_counts().head(20), title=f"Contagem: {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.write("Estatísticas:")
            st.write(df_work[col_sel].describe())

    with t2:
        df_num = df_work.select_dtypes(include='number')
        if len(df_num.columns) > 1:
            fig = px.imshow(df_num.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Colunas numéricas insuficientes.")

# --- MÓDULO 3: ENGENHARIA DE DADOS (O CORAÇÃO DA FERRAMENTA) ---
elif menu == "🛠️ Engenharia de Dados":
    st.header("Engenharia de Atributos (Feature Engineering)")
    st.markdown("Transforme seus dados com precisão cirúrgica.")
    
    tab_clean, tab_transform, tab_map = st.tabs(["Limpeza & Tipos", "Transformação Auto", "Mapeamento Manual (Custom)"])
    
    # ABA 1: LIMPEZA BÁSICA
    with tab_clean:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. Renomear Colunas")
            col_rename = st.selectbox("Coluna para renomear:", df_work.columns)
            new_name = st.text_input("Novo nome:", value=col_rename)
            if st.button("Renomear"):
                df_work.rename(columns={col_rename: new_name}, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()

            st.subheader("2. Excluir Colunas")
            cols_drop = st.multiselect("Selecionar para deletar:", df_work.columns)
            if st.button("Deletar Colunas"):
                df_work.drop(columns=cols_drop, inplace=True)
                st.session_state['df_work'] = df_work
                st.rerun()
        
        with c2:
            st.subheader("3. Converter Tipos")
            col_convert = st.selectbox("Coluna para converter:", df_work.columns)
            type_target = st.selectbox("Para qual tipo?", ["Numérico (Float)", "Texto (String)", "Data (DateTime)"])
            if st.button("Converter Tipo"):
                try:
                    if "Numérico" in type_target:
                        df_work[col_convert] = pd.to_numeric(df_work[col_convert], errors='coerce')
                    elif "Texto" in type_target:
                        df_work[col_convert] = df_work[col_convert].astype(str)
                    elif "Data" in type_target:
                        df_work[col_convert] = pd.to_datetime(df_work[col_convert], errors='coerce')
                    st.session_state['df_work'] = df_work
                    st.success("Convertido!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

            st.subheader("4. Tratar Nulos")
            col_null = st.selectbox("Coluna com Nulos:", df_work.columns[df_work.isna().any()].tolist()) if df_work.isna().any().any() else None
            if col_null:
                method = st.selectbox("Ação:", ["Preencher com 0", "Preencher com Média", "Preencher com Texto Personalizado", "Remover Linhas"])
                custom_val = st.text_input("Valor (se personalizado):") if "Personalizado" in method else None
                
                if st.button("Aplicar Tratamento"):
                    if "Remover" in method:
                        df_work.dropna(subset=[col_null], inplace=True)
                    elif "Média" in method:
                         df_work[col_null].fillna(df_work[col_null].mean(), inplace=True)
                    elif "0" in method:
                         df_work[col_null].fillna(0, inplace=True)
                    elif custom_val:
                         df_work[col_null].fillna(custom_val, inplace=True)
                    st.session_state['df_work'] = df_work
                    st.rerun()

    # ABA 2: TRANSFORMAÇÃO AUTOMÁTICA
    with tab_transform:
        st.subheader("One-Hot Encoding (Automático)")
        st.caption("Cria novas colunas para cada categoria (ex: Cor_Vermelho, Cor_Azul).")
        col_onehot = st.selectbox("Coluna Categórica:", df_work.select_dtypes(include='object').columns)
        if st.button("Gerar Dummies"):
            df_work = pd.get_dummies(df_work, columns=[col_onehot], drop_first=True, dtype=int)
            st.session_state['df_work'] = df_work
            st.success("Colunas geradas!")
            st.rerun()

        st.divider()
        st.subheader("Normalização (StandardScaler)")
        col_norm = st.multiselect("Colunas Numéricas:", df_work.select_dtypes(include='number').columns)
        if st.button("Normalizar Seleção"):
            scaler = StandardScaler()
            df_work[col_norm] = scaler.fit_transform(df_work[col_norm])
            st.session_state['df_work'] = df_work
            st.rerun()

    # ABA 3: MAPEAMENTO MANUAL (NOVIDADE)
    with tab_map:
        st.subheader("Mapeamento de Valores (Label Encoding Customizado)")
        st.caption("Defina manualmente qual número representa cada categoria (ex: Feminino=1, Masculino=0).")
        
        col_map = st.selectbox("Selecione a Coluna para Mapear:", df_work.columns, key="map_col")
        
        if col_map:
            unique_vals = df_work[col_map].unique()
            
            # Limite de segurança para não travar a UI
            if len(unique_vals) > 50:
                st.error(f"Esta coluna tem {len(unique_vals)} valores únicos. Mapeamento manual é recomendado apenas para colunas com poucas categorias.")
            else:
                st.write("Defina os valores de destino:")
                
                # Dicionário para guardar o input do usuário
                mapping_dict = {}
                cols_ui = st.columns(2)
                
                # Loop para criar inputs dinâmicos
                for i, val in enumerate(unique_vals):
                    with cols_ui[i % 2]:
                        # Tenta sugerir um número sequencial (0, 1, 2...)
                        new_val = st.text_input(f"Valor para '{val}':", value=str(i), key=f"map_{col_map}_{i}")
                        # Tenta converter para número se possível
                        try:
                            if '.' in new_val:
                                mapping_dict[val] = float(new_val)
                            else:
                                mapping_dict[val] = int(new_val)
                        except:
                            mapping_dict[val] = new_val # Mantém string se não for número
                
                if st.button("Aplicar Mapeamento Manual"):
                    df_work[col_map] = df_work[col_map].map(mapping_dict)
                    st.session_state['df_work'] = df_work
                    st.success(f"Mapeamento aplicado em {col_map}!")
                    st.rerun()

    # Exibição do DF Atual
    st.divider()
    st.markdown("#### Estado Atual dos Dados")
    st.dataframe(df_work.head())
    download_csv(df_work)

# --- MÓDULO 4: AUTO ML ---
elif menu == "🤖 AutoML":
    st.header("Machine Learning Lab")
    
    c_setup, c_params = st.columns([1, 1])
    
    with c_setup:
        st.subheader("1. Alvo e Features")
        target = st.selectbox("O que vamos prever?", df_work.columns)
        features = st.multiselect("Quais dados usar para prever?", [c for c in df_work.columns if c != target])
        
        # Detecção Automática
        is_reg = pd.api.types.is_numeric_dtype(df_work[target]) and df_work[target].nunique() > 10
        mode = "Regressão" if is_reg else "Classificação"
        st.info(f"Modo Detectado: {mode}")
        
        model_arch = st.selectbox("Algoritmo:", ["Random Forest", "Gradient Boosting", "Linear/Logistic"])

    with c_params:
        st.subheader("2. Hiperparâmetros (Opcional)")
        with st.expander("⚙️ Configuração Avançada"):
            n_estimators = st.slider("Número de Árvores (n_estimators)", 10, 500, 100)
            max_depth = st.slider("Profundidade Máxima (max_depth)", 1, 50, 10)
            test_size = st.slider("Tamanho do Teste (%)", 10, 50, 20) / 100

    if st.button("🚀 Treinar Modelo", type="primary"):
        if not features:
            st.error("Selecione as features!")
        else:
            try:
                with st.spinner("Treinando..."):
                    X = df_work[features]
                    y = df_work[target]
                    
                    # Verificação de segurança
                    if X.select_dtypes(include=['object']).shape[1] > 0:
                        st.error("Ainda existem colunas de texto nas Features! Use a aba 'Engenharia' para mapear ou converter.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # Seleção do Modelo e Params
                    if mode == "Regressão":
                        if model_arch == "Random Forest": 
                            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                        elif model_arch == "Gradient Boosting": 
                            model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
                        else: 
                            model = LinearRegression()
                    else:
                        # Label Encode no Target se for texto
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            y_train, y_test = train_test_split(y, test_size=test_size, random_state=42)
                            
                        if model_arch == "Random Forest": 
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        elif model_arch == "Gradient Boosting": 
                            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        else: 
                            model = LogisticRegression()
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    # Resultados
                    st.success("Treinamento Concluído!")
                    
                    res_col1, res_col2 = st.columns(2)
                    if mode == "Regressão":
                        res_col1.metric("R² Score", f"{r2_score(y_test, preds):.2%}")
                        res_col2.metric("MAE Error", f"{mean_absolute_error(y_test, preds):.2f}")
                        
                        fig = px.scatter(x=y_test, y=preds, labels={'x':'Real', 'y':'Previsto'}, title="Real vs Previsto")
                        fig.add_shape(type="line", line=dict(dash="dash", color="red"), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
                        st.plotly_chart(fig)
                    else:
                        res_col1.metric("Acurácia", f"{accuracy_score(y_test, preds):.2%}")
                        cm = confusion_matrix(y_test, preds)
                        fig = px.imshow(cm, text_auto=True, title="Matriz de Confusão")
                        st.plotly_chart(fig)
                    
                    st.session_state['model'] = model
                    download_model(model)

            except Exception as e:
                st.error(f"Erro no treinamento: {e}")

# --- MÓDULO 5: VISUALIZADOR ---
elif menu == "📊 Visualizador":
    st.header("Gráficos Flexíveis")
    
    col_control, col_graph = st.columns([1, 3])
    
    with col_control:
        type_graph = st.selectbox("Tipo:", ["Scatter", "Bar", "Line", "Histogram", "Box", "Violin"])
        x = st.selectbox("X:", df_work.columns)
        y = st.selectbox("Y:", df_work.columns, index=1)
        color = st.selectbox("Cor:", [None] + list(df_work.columns))
        title = st.text_input("Título do Gráfico:", f"{type_graph} of {y} by {x}")
        
    with col_graph:
        try:
            if type_graph == "Scatter": fig = px.scatter(df_work, x=x, y=y, color=color, title=title)
            elif type_graph == "Bar": fig = px.bar(df_work, x=x, y=y, color=color, title=title)
            elif type_graph == "Line": fig = px.line(df_work, x=x, y=y, color=color, title=title)
            elif type_graph == "Histogram": fig = px.histogram(df_work, x=x, color=color, title=title)
            elif type_graph == "Box": fig = px.box(df_work, x=x, y=y, color=color, title=title)
            elif type_graph == "Violin": fig = px.violin(df_work, x=x, y=y, color=color, title=title)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Selecione colunas compatíveis com o tipo de gráfico.")