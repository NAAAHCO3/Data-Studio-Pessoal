import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

# Configuração da Página
st.set_page_config(
    page_title="DS Master Toolbox v14",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Estilos CSS Personalizados
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #4F8BF9; }
    .tutorial-box { background-color: #e0f2fe; padding: 15px; border-radius: 10px; border-left: 5px solid #0284c7; margin-bottom: 20px; color: #0c4a6e; }
    .stTextArea textarea { font-family: 'Fira Code', monospace; background-color: #1e293b; color: #e2e8f0; }
    .success-box { padding: 10px; background-color: #dcfce7; color: #166534; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1. GESTÃO DE ESTADO (SESSION STATE)
# ---------------------------
def init_session():
    defaults = {
        'df': pd.DataFrame(),
        'df_name': "Nenhum",
        'tutorial_active': False,
        'tutorial_step': 0,
        'code_snippet': "import pandas as pd\nimport matplotlib.pyplot as plt\n\n# 'df' é seu dataframe atual.\n# Tente: st.write(df.describe())",
        'report_items': []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ---------------------------
# 2. MÓDULO ACADEMY (Baseado no Repo ds-cheatsheets)
# ---------------------------
def render_academy():
    st.markdown("<h1 class='main-header'>🎓 Academy: O Grande E-Book de Dados</h1>", unsafe_allow_html=True)
    st.write("Base de conhecimento integrada inspirada no repositório *FavioVazquez/ds-cheatsheets*.")

    # Organização baseada nas pastas do repositório
    tabs = st.tabs(["🐍 Python", "🗄️ SQL", "🤖 Machine Learning", "🧠 Deep Learning", "📊 Data Viz", "🧮 Math & Stats"])

    with tabs[0]: # Python
        st.subheader("Python para Data Science")
        col1, col2 = st.columns([1, 2])
        with col1:
            topic = st.radio("Tópico Python", ["Pandas Basics", "Numpy", "Limpeza de Dados"])
        with col2:
            if topic == "Pandas Basics":
                st.info("🐼 **Pandas** é a excel do Python.")
                st.code("""
import pandas as pd
df = pd.read_csv('dados.csv')
df.head()          # Primeiras linhas
df.info()          # Tipos de dados e nulos
df.describe()      # Estatísticas básicas
df['col'].value_counts() # Contagem de categorias
                """, language='python')
            elif topic == "Numpy":
                st.info("🔢 **Numpy** é a base matemática vetorial.")
                st.code("""
import numpy as np
arr = np.array([1, 2, 3])
np.mean(arr)       # Média
np.std(arr)        # Desvio Padrão
                """, language='python')

    with tabs[1]: # SQL
        st.subheader("Structured Query Language")
        st.warning("Dica: Use o **SQL Studio** no menu para testar esses comandos no seu DataFrame!")
        with st.expander("🔍 SELECT & FILTER (O Básico)"):
            st.markdown("""
            * **Selecionar tudo:** `SELECT * FROM df`
            * **Filtrar:** `SELECT * FROM df WHERE idade > 18`
            * **Texto:** `SELECT * FROM df WHERE nome LIKE 'A%'` (Começa com A)
            """)
        with st.expander("🔗 JOINS (Juntando Tabelas)"):
            st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/SQL_Joins.svg", caption="Visualização de Joins", width=400)
            st.markdown("""
            * **INNER JOIN**: Apenas o que tem nos dois.
            * **LEFT JOIN**: Tudo da esquerda, e o que der match na direita.
            """)

    with tabs[2]: # Machine Learning
        st.subheader("Machine Learning (Scikit-Learn)")
        st.markdown("Conceitos fundamentais baseados nos cheat sheets de ML.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🚦 Supervised Learning")
            st.write("Quando você tem o alvo (target) para treinar.")
            st.markdown("- **Regressão**: Prever número (Preço, Temperatura). *Ex: Linear Regression*")
            st.markdown("- **Classificação**: Prever categoria (Sim/Não, A/B/C). *Ex: Random Forest, Logistic Regression*")
        with c2:
            st.markdown("### 🕵️ Unsupervised Learning")
            st.write("Quando você NÃO tem alvo, quer achar padrões.")
            st.markdown("- **Clustering**: Agrupar similares. *Ex: K-Means*")
            st.markdown("- **Dimensionality Reduction**: Simplificar dados. *Ex: PCA*")

    with tabs[3]: # Deep Learning
        st.subheader("Deep Learning & Redes Neurais")
        st.markdown("Dicas rápidas de Keras e Arquiteturas.")
        st.code("""
# Exemplo Básico Keras (Pseudocódigo)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dense(1, activation='sigmoid')) # Para classificação binária
model.compile(optimizer='adam', loss='binary_crossentropy')
        """, language='python')

# ---------------------------
# 3. MÓDULO TUTORIAL (GAMIFICATION)
# ---------------------------
def tutorial_manager():
    if not st.session_state['tutorial_active']:
        return

    steps = {
        0: "Bem-vindo! Sua primeira missão é **Carregar um Dataset**. Vá para a aba '🏠 Home' e clique em 'Carregar Exemplo Titanic'.",
        1: "Ótimo! Dados carregados. Agora, precisamos entender os dados. Vá para o **🐍 Python Studio**, cole `st.write(df.describe())` e execute.",
        2: "Perfeito! Você viu as estatísticas. Agora vamos visualizar. Vá para o **🐍 Python Studio**, crie um histograma da idade: `fig = px.histogram(df, x='Age'); st.plotly_chart(fig)`.",
        3: "Excelente! Agora vamos treinar um modelo simples. Vá para o **🤖 ML Studio**, escolha 'Survived' como Target e 'Age', 'Fare' como Features. Treine o modelo.",
        4: "Parabéns! Você completou o ciclo básico de Data Science. 🏆"
    }
    
    current = st.session_state['tutorial_step']
    
    st.markdown(f"""
    <div class="tutorial-box">
        <h3>🎓 Modo Tutorial: Nível {current + 1}/5</h3>
        <p>{steps.get(current, "Tutorial Concluído!")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Botão para avançar manualmente se o usuário travar (ou lógica automática)
    if st.sidebar.button(">> Avançar Passo Tutorial"):
        st.session_state['tutorial_step'] += 1
        st.rerun()

# ---------------------------
# 4. PÁGINAS DO APP
# ---------------------------

def page_home():
    st.markdown("<h1 class='main-header'>🏠 Data Hub</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Input de Dados")
        uploaded_file = st.file_uploader("Arraste seu CSV/Excel aqui", type=['csv', 'xlsx'])
        
        if st.button("🚢 Carregar Exemplo Titanic (Tutorial)"):
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            try:
                df = pd.read_csv(url)
                st.session_state['df'] = df
                st.session_state['df_name'] = "Titanic Dataset"
                st.session_state['tutorial_step'] = 1 # Avança tutorial
                st.success("Titanic carregado com sucesso!")
                st.rerun()
            except:
                st.error("Erro ao carregar Titanic. Verifique internet.")

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['df'] = df
                st.session_state['df_name'] = uploaded_file.name
                st.success(f"Arquivo {uploaded_file.name} carregado!")
            except Exception as e:
                st.error(f"Erro: {e}")

    with col2:
        st.subheader("📊 Visão Geral")
        df = st.session_state['df']
        if not df.empty:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Linhas", df.shape[0])
            k2.metric("Colunas", df.shape[1])
            k3.metric("Faltantes", df.isna().sum().sum())
            k4.metric("Duplicatas", df.duplicated().sum())
            
            with st.expander("🔍 Espiar Dados (Head)"):
                st.dataframe(df.head(), use_container_width=True)
            
            with st.expander("ℹ️ Tipos de Dados"):
                st.write(df.dtypes.astype(str))
        else:
            st.info("Aguardando dados... Carregue um arquivo ou use o exemplo.")

def page_python_studio():
    st.markdown("<h1 class='main-header'>🐍 Python Studio (Code-First)</h1>", unsafe_allow_html=True)
    st.caption("Ambiente Sandbox. As variáveis `df` (seu dado), `pd`, `np`, `plt`, `px` já estão importadas.")
    
    if st.session_state['df'].empty:
        st.warning("⚠️ Carregue dados na Home primeiro!")
        return

    col_editor, col_result = st.columns([1, 1])
    
    with col_editor:
        st.markdown("### 📝 Editor")
        code = st.text_area("Seu Script:", value=st.session_state['code_snippet'], height=300)
        
        if st.button("▶️ Executar Código"):
            st.session_state['code_snippet'] = code # Salva estado
            
    with col_result:
        st.markdown("### 🖥️ Output")
        try:
            # Cria ambiente local seguro
            local_env = {
                'pd': pd, 'np': np, 'st': st, 
                'plt': plt, 'sns': sns, 'px': px, 'go': go,
                'df': st.session_state['df'].copy()
            }
            
            # Captura de output (truque para executar exec() e mostrar resultado)
            # No Streamlit, usamos st.write dentro do exec para ver outputs
            exec(code, {}, local_env)
            
            # Verifica tutorial
            if st.session_state['tutorial_active'] and st.session_state['tutorial_step'] in [1, 2]:
                st.success("✅ Ação detectada! Se você completou a tarefa, avance o tutorial na barra lateral.")
                
        except Exception as e:
            st.error(f"Erro de execução: {e}")

def page_ml_studio():
    st.markdown("<h1 class='main-header'>🤖 ML Studio (AutoML Lite)</h1>", unsafe_allow_html=True)
    
    df = st.session_state['df']
    if df.empty: st.warning("Sem dados."); return

    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Configuração")
        target = st.selectbox("🎯 Coluna Alvo (Target)", df.columns)
        
        # Identificar tipo de problema automaticamente
        is_numeric = pd.api.types.is_numeric_dtype(df[target])
        unique_vals = df[target].nunique()
        problem_type = "Regressão" if is_numeric and unique_vals > 10 else "Classificação"
        
        st.info(f"Problema detectado: **{problem_type}**")
        
        features = st.multiselect("⚙️ Variáveis (Features)", [c for c in df.columns if c != target])
        split_size = st.slider("Tamanho Teste (%)", 10, 50, 20)

        if st.button("🚀 Treinar Modelo"):
            if not features:
                st.error("Selecione features!")
            else:
                try:
                    # Preparação simples
                    X = df[features].select_dtypes(include=np.number).fillna(0) # Simplificação para demo
                    y = df[target]
                    
                    # Encoding se for classificação texto
                    if problem_type == "Classificação" and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    elif problem_type == "Regressão":
                        y = y.fillna(y.mean())
                    else:
                        y = y.fillna(y.mode()[0])

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size/100, random_state=42)
                    
                    if problem_type == "Regressão":
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        score = r2_score(y_test, preds)
                        metric_name = "R² Score"
                    else:
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        score = accuracy_score(y_test, preds)
                        metric_name = "Acurácia"
                    
                    st.session_state['last_model_score'] = score
                    st.balloons()
                    
                    # Salvar resultado no estado para exibir
                    st.session_state['ml_results'] = {
                        "tipo": problem_type,
                        "metric": metric_name,
                        "score": score,
                        "features": features
                    }
                    
                    if st.session_state['tutorial_active'] and st.session_state['tutorial_step'] == 3:
                        st.session_state['tutorial_step'] = 4
                        
                except Exception as e:
                    st.error(f"Erro no treino: {e} \n(Dica: O ML Studio Simplificado aceita apenas features numéricas por enquanto. Use o Python Studio para tratamento avançado.)")

    with c2:
        st.subheader("Resultados")
        if 'ml_results' in st.session_state:
            res = st.session_state['ml_results']
            st.metric(label=res['metric'], value=f"{res['score']:.2f}")
            st.write(f"**Features usadas:** {', '.join(res['features'])}")
            
            if res['score'] > 0.95:
                st.warning("⚠️ Score muito alto! Cuidado com Overfitting.")
            elif res['score'] < 0.5:
                st.warning("⚠️ Score baixo. Tente outras features ou limpe os dados.")
            else:
                st.success("✅ Modelo promissor!")

# ---------------------------
# MAIN APP LOGIC
# ---------------------------
def main():
    # Sidebar de Navegação
    with st.sidebar:
        st.title("🚀 Enterprise v14")
        st.write(f"Arquivo Atual: *{st.session_state['df_name']}*")
        
        st.markdown("---")
        st.markdown("### 🧭 Navegação")
        page = st.radio("", ["🏠 Home", "🐍 Python Studio", "🎓 Academy", "🤖 ML Studio"])
        
        st.markdown("---")
        st.markdown("### 🎮 Gamification")
        # Toggle do Tutorial
        tut_mode = st.checkbox("Modo Tutorial", value=st.session_state['tutorial_active'])
        if tut_mode != st.session_state['tutorial_active']:
            st.session_state['tutorial_active'] = tut_mode
            st.rerun()
            
        if st.session_state['tutorial_active']:
            st.progress(st.session_state['tutorial_step'] / 4)
            st.caption(f"Progresso: {st.session_state['tutorial_step']}/4")

    # Renderizador do Tutorial (Aparece no topo de todas as páginas se ativo)
    tutorial_manager()

    # Roteamento de Páginas
    if page == "🏠 Home": page_home()
    elif page == "🐍 Python Studio": page_python_studio()
    elif page == "🎓 Academy": render_academy()
    elif page == "🤖 ML Studio": page_ml_studio()

if __name__ == "__main__":
    main()