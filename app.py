"""
Enterprise Analytics — BI Edition (No-Code) — Refactor v2
- Single-file Streamlit app (modular functions/classes)
- Robust file reading (CSV/Excel, encodings fallback)
- Data Studio: create columns, conditional columns, date extraction, pivot/unpivot, join/merge, split text
- Visual Studio: many chart types + presets + palettes + add-to-dashboard
- Dashboard Builder: assemble charts, remove, export
- Data Quality: KPIs, missing heatmap, outliers detection
- Export: CSV, Excel, PDF (basic)
- Fallback local path for dev: UPLOADED_FILE_PATH
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import joblib
import logging
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# DEV LOCAL PATH (from conversation history)
UPLOADED_FILE_PATH = "/mnt/data/uploaded_dataset.csv"  # replace if you want

st.set_page_config(page_title="Enterprise Analytics — BI Edition", layout="wide", page_icon="📊")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Utilities & Robust IO
# ---------------------------
def try_read_csv(file_obj, encoding_list=("utf-8", "latin1", "cp1252")):
    last_exc = None
    for enc in encoding_list:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception as e:
            last_exc = e
            continue
    raise last_exc

def safe_read(file):
    """Read CSV or Excel robustly. Accepts file-like (uploader) or path."""
    try:
        if hasattr(file, "read"):
            # It's file-like from uploader
            name = getattr(file, "name", "")
            if name.lower().endswith(".csv"):
                try:
                    return try_read_csv(file)
                except Exception:
                    # try pandas autodetect engine
                    file.seek(0)
                    return pd.read_csv(file, engine="python", encoding_errors="ignore")
            else:
                # excel
                file.seek(0)
                return pd.read_excel(file, engine="openpyxl")
        else:
            path = str(file)
            if path.lower().endswith(".csv"):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return pd.read_csv(path, encoding="latin1", engine="python")
            else:
                return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df

def format_number(n):
    try:
        n = float(n)
    except Exception:
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.0f}"

# ---------------------------
# PDF exporter (basic)
# ---------------------------
def generate_pdf_report(df: pd.DataFrame, charts: List[dict], kpis: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Relatorio Executivo — Enterprise BI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    # KPIs
    pdf.set_fill_color(240,240,240)
    pdf.rect(10,40,190,28,'F')
    pdf.set_y(44)
    colw = 190/4
    titles = ['Linhas','Colunas','Nulos','Duplicatas']
    vals = [kpis.get('rows',''), kpis.get('cols',''), kpis.get('nulls',''), kpis.get('dups','')]
    for i,t in enumerate(titles):
        pdf.set_font("Helvetica","B",11)
        pdf.cell(colw,8,t, align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica","",11)
    for i,v in enumerate(vals):
        pdf.cell(colw,8,str(v), align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(12)
    # Short stats: top numeric
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Resumo Estatistico (Top variaveis numericas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    desc = df.describe().T.reset_index().head(8)
    if not desc.empty:
        cols = ['index','mean','min','max']
        if set(cols).issubset(desc.columns):
            desc = desc[cols]
            desc.columns = ['Variavel','Media','Min','Max']
            w=[70,40,40,40]
            pdf.set_font("Helvetica","B",10)
            for i,c in enumerate(desc.columns):
                pdf.cell(w[i],8,c,1,align='C', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
            pdf.set_font("Helvetica","",9)
            for _,row in desc.iterrows():
                for i,val in enumerate(row):
                    txt = str(val)[:28]
                    pdf.cell(w[i],7,txt,1,align='C' if i>0 else 'L', new_x=XPos.RIGHT if i<3 else XPos.LMARGIN, new_y=YPos.TOP if i<3 else YPos.NEXT)
    pdf.ln(6)
    # List charts titles
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Gráficos no Relatório:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica","",10)
    for ch in charts:
        pdf.multi_cell(0,6,f"- {ch.get('title','(sem titulo)')} [{ch.get('type','')}]")
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------
# Data Quality Tools
# ---------------------------
def missing_heatmap(df: pd.DataFrame):
    # Returns a plotly heatmap for missingness
    m = df.isna().astype(int)
    if m.shape[1] > 50:
        # sample columns for readability
        m = m.iloc[:, :50]
    fig = px.imshow(m.T, aspect='auto', color_continuous_scale=['#e0e0e0','#ff6b6b'], labels={'x':'Index', 'y':'Columns'}, title='Mapa de Missing (1 = missing)')
    return fig

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df[(df[col] < low) | (df[col] > high)]

# ---------------------------
# Data Studio (ETL features)
# ---------------------------
def create_column_arithmetic(df: pd.DataFrame, new_col: str, a: str, b: Optional[str], operator: str, b_value: Optional[float]=None):
    df = df.copy()
    try:
        if b is not None:
            if operator == '+':
                df[new_col] = df[a] + df[b]
            elif operator == '-':
                df[new_col] = df[a] - df[b]
            elif operator == '*':
                df[new_col] = df[a] * df[b]
            elif operator == '/':
                df[new_col] = df[a] / df[b].replace(0, np.nan)
        else:
            if operator == '+':
                df[new_col] = df[a] + b_value
            elif operator == '-':
                df[new_col] = df[a] - b_value
            elif operator == '*':
                df[new_col] = df[a] * b_value
            elif operator == '/':
                df[new_col] = df[a] / b_value if b_value != 0 else np.nan
    except Exception as e:
        raise
    return df

def create_column_if(df: pd.DataFrame, new_col: str, col: str, op: str, threshold: float, true_label: str, false_label: str):
    df = df.copy()
    ops = {
        ">": df[col] > threshold,
        "<": df[col] < threshold,
        ">=": df[col] >= threshold,
        "<=": df[col] <= threshold,
        "==": df[col] == threshold,
        "!=": df[col] != threshold
    }
    mask = ops.get(op, df[col] > threshold)
    df[new_col] = np.where(mask, true_label, false_label)
    return df

def extract_date_component(df: pd.DataFrame, date_col: str, component: str, new_name: str):
    df = df.copy()
    if component == "year":
        df[new_name] = df[date_col].dt.year
    elif component == "month":
        df[new_name] = df[date_col].dt.month
    elif component == "day":
        df[new_name] = df[date_col].dt.day
    elif component == "weekday":
        df[new_name] = df[date_col].dt.day_name()
    elif component == "quarter":
        df[new_name] = df[date_col].dt.quarter
    return df

def pivot_transform(df: pd.DataFrame, index: List[str], columns: str, values: str, aggfunc: str = "sum"):
    df = df.copy()
    aggfunc_map = {"sum":"sum", "mean":"mean", "count":"count", "min":"min", "max":"max"}
    if aggfunc not in aggfunc_map: aggfunc = "sum"
    res = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc_map[aggfunc]).reset_index()
    # Flatten columns
    res.columns = [f"{a}" if not isinstance(a, tuple) else "_".join([str(x) for x in a if x is not None]) for a in res.columns]
    return res

def unpivot_transform(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name="variable", value_name="value"):
    df = df.copy()
    res = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return res

def merge_datasets(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: List[str], how="left"):
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)

# ---------------------------
# Visual Studio builder (charts presets & save)
# ---------------------------
def build_chart(chart_type: str, df: pd.DataFrame, x: str=None, y: str=None, color: str=None,
                agg: Optional[str]=None, theme: str='plotly', show_labels: bool=True,
                height: int=500, size: Optional[str]=None):
    plot_df = df.copy()
    if agg and x and y and chart_type in ("Barras","Linha","Pizza"):
        if color:
            plot_df = plot_df.groupby([x, color])[y].agg(agg).reset_index()
        else:
            plot_df = plot_df.groupby(x)[y].agg(agg).reset_index()
    if chart_type == "Barras":
        fig = px.bar(plot_df, x=x, y=y, color=color, text_auto=show_labels, template=theme)
    elif chart_type == "Linha":
        fig = px.line(plot_df, x=x, y=y, color=color, markers=True, template=theme)
    elif chart_type == "Pizza":
        fig = px.pie(plot_df, names=x, values=y, template=theme)
    elif chart_type == "Dispersão":
        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, template=theme)
    elif chart_type == "Histograma":
        fig = px.histogram(plot_df, x=x, color=color, nbins=30, template=theme, text_auto=show_labels)
    elif chart_type == "Box":
        fig = px.box(plot_df, x=x, y=y, color=color, template=theme)
    elif chart_type == "Heatmap":
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True, template=theme, title="Matriz de Correlação")
    else:
        fig = go.Figure()
    fig.update_layout(height=height, title=f"{chart_type} - {y} vs {x}" if y and x else chart_type)
    return fig

# ---------------------------
# App Layout & Tabs
# ---------------------------
# CSS
st.markdown("""
<style>
    .block-container {padding-top:1rem;}
    .metric-box{background:#f3f6fb;border-left:6px solid #4F8BF9;padding:12px;border-radius:6px;margin-bottom:8px}
    .metric-box h3{margin:0;color:#444;font-size:13px}
    .metric-box h2{margin:4px 0 0 0;font-size:20px}
    @media (prefers-color-scheme: dark){
        .metric-box{background:#262730;color:#ddd;border-left:6px solid #ffbd45}
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Data connection
def sidebar_load():
    st.sidebar.header("📂 Conectar Dados")
    uploaded = st.sidebar.file_uploader("Arraste CSV/Excel", type=['csv','xlsx'])
    use_local = st.sidebar.checkbox("Usar caminho local (dev)", value=False)
    if st.sidebar.button("🔄 Resetar Sessão"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
    return uploaded, use_local

# Initialize session state
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
if 'df_main' not in st.session_state:
    st.session_state['df_main'] = pd.DataFrame()
if 'report_charts' not in st.session_state:
    st.session_state['report_charts'] = []  # dicts with title, fig, type, note
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}  # saved chart presets

# App
def main():
    uploaded, use_local = sidebar_load()
    st.title("Enterprise Analytics — BI Edition (No-Code)")
    st.caption("Foco: EDA, Visualização e Dashboard builder sem código")

    # Load data logic
    if uploaded is None and use_local:
        try:
            df = safe_read(UPLOADED_FILE_PATH)
            df = clean_colnames(df)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success("Dados carregados do caminho local.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar caminho local: {e}")
    elif uploaded is not None:
        try:
            df = safe_read(uploaded)
            df = clean_colnames(df)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success(f"Arquivo '{getattr(uploaded,'name', 'uploaded')}' carregado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler arquivo: {e}")

    if st.session_state['df_main'].empty:
        st.markdown("""
        ### Carregue um arquivo CSV ou Excel na barra lateral para começar.
        * Dica: use colunas com nomes legíveis; converta datas no Data Studio.
        """)
        return

    # Top menu
    menu = st.radio("", ["Data Quality", "Data Studio", "Visual Studio", "Relatório/Dashboard", "Exportar"],
                    horizontal=True, index=0)

    df = st.session_state['df_main']

    # ---- Data Quality ----
    if menu == "Data Quality":
        st.header("🔍 Data Quality & EDA")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-box'><h3>Linhas</h3><h2>{format_number(df.shape[0])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><h3>Colunas</h3><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><h3>Células Vazias</h3><h2>{format_number(int(df.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'><h3>Duplicatas</h3><h2>{format_number(int(df.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Mapa de Missing")
        fig = missing_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perfil das Colunas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Numéricas")
            num = df.select_dtypes(include=np.number)
            if not num.empty:
                st.dataframe(num.describe().T[['mean','std','min','max']], use_container_width=True)
                # Outlier detection sample
                col_sel = st.selectbox("Detectar outliers (IQR) em:", num.columns.tolist(), key="outlier_col")
                out_df = detect_outliers_iqr(df, col_sel)
                st.write(f"Outliers detectados (IQR) em {col_sel}: {len(out_df)}")
                if st.checkbox("Mostrar outliers sample", key="outliers_show"):
                    st.dataframe(out_df.head(100), use_container_width=True)
            else:
                st.info("Nenhuma coluna numérica detectada.")

        with col2:
            st.caption("Categóricas")
            cat = df.select_dtypes(include='object')
            if not cat.empty:
                stats_cat = pd.DataFrame({
                    "unique": cat.nunique(),
                    "missing": cat.isna().sum(),
                    "pct_missing": (cat.isna().mean()*100).round(2)
                })
                st.dataframe(stats_cat, use_container_width=True)
            else:
                st.info("Nenhuma coluna de texto.")

        with col3:
            st.caption("Datas")
            date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if date_cols:
                for dc in date_cols:
                    st.write(f"**{dc}**: {df[dc].min()} -> {df[dc].max()}")
            else:
                st.info("Sem colunas de data (converta no Data Studio).")

    # ---- Data Studio (ETL) ----
    elif menu == "Data Studio":
        st.header("🛠 Data Studio (ETL no-code)")
        st.markdown("Crie colunas, converta tipos, pivot/unpivot, join e filtre sem código.")

        tabs = st.tabs(["Criar Colunas", "Renomear/Converter", "Pivot/Unpivot & Merge", "Filtros & Limpeza"])
        # Create columns tab
        with tabs[0]:
            st.subheader("Criar Colunas")
            op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
            if op == "Aritmética":
                left_col = st.selectbox("Coluna A (numérica)", df.select_dtypes(include=np.number).columns.tolist(), key="cs_a")
                mode = st.radio("B é:", ["Outra Coluna","Valor Fixo"])
                if mode == "Outra Coluna":
                    right_col = st.selectbox("Coluna B (numérica)", df.select_dtypes(include=np.number).columns.tolist(), key="cs_b")
                    b_val = None
                else:
                    right_col = None
                    b_val = st.number_input("Valor Fixo", value=1.0)
                op_sym = st.selectbox("Operador", ["+","-","*","/"])
                new_name = st.text_input("Nome da nova coluna", value=f"{left_col}_{op_sym}_col")
                if st.button("Criar coluna aritmética"):
                    try:
                        df2 = create_column_arithmetic(df, new_name, left_col, right_col, op_sym, b_val)
                        st.session_state['df_main'] = df2
                        st.success(f"Coluna '{new_name}' criada.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

            elif op == "Condicional (IF)":
                numcol = st.selectbox("Coluna numérica alvo", df.select_dtypes(include=np.number).columns.tolist())
                operator = st.selectbox("Operador", [">", "<", ">=", "<=", "==", "!="])
                threshold = st.number_input("Valor de Corte", value=0.0)
                true_label = st.text_input("Rótulo se Verdadeiro", "Alto")
                false_label = st.text_input("Rótulo se Falso", "Baixo")
                new_col_name = st.text_input("Nome da nova coluna categórica", "cat_if")
                if st.button("Criar Condicional"):
                    df2 = create_column_if(df, new_col_name, numcol, operator, threshold, true_label, false_label)
                    st.session_state['df_main'] = df2
                    st.success("Coluna condicional criada.")

            elif op == "Extrair Data":
                date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
                if not date_cols:
                    st.warning("Nenhuma coluna de data detectada. Converta uma coluna para datetime primeiro em 'Renomear/Converter'.")
                else:
                    dc = st.selectbox("Coluna de data", date_cols)
                    comp = st.selectbox("Componente", ["year","month","day","weekday","quarter"])
                    new_name = st.text_input("Nome da nova coluna", value=f"{dc}_{comp}")
                    if st.button("Extrair componente"):
                        df2 = extract_date_component(df, dc, comp, new_name)
                        st.session_state['df_main'] = df2
                        st.success(f"Coluna {new_name} criada.")

            elif op == "Split Texto":
                txt_cols = df.select_dtypes(include='object').columns.tolist()
                if not txt_cols:
                    st.info("Nenhuma coluna de texto disponível.")
                else:
                    txt_col = st.selectbox("Coluna Texto", txt_cols)
                    sep = st.text_input("Separador (regex OK)", value="\\s+")
                    part = st.number_input("Parte (0 = primeira, 1 = segunda...)", min_value=0, step=1, value=0)
                    new_name = st.text_input("Nome nova coluna", value=f"{txt_col}_part{part}")
                    if st.button("Criar Split"):
                        try:
                            parts = df[txt_col].astype(str).str.split(sep, regex=True, expand=True)
                            if part < parts.shape[1]:
                                df[new_name] = parts[part]
                                st.session_state['df_main'] = df
                                st.success("Split criado.")
                            else:
                                st.error("Parte fora de alcance.")
                        except Exception as e:
                            st.error(f"Erro split: {e}")

        # Rename/convert
        with tabs[1]:
            st.subheader("Renomear / Converter Tipos")
            col = st.selectbox("Selecionar coluna", df.columns.tolist(), key="rename_col")
            new_name = st.text_input("Novo nome (deixe vazio para ignorar)")
            if st.button("Aplicar Renome"):
                if new_name:
                    df.rename(columns={col:new_name}, inplace=True)
                    st.session_state['df_main'] = df
                    st.success("Renomeado.")
            st.markdown("---")
            st.subheader("Converter tipo")
            colc = st.selectbox("Coluna", df.columns.tolist(), key="convert_col")
            dtype = st.selectbox("Converter para", ["Data (datetime)", "Número (float)", "Texto (string)"])
            if st.button("Converter Tipo"):
                try:
                    if "Data" in dtype:
                        df[colc] = pd.to_datetime(df[colc], errors='coerce')
                    elif "Número" in dtype:
                        df[colc] = pd.to_numeric(df[colc], errors='coerce')
                    else:
                        df[colc] = df[colc].astype(str)
                    st.session_state['df_main'] = df
                    st.success("Conversão aplicada.")
                except Exception as e:
                    st.error(f"Erro conversão: {e}")

        # Pivot/Unpivot & Merge
        with tabs[2]:
            st.subheader("Pivot / Unpivot")
            pv = st.radio("Operação", ["Pivot (wide)", "Unpivot (melt)"])
            if pv == "Pivot (wide)":
                idx = st.multiselect("Index (linhas)", df.columns.tolist(), default=df.columns.tolist()[:1])
                col = st.selectbox("Coluna que vira colunas", df.columns.tolist())
                val = st.selectbox("Valores", df.select_dtypes(include=np.number).columns.tolist())
                agg = st.selectbox("Agregação", ["sum","mean","count","min","max"])
                if st.button("Executar Pivot"):
                    try:
                        res = pivot_transform(df, idx, col, val, agg)
                        st.session_state['df_main'] = res
                        st.success("Pivot executado, substituindo df principal.")
                    except Exception as e:
                        st.error(f"Erro pivot: {e}")
            else:
                idv = st.multiselect("Id_vars", df.columns.tolist(), default=df.columns.tolist()[:1])
                valv = st.multiselect("Value_vars (colunas a 'derreter')", df.columns.tolist(), default=df.columns.tolist()[1:4])
                if st.button("Executar Unpivot"):
                    try:
                        res = unpivot_transform(df, idv, valv)
                        st.session_state['df_main'] = res
                        st.success("Unpivot executado.")
                    except Exception as e:
                        st.error(f"Erro unpivot: {e}")

            st.markdown("---")
            st.subheader("Merge / Join com outro arquivo")
            join_file = st.file_uploader("Carregar segundo arquivo para merge (opcional)", key="merge_file")
            if join_file:
                try:
                    right = safe_read(join_file)
                    right = clean_colnames(right)
                    st.write("Preview do arquivo secundário:")
                    st.dataframe(right.head(3))
                    left_on = st.multiselect("Chave(s) esquerda (no dataset atual)", df.columns.tolist())
                    right_on = st.multiselect("Chave(s) direita (no arquivo secundário)", right.columns.tolist())
                    how = st.selectbox("Tipo de merge", ["left","inner","right","outer"])
                    if st.button("Executar Merge"):
                        if left_on and right_on and len(left_on) == len(right_on):
                            res = merge_datasets(df, right, left_on, right_on, how=how)
                            st.session_state['df_main'] = res
                            st.success("Merge realizado.")
                        else:
                            st.error("Defina chaves correspondentes (mesmo número).")
                except Exception as e:
                    st.error(f"Erro lendo arquivo secundário: {e}")

        # Filters & cleaning
        with tabs[3]:
            st.subheader("Filtros Ativos")
            colf = st.selectbox("Coluna para filtrar", df.columns.tolist())
            if pd.api.types.is_numeric_dtype(df[colf]):
                mn, mx = float(df[colf].min()), float(df[colf].max())
                rng = st.slider("Intervalo", mn, mx, (mn, mx))
                if st.button("Aplicar filtro numérico"):
                    df2 = df[(df[colf] >= rng[0]) & (df[colf] <= rng[1])]
                    st.session_state['df_main'] = df2
                    st.success("Filtro aplicado.")
            elif np.issubdtype(df[colf].dtype, np.datetime64):
                dmin, dmax = df[colf].min().date(), df[colf].max().date()
                dr = st.date_input("Período", [dmin, dmax])
                if st.button("Aplicar filtro de data"):
                    df2 = df[(df[colf].dt.date >= dr[0]) & (df[colf].dt.date <= dr[1])]
                    st.session_state['df_main'] = df2
                    st.success("Filtro de data aplicado.")
            else:
                vals = df[colf].dropna().unique().tolist()
                if len(vals) < 200:
                    sel = st.multiselect("Valores", vals, default=vals[:10])
                    if st.button("Aplicar filtro categórico"):
                        df2 = df[df[colf].isin(sel)]
                        st.session_state['df_main'] = df2
                        st.success("Filtro aplicado.")
                else:
                    txt = st.text_input("Procurar (filtrar por substring)")
                    if st.button("Filtrar texto"):
                        df2 = df[df[colf].astype(str).str.contains(txt, case=False, na=False)]
                        st.session_state['df_main'] = df2
                        st.success("Filtro de texto aplicado.")
            st.markdown("---")
            if st.button("Remover linhas vazias (dropna)"):
                before = len(df)
                df2 = df.dropna()
                st.session_state['df_main'] = df2
                st.success(f"Removidas {before - len(df2)} linhas vazias.")

    # ---- Visual Studio ----
    elif menu == "Visual Studio":
        st.header("🎨 Visual Studio (Criador de gráficos)")
        df = st.session_state['df_main']
        left, right = st.columns([1,2])
        with left:
            chart_type = st.selectbox("Tipo de gráfico", ["Barras","Linha","Dispersão","Pizza","Histograma","Box","Heatmap"])
            x = st.selectbox("Eixo X", df.columns.tolist(), index=0)
            y = None
            if chart_type not in ("Pizza","Histograma","Heatmap"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    y = st.selectbox("Eixo Y", numeric_cols, index=0)
                else:
                    st.warning("Nenhuma coluna numérica disponível para eixo Y.")
            color = st.selectbox("Cor / Legenda (opcional)", ["Nenhum"] + df.columns.tolist())
            if color == "Nenhum": color = None
            agg = None
            if chart_type in ("Barras","Linha","Pizza"):
                agg = st.selectbox("Aggregação (groupby)", ["sum","mean","count","min","max"], index=0)
            size = None
            if chart_type == "Dispersão":
                sizes = df.select_dtypes(include=np.number).columns.tolist()
                sizes = ["Nenhum"] + sizes
                size = st.selectbox("Tamanho (opcional)", sizes)
                if size == "Nenhum": size = None
            theme = st.selectbox("Tema", ["plotly_dark","plotly","simple_white","ggplot2"], index=0)
            show_labels = st.checkbox("Mostrar rótulos", value=True)
            title = st.text_input("Título do gráfico", value=f"{chart_type}: {y} por {x}" if y else f"{chart_type}: {x}")
            height = st.slider("Altura", 300, 900, 500)
            if st.button("Gerar gráfico"):
                try:
                    fig = build_chart(chart_type if chart_type!="Dispersão" else "Dispersão", df, x=x, y=y, color=color, agg=agg, theme=theme, show_labels=show_labels, height=height, size=size)
                    st.session_state['last_chart'] = {"fig":fig, "title": title, "type": chart_type}
                    st.session_state['last_fig'] = fig
                    st.success("Gráfico criado (veja à direita).")
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")

            # presets
            st.markdown("---")
            st.subheader("Presets")
            p_name = st.text_input("Nome do preset")
            if st.button("Salvar preset") and 'last_chart' in st.session_state and p_name:
                st.session_state['presets'][p_name] = {
                    "chart_type": chart_type, "x": x, "y": y, "color": color, "agg": agg, "theme": theme, "show_labels": show_labels, "height": height
                }
                st.success("Preset salvo.")
            if st.session_state['presets']:
                preset_choice = st.selectbox("Aplicar preset salvo", ["Nenhum"] + list(st.session_state['presets'].keys()))
                if preset_choice and preset_choice != "Nenhum":
                    p = st.session_state['presets'][preset_choice]
                    st.write("Preset:", p)

        with right:
            st.subheader("Preview")
            if 'last_fig' in st.session_state:
                st.plotly_chart(st.session_state['last_fig'], use_container_width=True)
                st.markdown("---")
                note = st.text_area("Nota/Comentário do gráfico", key="chart_note")
                if st.button("Adicionar ao Relatório"):
                    entry = {
                        "title": st.session_state.get('last_chart', {}).get('title','Sem titulo'),
                        "fig": st.session_state['last_fig'],
                        "type": st.session_state.get('last_chart', {}).get('type',''),
                        "note": note
                    }
                    st.session_state['report_charts'].append(entry)
                    st.success("Adicionado ao relatório.")
            else:
                st.info("Gere um gráfico com as opções à esquerda para pré-visualizar.")

    # ---- Dashboard / Report ----
    elif menu == "Relatório/Dashboard":
        st.header("📑 Relatório / Dashboard Builder")
        charts = st.session_state['report_charts']
        top1, top2 = st.columns([4,1])
        top1.write(f"Gráficos no relatório: {len(charts)}")
        if top2.button("Limpar Relatório"):
            st.session_state['report_charts'] = []
            st.experimental_rerun()
        st.markdown("---")
        if not charts:
            st.info("Nenhum gráfico no relatório. Crie no Visual Studio e clique em 'Adicionar ao Relatório'.")
        for i, ch in enumerate(charts):
            st.markdown("---")
            c1, c2 = st.columns([3,1])
            with c1:
                st.plotly_chart(ch['fig'], use_container_width=True)
            with c2:
                st.write(f"### {ch.get('title','Sem titulo')}")
                st.write(f"Tipo: {ch.get('type','')}")
                if ch.get('note'):
                    st.info(ch.get('note'))
                if st.button(f"Remover gráfico {i+1}", key=f"rm_{i}"):
                    st.session_state['report_charts'].pop(i)
                    st.experimental_rerun()

    # ---- Export ----
    elif menu == "Exportar":
        st.header("📤 Exportar dados e relatório")
        df = st.session_state['df_main']
        st.markdown("### Exportar Dados")
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV", csv, "dados_tratados.csv", "text/csv")
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Dados")
            st.download_button("Baixar Excel", buffer.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("---")
        st.subheader("Exportar Relatório (PDF)")
        kpis = {"rows": len(df), "cols": df.shape[1], "nulls": int(df.isna().sum().sum()), "dups": int(df.duplicated().sum())}
        if st.button("Gerar PDF Executivo"):
            pdf_bytes = generate_pdf_report(df, st.session_state['report_charts'], kpis)
            st.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
        st.markdown("---")
        st.info("Dica: salve presets e reutilize para acelerar construção de múltiplos relatórios.")

    # persist master df
    st.session_state['df_main'] = st.session_state.get('df_main', df)

if __name__ == "__main__":
    main()
