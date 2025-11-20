"""
Enterprise Analytics — BI Studio (Laudo Técnico, Dark theme, DuckDB)
Single-file Streamlit app. Full rebuild: robust IO, ETL, visual builder, dashboard, exports.

Requirements (suggested):
streamlit==1.51.0
pandas==2.3.3
numpy==2.3.4
plotly==6.4.0
fpdf2==2.8.5
duckdb
kaleido (optional, for exporting plot images into PDF)
openpyxl
joblib
pyarrow
scikit-learn
statsmodels
scipy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import joblib
import logging
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Optional libs
try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

# optional for saving figures to PNG for PDF embedding
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# DEV local path from conversation history (provided by developer instruction)
DEFAULT_LOCAL_PATH = "/mnt/data/uploaded_dataset.csv"

st.set_page_config(page_title="Enterprise Analytics — BI Studio", layout="wide", page_icon="📊", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Utilities & robust IO
# ---------------------------
def try_read_csv(file_obj, encodings=("utf-8", "latin1", "cp1252")) -> pd.DataFrame:
    last_exc = None
    for enc in encodings:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception as e:
            last_exc = e
            continue
    # last fallback: use python engine with errors ignored
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, engine="python", encoding_errors="ignore")
    except Exception as e:
        raise last_exc or e

def safe_read(path_or_buffer) -> pd.DataFrame:
    """Read either uploaded file (file-like) or path. Robust to encodings."""
    if hasattr(path_or_buffer, "read"):
        name = getattr(path_or_buffer, "name", "")
        if name.lower().endswith(".csv"):
            return try_read_csv(path_or_buffer)
        else:
            # excel read
            path_or_buffer.seek(0)
            return pd.read_excel(path_or_buffer, engine="openpyxl")
    else:
        p = str(path_or_buffer)
        if p.lower().endswith(".csv"):
            try:
                return pd.read_csv(p)
            except Exception:
                return pd.read_csv(p, encoding="latin1", engine="python")
        else:
            return pd.read_excel(p, engine="openpyxl")

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]", "", regex=True)
    )
    return df

def safe_display_df(df: pd.DataFrame, height: int = 300):
    """Convert object-like columns to string to avoid pyarrow conversion errors, then display."""
    dd = df.copy()
    # convert problematic columns to strings only for display (not mutate session data)
    for c in dd.columns:
        if dd[c].dtype == object:
            # convert but keep NaN as empty string for display
            dd[c] = dd[c].astype(str).where(~dd[c].isna(), "")
    st.dataframe(dd, width="stretch", use_container_width=False)  # width arg used as replacement

# ---------------------------
# PDF: Laudo Técnico layout
# ---------------------------
def generate_pdf_laudo(df: pd.DataFrame, charts: List[dict], kpis: Dict[str, any], notes: str = "") -> bytes:
    """
    Create a 'Laudo Técnico' style PDF.
    If kaleido is available, embed PNG thumbnails of charts.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Laudo Técnico — Relatório Executivo", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # KPIs block
    pdf.set_fill_color(245,245,245)
    pdf.rect(10, 30, 190, 28, style='F')
    pdf.set_y(34)
    colw = 190 / 4
    labels = ["Linhas", "Colunas", "Células Nulas", "Duplicatas"]
    vals = [kpis.get("rows", ""), kpis.get("cols", ""), kpis.get("nulls", ""), kpis.get("dups", "")]
    pdf.set_font("Helvetica", "B", 11)
    for i, lab in enumerate(labels):
        pdf.cell(colw, 7, lab, align="C", new_x=XPos.RIGHT if i < 3 else XPos.LMARGIN, new_y=YPos.TOP if i < 3 else YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    for i, v in enumerate(vals):
        pdf.cell(colw, 8, str(v), align="C", new_x=XPos.RIGHT if i < 3 else XPos.LMARGIN, new_y=YPos.TOP if i < 3 else YPos.NEXT)
    pdf.ln(10)

    # Observações / notas resumidas
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Observações Técnicas", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    if notes:
        pdf.multi_cell(0, 6, notes)
    else:
        pdf.multi_cell(0, 6, "Relatório gerado automaticamente. Verifique transformações e filtros aplicados no Data Studio.")
    pdf.ln(6)

    # Short statistics table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Resumo Estatístico (Principais variáveis numéricas)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    desc = df.describe().T.reset_index().head(8)
    if not desc.empty:
        cols = ["index", "mean", "std", "min", "max"]
        available = [c for c in cols if c in desc.columns]
        # Render table flexibly
        widths = [60] + [int((190 - 60) / max(1, len(available)-1))] * (len(available)-1)
        pdf.set_font("Helvetica", "B", 10)
        for i, c in enumerate(available):
            name = c.capitalize() if c != "index" else "Variável"
            pdf.cell(widths[i], 8, name, 1, align="C", new_x=XPos.RIGHT if i < len(available)-1 else XPos.LMARGIN, new_y=YPos.TOP if i < len(available)-1 else YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        for _, row in desc.iterrows():
            for i, c in enumerate(available):
                val = row[c] if c in row else ""
                s = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                pdf.cell(widths[i], 7, s[:30], 1, align="C" if i>0 else "L", new_x=XPos.RIGHT if i < len(available)-1 else XPos.LMARGIN, new_y=YPos.TOP if i < len(available)-1 else YPos.NEXT)
    pdf.ln(8)

    # Charts list + optional thumbnails
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Gráficos incluídos no relatório", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    for ch in charts:
        title = ch.get("title", "(sem título)")
        typ = ch.get("type", "")
        pdf.multi_cell(0, 6, f"- {title} [{typ}]")
        # try to embed if kaleido present and fig saved in ch
        if HAS_KALEIDO and "fig" in ch:
            fig = ch["fig"]
            try:
                img_bytes = fig.to_image(format="png", engine="kaleido", scale=1)
                # write to a temp buffer and embed
                img_stream = io.BytesIO(img_bytes)
                # temp filename: we must save to disk then image path. FPDF accepts bytes via image method only with path,
                # but fpdf2 supports image from file-like using .image. We'll dump to a temporary file.
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(img_bytes)
                tmp.flush()
                tmp.close()
                pdf.image(tmp.name, w=160)
                os.unlink(tmp.name)
                pdf.ln(4)
            except Exception:
                # ignore thumbnail errors
                pass

    # Footer signature
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Relatório gerado por Enterprise BI Studio. Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    return pdf.output(dest="S").encode("latin-1", "replace")


# ---------------------------
# Data Quality helpers
# ---------------------------
def missing_heatmap(df: pd.DataFrame):
    m = df.isna().astype(int)
    # limit columns for readability
    if m.shape[1] > 80:
        m = m.iloc[:, :80]
    fig = px.imshow(m.T, aspect="auto", color_continuous_scale=["#ffffff", "#ff6b6b"], labels={"x":"Index","y":"Columns"}, title="Mapa de Missing (1 = missing)")
    return fig

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].dropna()
    if s.empty: return pd.DataFrame()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df[(df[col] < low) | (df[col] > high)]

# ---------------------------
# Data Studio transformations
# ---------------------------
def create_column_arithmetic(df: pd.DataFrame, new_col: str, a: str, b: Optional[str], operator: str, b_value: Optional[float] = None):
    df = df.copy()
    if b is not None:
        if operator == "+": df[new_col] = df[a] + df[b]
        elif operator == "-": df[new_col] = df[a] - df[b]
        elif operator == "*": df[new_col] = df[a] * df[b]
        elif operator == "/": df[new_col] = df[a] / df[b].replace(0, np.nan)
    else:
        if operator == "+": df[new_col] = df[a] + b_value
        elif operator == "-": df[new_col] = df[a] - b_value
        elif operator == "*": df[new_col] = df[a] * b_value
        elif operator == "/": df[new_col] = df[a] / (b_value if b_value != 0 else np.nan)
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
    if component == "year": df[new_name] = df[date_col].dt.year
    elif component == "month": df[new_name] = df[date_col].dt.month
    elif component == "day": df[new_name] = df[date_col].dt.day
    elif component == "weekday": df[new_name] = df[date_col].dt.day_name()
    elif component == "quarter": df[new_name] = df[date_col].dt.quarter
    return df

def pivot_transform(df: pd.DataFrame, index: List[str], columns: str, values: str, aggfunc: str = "sum"):
    agg_map = {"sum":"sum", "mean":"mean", "count":"count", "min":"min", "max":"max"}
    if aggfunc not in agg_map: aggfunc = "sum"
    res = df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg_map[aggfunc]).reset_index()
    # flatten multiindex columns
    res.columns = [("_".join(map(str, c)) if isinstance(c, tuple) else str(c)).strip("_") for c in res.columns]
    return res

def unpivot_transform(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name="variable", value_name="value"):
    return df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

def merge_datasets(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: List[str], how="left"):
    if len(left_on) != len(right_on):
        raise ValueError("Número de chaves diferentes")
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)

# ---------------------------
# Visual builder
# ---------------------------
def build_chart(chart_type: str, df: pd.DataFrame, x: Optional[str]=None, y: Optional[str]=None, color: Optional[str]=None,
                agg: Optional[str]=None, theme: str="plotly_dark", show_labels: bool=True, height: int=480, size: Optional[str]=None):
    plot_df = df.copy()
    if agg and x and y and chart_type in ("Barras", "Linha", "Pizza"):
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
    fig.update_layout(height=height, title=f"{chart_type} — {y} por {x}" if x and y else chart_type, template=theme)
    return fig

# ---------------------------
# DataEngine for SQL (DuckDB preferred)
# ---------------------------
class DataEngine:
    def __init__(self, use_duckdb=True):
        self.use_duckdb = use_duckdb and HAS_DUCKDB

    def run_query(self, df: pd.DataFrame, sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            if self.use_duckdb:
                con = duckdb.connect(database=':memory:')
                # register dataframe as table 'dados'
                con.register('dados', df)
                res = con.execute(sql).df()
                con.close()
                return res, None
            else:
                # fallback to sqlite
                import sqlite3
                conn = sqlite3.connect(':memory:')
                df.to_sql('dados', conn, index=False, if_exists='replace')
                res = pd.read_sql_query(sql, conn)
                conn.close()
                return res, None
        except Exception as e:
            return pd.DataFrame(), str(e)

# ---------------------------
# UI: CSS / Theme (Dark)
# ---------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .metric { background:#22232a; color:#fff; padding:12px; border-radius:8px; margin-bottom:8px }
      .metric h3 { margin:0; color:#aaa; font-size:13px }
      .metric h2 { margin:4px 0 0; font-size:22px; color:#fff }
      .stButton>button { border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Session state init
# ---------------------------
if "df_raw" not in st.session_state: st.session_state["df_raw"] = pd.DataFrame()
if "df_main" not in st.session_state: st.session_state["df_main"] = pd.DataFrame()
if "report_charts" not in st.session_state: st.session_state["report_charts"] = []  # list of dicts {fig, title, type, note}
if "last_fig" not in st.session_state: st.session_state["last_fig"] = None
if "last_meta" not in st.session_state: st.session_state["last_meta"] = {}

# ---------------------------
# Sidebar: Load & Controls
# ---------------------------
def sidebar_controls():
    st.sidebar.header("Conectar Dados")
    uploaded = st.sidebar.file_uploader("Arraste CSV/Excel", type=["csv", "xlsx"])
    use_local = st.sidebar.checkbox("Usar caminho local (dev)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Ferramentas:")
    st.sidebar.caption("Data Studio = ETL sem código | Visual Studio = criador de gráficos | Relatório = montar dashboard")
    st.sidebar.markdown("---")
    duckdb_choice = st.sidebar.checkbox("Usar DuckDB (SQL)", value=HAS_DUCKDB)
    # reset
    if st.sidebar.button("🔄 Resetar Sessão"):
        keys = list(st.session_state.keys())
        for k in keys:
            del st.session_state[k]
        st.experimental_rerun()
    return uploaded, use_local, duckdb_choice

uploaded_file, use_local, duckdb_choice = sidebar_controls()
de = DataEngine(use_duckdb=duckdb_choice and HAS_DUCKDB)

# Load data
if uploaded_file is not None:
    try:
        df = safe_read(uploaded_file)
        df = clean_colnames(df)
        st.session_state["df_raw"] = df.copy()
        st.session_state["df_main"] = df.copy()
        st.success(f"Arquivo '{getattr(uploaded_file,'name','uploaded')}' carregado.")
    except Exception as e:
        st.sidebar.error(f"Erro leitura: {e}")
elif use_local:
    try:
        df = safe_read(DEFAULT_LOCAL_PATH)
        df = clean_colnames(df)
        st.session_state["df_raw"] = df.copy()
        st.session_state["df_main"] = df.copy()
        st.sidebar.success(f"Arquivo local '{DEFAULT_LOCAL_PATH}' carregado.")
    except Exception as e:
        st.sidebar.error(f"Erro no arquivo local: {e}")

# Main app
st.title("Enterprise Analytics — BI Studio (Laudo Técnico)")
st.caption("Tema: Escuro — SQL: DuckDB (se disponível) — PDF: Laudo Técnico")

if st.session_state["df_main"].empty:
    st.markdown("### Carregue um arquivo CSV/Excel na barra lateral para começar.")
    st.info("Sugestão: arquivos com textos longos são suportados; se houver erro de leitura, tente marcar 'usar caminho local' e ajustar encodings.")
    st.stop()

df_main: pd.DataFrame = st.session_state["df_main"]

# Top menu
menu = st.radio("", ["Data Quality", "Data Studio", "Visual Studio", "SQL Lab", "Relatório/Dashboard", "Exportar"], horizontal=True)

# ---------------------------
# Data Quality
# ---------------------------
if menu == "Data Quality":
    st.header("🔍 Data Quality & EDA")
    r, c = df_main.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric'><h3>Registros</h3><h2>{r:,}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><h3>Colunas</h3><h2>{c:,}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><h3>Células Nulas</h3><h2>{int(df_main.isna().sum().sum()):,}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'><h3>Duplicatas</h3><h2>{int(df_main.duplicated().sum()):,}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Mapa de Missing")
    fig_missing = missing_heatmap(df_main)
    st.plotly_chart(fig_missing, width="stretch")

    st.subheader("Perfil das Colunas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔢 Numéricas")
        num = df_main.select_dtypes(include=np.number)
        if not num.empty:
            st.dataframe(num.describe().T[["mean", "std", "min", "max"]], width="stretch")
            col_sel = st.selectbox("Detectar outliers (IQR) em:", num.columns.tolist(), key="out_col")
            out_df = detect_outliers_iqr(df_main, col_sel)
            st.write(f"Outliers detectados: {len(out_df)}")
            if st.checkbox("Mostrar sample dos outliers", key="show_out_sample"):
                safe_display_df(out_df.head(200))
        else:
            st.info("Sem colunas numéricas.")

    with col2:
        st.caption("🔤 Categóricas")
        cat = df_main.select_dtypes(include="object")
        if not cat.empty:
            stats_cat = pd.DataFrame({"unique": cat.nunique(), "missing": cat.isna().sum(), "% missing": (cat.isna().mean()*100).round(2)})
            st.dataframe(stats_cat, width="stretch")
        else:
            st.info("Sem colunas categóricas.")

    with col3:
        st.caption("📅 Datas (detectadas)")
        date_cols = [c for c in df_main.columns if np.issubdtype(df_main[c].dtype, np.datetime64)]
        if date_cols:
            for dc in date_cols:
                st.write(f"**{dc}**: {df_main[dc].min()} → {df_main[dc].max()}")
        else:
            st.info("Nenhuma coluna de data detectada. Use Data Studio → Renomear/Converter para DateTime.")

    st.markdown("---")
    st.subheader("Amostra dos Dados")
    safe_display_df(df_main.sample(min(200, len(df_main))))

# ---------------------------
# Data Studio (ETL No-Code)
# ---------------------------
elif menu == "Data Studio":
    st.header("🛠 Data Studio (ETL No-Code)")
    tabs = st.tabs(["Criar Colunas", "Renomear/Converter", "Pivot/Merge", "Filtros/Transformações"])

    # CREATE
    with tabs[0]:
        st.markdown("### Criar Colunas")
        op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
        if op == "Aritmética":
            cols_num = df_main.select_dtypes(include=np.number).columns.tolist()
            if not cols_num:
                st.info("Sem colunas numéricas para operação.")
            else:
                a = st.selectbox("Coluna A", cols_num, key="aop")
                mode = st.radio("Coluna B é:", ["Outra Coluna", "Valor Fixo"])
                if mode == "Outra Coluna":
                    b = st.selectbox("Coluna B", cols_num, key="bop")
                    bval = None
                else:
                    b = None
                    bval = st.number_input("Valor Fixo", value=1.0)
                op_sign = st.selectbox("Operador", ["+", "-", "*", "/"])
                new_name = st.text_input("Nome nova coluna", value=f"{a}_{op_sign}_res")
                if st.button("Criar coluna"):
                    try:
                        st.session_state["df_main"] = create_column_arithmetic(df_main, new_name, a, b, op_sign, bval)
                        st.success(f"Coluna '{new_name}' criada.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        elif op == "Condicional (IF)":
            cols_num = df_main.select_dtypes(include=np.number).columns.tolist()
            if not cols_num:
                st.info("Sem colunas numéricas.")
            else:
                col = st.selectbox("Coluna Alvo", cols_num)
                comparator = st.selectbox("Operador", [">", "<", ">=", "<=", "==", "!="])
                threshold = st.number_input("Threshold", value=0.0)
                true_label = st.text_input("Rótulo (True)", "Alto")
                false_label = st.text_input("Rótulo (False)", "Baixo")
                new_name = st.text_input("Nome nova coluna", value=f"{col}_cat")
                if st.button("Criar condicional"):
                    st.session_state["df_main"] = create_column_if(df_main, new_name, col, comparator, threshold, true_label, false_label)
                    st.success("Coluna condicional criada.")

        elif op == "Extrair Data":
            date_cols = [c for c in df_main.columns if np.issubdtype(df_main[c].dtype, np.datetime64)]
            if not date_cols:
                st.info("Nenhuma coluna de data detectada. Converta uma coluna para DateTime primeiro.")
            else:
                dc = st.selectbox("Coluna de Data", date_cols)
                comp = st.selectbox("Componente", ["year", "month", "day", "weekday", "quarter"])
                new_name = st.text_input("Nome nova coluna", value=f"{dc}_{comp}")
                if st.button("Extrair componente"):
                    st.session_state["df_main"] = extract_date_component(df_main, dc, comp, new_name)
                    st.success("Componente extraído.")

        elif op == "Split Texto":
            text_cols = df_main.select_dtypes(include="object").columns.tolist()
            if not text_cols:
                st.info("Sem colunas de texto")
            else:
                tc = st.selectbox("Coluna texto", text_cols)
                sep = st.text_input("Separador", " ")
                idx = st.number_input("Índice da parte (0 = primeira)", value=0, step=1)
                new_name = st.text_input("Nome nova coluna", value=f"{tc}_part_{idx}")
                if st.button("Split"):
                    try:
                        parts = df_main[tc].astype(str).str.split(sep, expand=True)
                        if idx < parts.shape[1]:
                            df_main[new_name] = parts[idx]
                            st.session_state["df_main"] = df_main
                            st.success("Split aplicado.")
                        else:
                            st.error("Índice fora do range das partes.")
                    except Exception as e:
                        st.error(e)

    # RENAME / CONVERT
    with tabs[1]:
        st.markdown("### Renomear e Converter tipos")
        col = st.selectbox("Coluna", df_main.columns, key="rename_col")
        new_name = st.text_input("Novo nome")
        if st.button("Renomear coluna"):
            if new_name:
                df_main.rename(columns={col: new_name}, inplace=True)
                st.session_state["df_main"] = df_main
                st.success("Renomeado.")
            else:
                st.error("Informe um novo nome.")

        st.markdown("---")
        st.markdown("Converter tipo")
        col_conv = st.selectbox("Coluna", df_main.columns, key="conv_col")
        dtype = st.selectbox("Para", ["Data (DateTime)", "Número (float)", "Texto (string)"])
        if st.button("Converter tipo"):
            try:
                if "Data" in dtype:
                    df_main[col_conv] = pd.to_datetime(df_main[col_conv], errors="coerce")
                elif "Número" in dtype:
                    df_main[col_conv] = pd.to_numeric(df_main[col_conv], errors="coerce")
                else:
                    df_main[col_conv] = df_main[col_conv].astype(str)
                st.session_state["df_main"] = df_main
                st.success("Conversão aplicada.")
            except Exception as e:
                st.error(e)

    # PIVOT / MERGE / UNPIVOT
    with tabs[2]:
        st.markdown("### Pivot / Unpivot / Merge")
        mode = st.selectbox("Modo", ["Pivot", "Unpivot", "Merge"])
        if mode == "Pivot":
            idx = st.multiselect("Index (linhas)", df_main.columns.tolist())
            col = st.selectbox("Columns", df_main.columns.tolist())
            vals = st.selectbox("Values (numéricas)", df_main.select_dtypes(include=np.number).columns.tolist())
            agg = st.selectbox("Agg", ["sum", "mean", "count", "min", "max"])
            if st.button("Pivotar"):
                try:
                    res = pivot_transform(df_main, idx, col, vals, agg)
                    st.session_state["df_main"] = res
                    st.success("Pivot realizado.")
                except Exception as e:
                    st.error(e)
        elif mode == "Unpivot":
            ids = st.multiselect("id_vars", df_main.columns.tolist())
            vals = st.multiselect("value_vars", df_main.columns.tolist())
            if st.button("Unpivot"):
                try:
                    res = unpivot_transform(df_main, ids, vals)
                    st.session_state["df_main"] = res
                    st.success("Unpivot realizado.")
                except Exception as e:
                    st.error(e)
        else:  # Merge
            st.info("Faça upload do arquivo a ser mesclado (right)")
            uf = st.file_uploader("Arquivo para merge (right)", key="merge_file", type=["csv", "xlsx"])
            if uf is not None:
                try:
                    r = safe_read(uf)
                    r = clean_colnames(r)
                    left_keys = st.multiselect("Left keys (this df)", df_main.columns.tolist())
                    right_keys = st.multiselect("Right keys (uploaded df)", r.columns.tolist())
                    how = st.selectbox("Tipo de join", ["left", "inner", "right", "outer"])
                    if st.button("Executar Merge") and left_keys and right_keys and len(left_keys) == len(right_keys):
                        merged = merge_datasets(df_main, r, left_keys, right_keys, how=how)
                        st.session_state["df_main"] = merged
                        st.success("Merge realizado.")
                    elif st.button("Executar Merge") and (not left_keys or not right_keys):
                        st.error("Defina chaves para merge.")
                except Exception as e:
                    st.error(e)

    # FILTERS
    with tabs[3]:
        st.markdown("### Filtros e limpeza rápida")
        colf = st.selectbox("Filtrar por coluna", df_main.columns.tolist(), key="filter_col")
        if pd.api.types.is_numeric_dtype(df_main[colf]):
            mn, mx = float(df_main[colf].min()), float(df_main[colf].max())
            rng = st.slider("Intervalo", mn, mx, (mn, mx))
            if st.button("Aplicar filtro numérico"):
                st.session_state["df_main"] = df_main[(df_main[colf] >= rng[0]) & (df_main[colf] <= rng[1])]
                st.success("Filtro aplicado.")
        elif np.issubdtype(df_main[colf].dtype, np.datetime64):
            min_d, max_d = df_main[colf].min().date(), df_main[colf].max().date()
            dr = st.date_input("Período", [min_d, max_d])
            if st.button("Aplicar filtro de data"):
                st.session_state["df_main"] = df_main[(df_main[colf].dt.date >= dr[0]) & (df_main[colf].dt.date <= dr[1])]
                st.success("Filtro de data aplicado.")
        else:
            vals = df_main[colf].dropna().unique().tolist()
            if len(vals) <= 200:
                sel = st.multiselect("Valores", vals, default=vals)
                if st.button("Aplicar filtro categórico"):
                    st.session_state["df_main"] = df_main[df_main[colf].isin(sel)]
                    st.success("Filtro categórico aplicado.")
            else:
                txt = st.text_input("Pesquisar")
                if st.button("Aplicar filtro por texto"):
                    st.session_state["df_main"] = df_main[df_main[colf].astype(str).str.contains(txt, case=False, na=False)]
                    st.success("Filtro por texto aplicado.")
        if st.button("Remover linhas com NA"):
            old = len(df_main)
            st.session_state["df_main"] = df_main.dropna()
            st.success(f"Removidas {old - len(st.session_state['df_main'])} linhas.")

# ---------------------------
# Visual Studio (charts)
# ---------------------------
elif menu == "Visual Studio":
    st.header("🎨 Visual Studio (Criador de Gráficos)")
    left, right = st.columns([1, 2])
    with left:
        chart_type = st.selectbox("Tipo de gráfico", ["Barras", "Linha", "Dispersão", "Pizza", "Histograma", "Box", "Heatmap"])
        x_col = st.selectbox("Eixo X", df_main.columns.tolist())
        y_col = None
        if chart_type not in ("Pizza", "Histograma", "Heatmap"):
            y_col = st.selectbox("Eixo Y (valor)", df_main.select_dtypes(include=np.number).columns.tolist())
        color_col = st.selectbox("Cor (opcional)", ["Nenhum"] + df_main.columns.tolist())
        color_col = None if color_col == "Nenhum" else color_col
        agg = None
        if chart_type in ("Barras", "Linha", "Pizza"):
            agg = st.selectbox("Agregação (se aplicável)", ["sum", "mean", "count", "min", "max"])
        theme = st.selectbox("Tema", ["plotly", "plotly_dark", "ggplot2"])
        labels = st.checkbox("Mostrar rótulos", value=False)
        height = st.slider("Altura (px)", 300, 900, 520)
        title = st.text_input("Título", value=f"{chart_type} — {y_col} por {x_col}" if y_col else chart_type)

        if st.button("Gerar gráfico"):
            try:
                fig = build_chart(chart_type, df_main, x_col, y_col, color_col, agg, theme, labels, height)
                st.session_state["last_fig"] = fig
                st.session_state["last_meta"] = {"title": title, "type": chart_type}
                st.success("Gráfico gerado.")
            except Exception as e:
                st.error(f"Erro ao gerar gráfico: {e}")

    with right:
        if st.session_state.get("last_fig") is not None:
            st.plotly_chart(st.session_state["last_fig"], width="stretch")
            note = st.text_area("Nota/Comentário para o relatório", key="chart_note")
            col_add, col_clear = st.columns([1, 1])
            with col_add:
                if st.button("➕ Adicionar ao Relatório"):
                    st.session_state["report_charts"].append({
                        "fig": st.session_state["last_fig"],
                        "title": st.session_state["last_meta"].get("title", "Gráfico"),
                        "type": st.session_state["last_meta"].get("type", ""),
                        "note": note
                    })
                    st.success("Adicionado ao relatório.")
            with col_clear:
                if st.button("Limpar gráfico atual"):
                    st.session_state["last_fig"] = None
                    st.session_state["last_meta"] = {}
        else:
            st.info("Gere um gráfico e adicione ao relatório.")

# ---------------------------
# SQL Lab (DuckDB or sqlite fallback)
# ---------------------------
elif menu == "SQL Lab":
    st.header("💾 SQL Playground")
    st.markdown("Tabela disponível: **dados** (o dataset carregado). Use SQL para consulta rápida.")
    default = "SELECT * FROM dados LIMIT 100"
    sql = st.text_area("Escreva sua query SQL", value=default, height=180)
    if st.button("Executar SQL"):
        res, err = de.run_query(df_main, sql)
        if err:
            st.error(f"Erro SQL: {err}")
        else:
            st.success(f"{len(res)} linhas retornadas")
            safe_display_df(res.head(500))
            csv = res.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar resultado (CSV)", csv, "sql_result.csv", "text/csv")

# ---------------------------
# Relatório / Dashboard builder
# ---------------------------
elif menu == "Relatório/Dashboard":
    st.header("📑 Relatório / Dashboard Builder")
    charts = st.session_state.get("report_charts", [])
    col_top, col_actions = st.columns([3, 1])
    col_top.subheader(f"Relatório — {len(charts)} gráficos")
    if col_actions.button("Limpar relatório"):
        st.session_state["report_charts"] = []
        st.experimental_rerun()

    if not charts:
        st.info("Relatório vazio. Vá ao Visual Studio e adicione gráficos.")
    for idx, ch in enumerate(charts):
        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(ch["fig"], width="stretch")
        with c2:
            st.write(f"**{ch.get('title','(sem título)')}**")
            st.write(f"Tipo: {ch.get('type','')}")
            if ch.get("note"):
                st.info(ch.get("note"))
            if st.button(f"Remover gráfico #{idx+1}", key=f"del_{idx}"):
                st.session_state["report_charts"].pop(idx)
                st.experimental_rerun()

# ---------------------------
# Export
# ---------------------------
elif menu == "Exportar":
    st.header("📤 Exportar dados e relatório")
    df_export = st.session_state["df_main"]
    col1, col2 = st.columns(2)
    with col1:
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV (dados tratados)", csv, "dados_tratados.csv", "text/csv")
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Dados")
        st.download_button("Baixar Excel (dados tratados)", buffer.getvalue(), "dados_tratados.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.subheader("Gerar Laudo Técnico (PDF)")
    notes = st.text_area("Observações do laudo (aparecerão no relatório)", height=120)
    kpis = {"rows": len(df_export), "cols": df_export.shape[1], "nulls": int(df_export.isna().sum().sum()), "dups": int(df_export.duplicated().sum())}
    if st.button("Gerar PDF (Laudo Técnico)"):
        try:
            pdf_bytes = generate_pdf_laudo(df_export, st.session_state.get("report_charts", []), kpis, notes)
            st.download_button("Download Laudo (PDF)", pdf_bytes, "laudo_tecnico.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")
            # fallback: return a textual report as .txt
            rep = f"Relatório: KPIs: {kpis}\nObs: {notes}"
            st.download_button("Baixar relatório (TXT)", rep.encode("utf-8"), "relatorio.txt", "text/plain")

# persist back
st.session_state["df_main"] = st.session_state.get("df_main", df_main)
