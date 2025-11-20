"""
Enterprise Analytics — BI Edition (No-Code) — Refactor v3
- Fixes deprecation warnings for Streamlit 1.51.0+ (width="stretch")
- Removes legacy ML artifacts causing PCA errors
- Single-file Streamlit app (modular functions/classes)
- Robust file reading (CSV/Excel, encodings fallback)
- Data Studio: create columns, conditional columns, date extraction, pivot/unpivot, join/merge, split text
- Visual Studio: many chart types + presets + palettes + add-to-dashboard
- Dashboard Builder: assemble charts, remove, export
- Data Quality: KPIs, missing heatmap, outliers detection
- Export: CSV, Excel, PDF (basic)
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

# DEV LOCAL PATH
UPLOADED_FILE_PATH = "/mnt/data/uploaded_dataset.csv" 

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
            name = getattr(file, "name", "")
            if name.lower().endswith(".csv"):
                try:
                    return try_read_csv(file)
                except Exception:
                    file.seek(0)
                    return pd.read_csv(file, engine="python", encoding_errors="ignore")
            else:
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
    
    # Short stats
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
    m = df.isna().astype(int)
    if m.shape[1] > 50:
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
            if operator == '+': df[new_col] = df[a] + df[b]
            elif operator == '-': df[new_col] = df[a] - df[b]
            elif operator == '*': df[new_col] = df[a] * df[b]
            elif operator == '/': df[new_col] = df[a] / df[b].replace(0, np.nan)
        else:
            if operator == '+': df[new_col] = df[a] + b_value
            elif operator == '-': df[new_col] = df[a] - b_value
            elif operator == '*': df[new_col] = df[a] * b_value
            elif operator == '/': df[new_col] = df[a] / b_value if b_value != 0 else np.nan
    except Exception:
        raise
    return df

def create_column_if(df: pd.DataFrame, new_col: str, col: str, op: str, threshold: float, true_label: str, false_label: str):
    df = df.copy()
    ops = {">": df[col] > threshold, "<": df[col] < threshold, ">=": df[col] >= threshold, "<=": df[col] <= threshold, "==": df[col] == threshold, "!=": df[col] != threshold}
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
    df = df.copy()
    aggfunc_map = {"sum":"sum", "mean":"mean", "count":"count", "min":"min", "max":"max"}
    if aggfunc not in aggfunc_map: aggfunc = "sum"
    res = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc_map[aggfunc]).reset_index()
    res.columns = [f"{a}" if not isinstance(a, tuple) else "_".join([str(x) for x in a if x is not None]) for a in res.columns]
    return res

def unpivot_transform(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name="variable", value_name="value"):
    df = df.copy()
    res = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return res

def merge_datasets(left: pd.DataFrame, right: pd.DataFrame, left_on: List[str], right_on: List[str], how="left"):
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how)

# ---------------------------
# Visual Studio builder
# ---------------------------
def build_chart(chart_type: str, df: pd.DataFrame, x: str=None, y: str=None, color: str=None,
                agg: Optional[str]=None, theme: str='plotly', show_labels: bool=True,
                height: int=500, size: Optional[str]=None):
    plot_df = df.copy()
    if agg and x and y and chart_type in ("Barras","Linha","Pizza"):
        if color: plot_df = plot_df.groupby([x, color])[y].agg(agg).reset_index()
        else: plot_df = plot_df.groupby(x)[y].agg(agg).reset_index()
    
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

def sidebar_load():
    st.sidebar.header("📂 Conectar Dados")
    uploaded = st.sidebar.file_uploader("Arraste CSV/Excel", type=['csv','xlsx'])
    use_local = st.sidebar.checkbox("Usar caminho local (dev)", value=False)
    
    if st.sidebar.button("🔄 Resetar Sessão"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    return uploaded, use_local

# Session State Init
if 'df_raw' not in st.session_state: st.session_state['df_raw'] = pd.DataFrame()
if 'df_main' not in st.session_state: st.session_state['df_main'] = pd.DataFrame()
if 'report_charts' not in st.session_state: st.session_state['report_charts'] = []
if 'presets' not in st.session_state: st.session_state['presets'] = {}

def main():
    uploaded, use_local = sidebar_load()
    st.title("Enterprise Analytics — BI Edition")
    st.caption("Foco: EDA, Visualização e Dashboard builder sem código")

    # Load Logic
    if uploaded is None and use_local:
        try:
            df = safe_read(UPLOADED_FILE_PATH)
            df = clean_colnames(df)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success("Dados carregados do caminho local.")
        except Exception as e:
            st.sidebar.error(f"Erro local: {e}")
    elif uploaded is not None:
        try:
            df = safe_read(uploaded)
            df = clean_colnames(df)
            st.session_state['df_raw'] = df.copy()
            st.session_state['df_main'] = df.copy()
            st.success(f"Arquivo '{getattr(uploaded,'name', 'uploaded')}' carregado.")
        except Exception as e:
            st.sidebar.error(f"Erro leitura: {e}")

    if st.session_state['df_main'].empty:
        st.markdown("### Carregue um arquivo CSV ou Excel na barra lateral para começar.")
        return

    menu = st.radio("", ["Data Quality", "Data Studio", "Visual Studio", "Relatório/Dashboard", "Exportar"], horizontal=True, index=0)
    df = st.session_state['df_main']

    # ---- Data Quality ----
    if menu == "Data Quality":
        st.header("🔍 Data Quality & EDA")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-box'><h3>Linhas</h3><h2>{format_number(df.shape[0])}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><h3>Colunas</h3><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><h3>Nulos</h3><h2>{format_number(int(df.isna().sum().sum()))}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'><h3>Duplicatas</h3><h2>{format_number(int(df.duplicated().sum()))}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Mapa de Missing")
        st.plotly_chart(missing_heatmap(df), width="stretch")

        st.subheader("Perfil das Colunas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Numéricas")
            num = df.select_dtypes(include=np.number)
            if not num.empty:
                st.dataframe(num.describe().T[['mean','std','min','max']], width="stretch")
                col_sel = st.selectbox("Detectar outliers (IQR) em:", num.columns.tolist())
                out_df = detect_outliers_iqr(df, col_sel)
                st.write(f"Outliers detectados: {len(out_df)}")
                if st.checkbox("Mostrar outliers sample"):
                    st.dataframe(out_df.head(50), width="stretch")
            else:
                st.info("Sem colunas numéricas.")

        with col2:
            st.caption("Categóricas")
            cat = df.select_dtypes(include='object')
            if not cat.empty:
                stats_cat = pd.DataFrame({"unique": cat.nunique(), "missing": cat.isna().sum(), "% missing": (cat.isna().mean()*100).round(2)})
                st.dataframe(stats_cat, width="stretch")
            else:
                st.info("Sem colunas de texto.")

        with col3:
            st.caption("Datas")
            date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if date_cols:
                for dc in date_cols:
                    st.write(f"**{dc}**: {df[dc].min()} -> {df[dc].max()}")
            else:
                st.info("Sem colunas de data.")

    # ---- Data Studio ----
    elif menu == "Data Studio":
        st.header("🛠 Data Studio")
        tabs = st.tabs(["Criar Colunas", "Renomear/Converter", "Pivot/Merge", "Filtros"])
        
        with tabs[0]:
            op = st.selectbox("Operação", ["Aritmética", "Condicional (IF)", "Extrair Data", "Split Texto"])
            if op == "Aritmética":
                lc = st.selectbox("Col A", df.select_dtypes(include=np.number).columns.tolist(), key="csa")
                md = st.radio("Col B", ["Coluna","Valor Fixo"])
                rc = st.selectbox("Col B", df.select_dtypes(include=np.number).columns.tolist(), key="csb") if md=="Coluna" else None
                bv = st.number_input("Valor", 1.0) if md=="Valor Fixo" else None
                sy = st.selectbox("Op", ["+","-","*","/"])
                nm = st.text_input("Nome nova col", f"{lc}_{sy}_res")
                if st.button("Criar"):
                    try:
                        st.session_state['df_main'] = create_column_arithmetic(df, nm, lc, rc, sy, bv)
                        st.success("Criado.")
                    except Exception as e: st.error(e)
            
            elif op == "Condicional (IF)":
                nc = st.selectbox("Col Alvo", df.select_dtypes(include=np.number).columns.tolist())
                opr = st.selectbox("Op", [">","<",">=","<=","==","!="])
                th = st.number_input("Threshold", 0.0)
                tl, fl = st.text_input("True Label", "High"), st.text_input("False Label", "Low")
                nm = st.text_input("Nome", "cat_if")
                if st.button("Criar IF"):
                    st.session_state['df_main'] = create_column_if(df, nm, nc, opr, th, tl, fl)
                    st.success("Criado.")
            
            elif op == "Extrair Data":
                dc = st.selectbox("Data Col", [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)])
                cp = st.selectbox("Componente", ["year","month","day","weekday"])
                nm = st.text_input("Nome", f"{dc}_{cp}")
                if st.button("Extrair") and dc:
                    st.session_state['df_main'] = extract_date_component(df, dc, cp, nm)
                    st.success("Extraído.")
            
            elif op == "Split Texto":
                tc = st.selectbox("Texto Col", df.select_dtypes(include='object').columns.tolist())
                sp = st.text_input("Separador", " ")
                pt = st.number_input("Index Parte", 0, step=1)
                nm = st.text_input("Nome", f"{tc}_split")
                if st.button("Split") and tc:
                    try:
                        s = df[tc].astype(str).str.split(sp, expand=True)
                        if pt < s.shape[1]:
                            df[nm] = s[pt]
                            st.session_state['df_main'] = df
                            st.success("Split OK.")
                        else: st.error("Index fora do range.")
                    except Exception as e: st.error(e)

        with tabs[1]:
            c = st.selectbox("Col", df.columns, key="ren")
            nn = st.text_input("Novo nome")
            if st.button("Renomear") and nn:
                df.rename(columns={c:nn}, inplace=True)
                st.session_state['df_main'] = df
                st.success("Renomeado.")
            st.markdown("---")
            cc = st.selectbox("Col", df.columns, key="conv")
            to = st.selectbox("Para", ["Data","Num","Texto"])
            if st.button("Converter"):
                try:
                    if to=="Data": df[cc] = pd.to_datetime(df[cc], errors='coerce')
                    elif to=="Num": df[cc] = pd.to_numeric(df[cc], errors='coerce')
                    else: df[cc] = df[cc].astype(str)
                    st.session_state['df_main'] = df
                    st.success("Convertido.")
                except Exception as e: st.error(e)

        with tabs[2]:
            mode = st.radio("Modo", ["Pivot","Unpivot","Merge"])
            if mode=="Pivot":
                idx = st.multiselect("Index", df.columns)
                col = st.selectbox("Columns", df.columns)
                val = st.selectbox("Values", df.select_dtypes(include=np.number).columns)
                af = st.selectbox("Agg", ["sum","mean","count"])
                if st.button("Pivotar"):
                    st.session_state['df_main'] = pivot_transform(df, idx, col, val, af)
                    st.success("Pivotado.")
            elif mode=="Unpivot":
                ids = st.multiselect("Ids", df.columns)
                vals = st.multiselect("Vals", df.columns)
                if st.button("Unpivot"):
                    st.session_state['df_main'] = unpivot_transform(df, ids, vals)
                    st.success("Unpivotado.")
            elif mode=="Merge":
                uf = st.file_uploader("Arquivo 2", key="mg")
                if uf:
                    r = safe_read(uf)
                    r = clean_colnames(r)
                    lo = st.multiselect("Left Keys", df.columns)
                    ro = st.multiselect("Right Keys", r.columns)
                    if st.button("Merge") and lo and ro:
                        st.session_state['df_main'] = merge_datasets(df, r, lo, ro)
                        st.success("Merge OK.")

        with tabs[3]:
            cf = st.selectbox("Filtro Col", df.columns)
            if pd.api.types.is_numeric_dtype(df[cf]):
                mn, mx = float(df[cf].min()), float(df[cf].max())
                rn = st.slider("Range", mn, mx, (mn, mx))
                if st.button("Filtrar Num"):
                    st.session_state['df_main'] = df[(df[cf]>=rn[0]) & (df[cf]<=rn[1])]
                    st.success("Filtrado.")
            else:
                sl = st.multiselect("Vals", df[cf].unique())
                if st.button("Filtrar Cat") and sl:
                    st.session_state['df_main'] = df[df[cf].isin(sl)]
                    st.success("Filtrado.")
            if st.button("Drop NA"):
                st.session_state['df_main'] = df.dropna()
                st.success("NAs removidos.")

    # ---- Visual Studio ----
    elif menu == "Visual Studio":
        st.header("🎨 Visual Studio")
        l, r = st.columns([1,2])
        with l:
            ct = st.selectbox("Tipo", ["Barras","Linha","Dispersão","Pizza","Histograma","Box","Heatmap"])
            x = st.selectbox("X", df.columns)
            y = st.selectbox("Y", df.select_dtypes(include=np.number).columns) if ct not in ("Pizza","Histograma","Heatmap") else None
            clr = st.selectbox("Cor", ["Nenhum"]+df.columns.tolist())
            clr = None if clr=="Nenhum" else clr
            agg = st.selectbox("Agg", ["sum","mean","count"]) if ct in ("Barras","Linha","Pizza") else None
            th = st.selectbox("Tema", ["plotly","plotly_dark","ggplot2"])
            sl = st.checkbox("Labels", True)
            tt = st.text_input("Titulo", f"{ct}")
            
            if st.button("Gerar"):
                try:
                    fig = build_chart(ct, df, x, y, clr, agg, th, sl)
                    st.session_state['last_fig'] = fig
                    st.session_state['last_meta'] = {"title":tt, "type":ct}
                    st.success("Gerado.")
                except Exception as e: st.error(e)

        with r:
            if 'last_fig' in st.session_state:
                st.plotly_chart(st.session_state['last_fig'], width="stretch")
                nt = st.text_area("Nota")
                if st.button("Add ao Relatório"):
                    st.session_state['report_charts'].append({"fig":st.session_state['last_fig'], "title":st.session_state['last_meta']['title'], "type":st.session_state['last_meta']['type'], "note":nt})
                    st.success("Adicionado.")

    # ---- Dashboard ----
    elif menu == "Relatório/Dashboard":
        st.header("📑 Relatório")
        cs = st.session_state['report_charts']
        if st.button("Limpar"):
            st.session_state['report_charts'] = []
            st.rerun()
        
        if not cs: st.info("Vazio.")
        for i, c in enumerate(cs):
            st.markdown("---")
            c1, c2 = st.columns([3,1])
            with c1: st.plotly_chart(c['fig'], width="stretch")
            with c2:
                st.write(f"**{c['title']}**")
                if c.get('note'): st.info(c['note'])
                if st.button(f"X ##{i}"):
                    st.session_state['report_charts'].pop(i)
                    st.rerun()

    # ---- Export ----
    elif menu == "Exportar":
        st.header("📤 Exportar")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("CSV", csv, "data.csv", "text/csv")
        
        kpis = {"rows":len(df), "cols":df.shape[1], "nulls":int(df.isna().sum().sum()), "dups":int(df.duplicated().sum())}
        if st.button("PDF Relatório"):
            b = generate_pdf_report(df, st.session_state['report_charts'], kpis)
            st.download_button("PDF", b, "rep.pdf", "application/pdf")

    st.session_state['df_main'] = st.session_state.get('df_main', df)

if __name__ == "__main__":
    main()