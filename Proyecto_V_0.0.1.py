import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analítico Avanzado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard Analítico Avanzado")
st.markdown("Explora métricas, aplica **filtros dinámicos multinivel** y analiza múltiples proyecciones interactivas.")

# ---------------------------------------------------------
# GENERACIÓN / CARGA DE DATOS
# ---------------------------------------------------------
@st.cache_data
def generar_datos_prueba():
    np.random.seed(42)
    n = 500
    fechas = pd.date_range(start="2026-01-01", periods=n, freq="D")
    regiones = ["Norte", "Centro", "Sur", "Occidente"]
    categorias = ["Electrónica", "Servicios", "Industrial", "Software"]
    vendedores = ["Ana", "Carlos", "Diana", "Eduardo", "Sofía"]
    
    df = pd.DataFrame({
        "Fecha": np.random.choice(fechas, size=n),
        "Region": np.random.choice(regiones, size=n, p=[0.3, 0.3, 0.2, 0.2]),
        "Categoria": np.random.choice(categorias, size=n),
        "Vendedor": np.random.choice(vendedores, size=n),
        "Ventas_USD": np.random.uniform(500, 15000, size=n).round(2),
        "Costos_USD": np.random.uniform(200, 8000, size=n).round(2),
        "Satisfaccion_Cliente": np.random.uniform(1.0, 5.0, size=n).round(1),
        "Unidades_Vendidas": np.random.randint(1, 50, size=n)
    })
    df["Ganancia_USD"] = (df["Ventas_USD"] - df["Costos_USD"]).round(2)
    df["Margen_%"] = ((df["Ganancia_USD"] / df["Ventas_USD"]) * 100).round(2)
    return df.sort_values("Fecha").reset_index(drop=True)

# Barra lateral: Opción de carga de archivo propio o dataset demo
st.sidebar.header("📁 Carga de Datos")
fuente_datos = st.sidebar.radio("Origen de datos:", ["Dataset de Ejemplo (Demo)", "Subir archivo propio (CSV/Excel)"])

if fuente_datos == "Subir archivo propio (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Selecciona archivo", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success(f"¡{uploaded_file.name} cargado correctamente!")
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
            st.stop()
    else:
        st.info("👈 Sube un archivo en la barra lateral para continuar o selecciona 'Dataset de Ejemplo'.")
        st.stop()
else:
    df_raw = generar_datos_prueba()

# ---------------------------------------------------------
# FILTROS AVANZADOS EN LA BARRA LATERAL
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Filtros Avanzados")

df_filtered = df_raw.copy()

# Detectar columnas categóricas y de fecha
columnas_texto = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
columnas_num = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
columnas_fecha = df_filtered.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

# Si hay fechas, agregar filtro de rango temporal
if columnas_fecha:
    col_fecha = columnas_fecha[0]
    min_date = df_filtered[col_fecha].min().date()
    max_date = df_filtered[col_fecha].max().date()
    rango_fecha = st.sidebar.date_input(
        f"Rango de {col_fecha}",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(rango_fecha) == 2:
        df_filtered = df_filtered[
            (df_filtered[col_fecha].dt.date >= rango_fecha[0]) & 
            (df_filtered[col_fecha].dt.date <= rango_fecha[1])
        ]

# Filtros dinámicos multinivel para variables categóricas
for col in columnas_texto:
    opciones = list(df_raw[col].unique())
    seleccionados = st.sidebar.multiselect(f"Filtrar por {col}", opciones, default=opciones)
    if seleccionados:
        df_filtered = df_filtered[df_filtered[col].isin(seleccionados)]

# Filtro por rango numérico (slider dinámico)
if columnas_num:
    col_num_filtro = st.sidebar.selectbox("Filtro Numérico por Columna", columnas_num)
    val_min = float(df_raw[col_num_filtro].min())
    val_max = float(df_raw[col_num_filtro].max())
    rango_num = st.sidebar.slider(
        f"Rango de {col_num_filtro}",
        min_value=val_min,
        max_value=val_max,
        value=(val_min, val_max)
    )
    df_filtered = df_filtered[
        (df_filtered[col_num_filtro] >= rango_num[0]) & 
        (df_filtered[col_num_filtro] <= rango_num[1])
    ]

# ---------------------------------------------------------
# MÉTRICAS Y KPIS PRINCIPALES
# ---------------------------------------------------------
st.subheader("📌 Indicadores Clave de Desempeño (KPIs)")
if df_filtered.empty:
    st.warning("No hay datos disponibles con los filtros aplicados.")
    st.stop()

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

total_registros = len(df_filtered)
col_kpi1.metric("Total Registros", f"{total_registros:,}")

if "Ventas_USD" in df_filtered.columns:
    col_kpi2.metric("Ventas Totales", f"${df_filtered['Ventas_USD'].sum():,.2f}")
elif len(columnas_num) > 0:
    col_kpi2.metric(f"Suma {columnas_num[0]}", f"{df_filtered[columnas_num[0]].sum():,.2f}")

if "Ganancia_USD" in df_filtered.columns:
    col_kpi3.metric("Ganancia Total", f"${df_filtered['Ganancia_USD'].sum():,.2f}")
elif len(columnas_num) > 1:
    col_kpi3.metric(f"Promedio {columnas_num[1]}", f"{df_filtered[columnas_num[1]].mean():,.2f}")

if "Margen_%" in df_filtered.columns:
    col_kpi4.metric("Margen Promedio", f"{df_filtered['Margen_%'].mean():.2f}%")
elif len(columnas_num) > 2:
    col_kpi4.metric(f"Máximo {columnas_num[2]}", f"{df_filtered[columnas_num[2]].max():,.2f}")

st.markdown("---")

# ---------------------------------------------------------
# SECCIÓN DE MÚLTIPLES GRÁFICOS INTERACTIVOS (PLOTLY)
# ---------------------------------------------------------
st.subheader("📈 Análisis Multidimensional de Datos")

tab1, tab2, tab3 = st.tabs(["📊 Tendencias y Composición", "🔬 Dispersión y Correlación", "📦 Distribución y Outliers"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("##### 1. Serie Temporal / Tendencia")
        if columnas_fecha and len(columnas_num) > 0:
            eje_y_temp = st.selectbox("Métrica temporal", columnas_num, index=0, key="t1")
            df_temp = df_filtered.groupby(col_fecha)[eje_y_temp].sum().reset_index()
            fig_line = px.line(df_temp, x=col_fecha, y=eje_y_temp, title=f"Evolución de {eje_y_temp}", template="plotly_white")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Requiere columna de fecha y variables numéricas.")

    with col_g2:
        st.markdown("##### 2. Composición Categórica")
        if len(columnas_texto) > 0 and len(columnas_num) > 0:
            cat_pie = st.selectbox("Categoría a desagregar", columnas_texto, index=0, key="t2")
            val_pie = st.selectbox("Métrica en Pastel", columnas_num, index=0, key="t3")
            fig_pie = px.pie(df_filtered, names=cat_pie, values=val_pie, title=f"Distribución de {val_pie} por {cat_pie}", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.markdown("##### 3. Gráfico de Dispersión Interactivo (Scatter Plot)")
    if len(columnas_num) >= 2:
        c1, c2, c3 = st.columns(3)
        x_scat = c1.selectbox("Eje X", columnas_num, index=0, key="s1")
        y_scat = c2.selectbox("Eje Y", columnas_num, index=min(1, len(columnas_num)-1), key="s2")
        color_scat = c3.selectbox("Color por", [None] + columnas_texto, index=1 if len(columnas_texto)>0 else 0, key="s3")
        
        fig_scatter = px.scatter(
            df_filtered, x=x_scat, y=y_scat, color=color_scat,
            hover_data=columnas_texto, title=f"Relación {y_scat} vs {x_scat}", template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("##### 4. Histograma de Distribución")
        if len(columnas_num) > 0:
            var_hist = st.selectbox("Variable para Histograma", columnas_num, index=0, key="h1")
            fig_hist = px.histogram(df_filtered, x=var_hist, nbins=30, title=f"Distribución de {var_hist}", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
    with col_d2:
        st.markdown("##### 5. Diagrama de Caja (Boxplot)")
        if len(columnas_num) > 0:
            var_box = st.selectbox("Variable Numérica (Boxplot)", columnas_num, index=0, key="b1")
            cat_box = st.selectbox("Agrupar por (Opcional)", [None] + columnas_texto, index=0, key="b2")
            fig_box = px.box(df_filtered, y=var_box, x=cat_box, title=f"Detección de outliers en {var_box}", template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# VISTA PREVIA Y DESCARGA DE DATOS FILTRADOS
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📋 Ver Tabla de Datos Filtrados y Exportar"):
    st.dataframe(df_filtered, use_container_width=True)
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados (CSV)",
        data=csv_data,
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )