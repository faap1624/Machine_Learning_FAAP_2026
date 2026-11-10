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