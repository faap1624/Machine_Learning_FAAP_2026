import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración inicial de la página
st.set_page_config(
    page_title="Analizador de Datos con Streamlit",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analizador y Visor de Datos Dinámico")
st.markdown("Sube tu archivo **CSV** o **Excel** para explorar métricas, consultar la estructura de los datos y generar visualizaciones al instante.")

# --- BARRA LATERAL: CARGA DE ARCHIVOS ---
st.sidebar.header("📁 1. Carga de Archivos")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona un archivo de datos", 
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        # Detectar el tipo de archivo y leerlo con Pandas
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.sidebar.success(f"¡Archivo **{uploaded_file.name}** cargado con éxito!")

        # --- MÉTRICAS GENERALES ---
        st.subheader("📌 Resumen General")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Filas", f"{df.shape[0]:,}")
        col2.metric("Total de Columnas", df.shape[1])
        col3.metric("Valores Nulos Totales", int(df.isna().sum().sum()))

        # --- VISTA PREVIA DEL DATAFRAME ---
        with st.expander("🔍 Vista previa de los datos (DataFrame)"):
            st.dataframe(df, use_container_width=True)

        # --- ESTADÍSTICAS DESCRIPTIVAS ---
        with st.expander("📈 Estadísticas Descriptivas"):
            st.write(df.describe())

        # --- VISUALIZACIÓN RÁPIDA CON PLOTLY ---
        st.markdown("---")
        st.subheader("📈 Generador de Gráficos Interactivos")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()

        if len(numeric_cols) >= 1 and len(all_cols) >= 2:
            col_x = st.selectbox("Selecciona la variable para el Eje X", all_cols, index=0)
            col_y = st.selectbox("Selecciona la variable numérica para el Eje Y", numeric_cols, index=0)
            color_col = st.selectbox("Agrupar por color (Opcional)", [None] + all_cols)

            # Gráfico de dispersión o líneas según convenga
            fig = px.scatter(
                df, 
                x=col_x, 
                y=col_y, 
                color=color_col, 
                title=f"Relación de {col_y} vs {col_x}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("El dataset necesita al menos una columna numérica para generar gráficos.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("👈 Utiliza la barra lateral para cargar un archivo CSV o Excel y comenzar.")