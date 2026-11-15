import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Mantenimiento Predictivo - Sistema de Predicción",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Sistema de Mantenimiento Predictivo")
st.markdown("Carga de modelos entrenados con SMOTE, generación de nuevas instancias y predicción individual o por lotes.")

# -------------------------------------------------------------------------
# 1. Funciones con caché para cargar datos y modelos
# -------------------------------------------------------------------------
@st.cache_data
def cargar_datos_referencia():
    ruta_csv = "mantenimiento_predictivo_desbalanceado.csv"
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv)
        X = df.drop(columns=["falla"], errors="ignore")
        return X
    return None

@st.cache_resource
def cargar_modelo(ruta_modelo):
    return joblib.load(ruta_modelo)

# -------------------------------------------------------------------------
# 2. Barra Lateral: Selección de Modelo y Umbral
# -------------------------------------------------------------------------
st.sidebar.header("🔧 Configuración del Modelo")

carpeta_modelos = "modelos_guardados"
modelos_disponibles = []

if os.path.exists(carpeta_modelos):
    modelos_disponibles = [f for f in os.listdir(carpeta_modelos) if f.endswith(".joblib")]

if not modelos_disponibles:
    st.sidebar.error(f"No se encontraron archivos .joblib en la carpeta '{carpeta_modelos}/'. Ejecuta primero tu script de entrenamiento.")
    st.stop()

modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo guardado:",
    modelos_disponibles,
    index=modelos_disponibles.index("modelo_random_forest_smote.joblib") if "modelo_random_forest_smote.joblib" in modelos_disponibles else 0
)

ruta_completa_modelo = os.path.join(carpeta_modelos, modelo_seleccionado)
modelo = cargar_modelo(ruta_completa_modelo)
st.sidebar.success(f"Modelo cargado: `{modelo_seleccionado}`")

# Control de umbral de decisión (Threshold Tuning)
umbral = st.sidebar.slider(
    "Umbral de decisión para Falla (%):",
    min_value=1, max_value=99, value=50, step=1,
    help="Por defecto es 50%. Bájalo si deseas detectar más fallas reales (mayor Recall)."
) / 100.0

# Cargar estructura de columnas desde el CSV de referencia
X_ref = cargar_datos_referencia()

if X_ref is None:
    st.error("No se encontró 'mantenimiento_predictivo_desbalanceado.csv' en el directorio actual. Necesario para conocer la estructura y rangos de las variables.")
    st.stop()

columnas_features = list(X_ref.columns)

# -------------------------------------------------------------------------
# 3. Pestañas Principales
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Generación de Instancias y Lotes", "🔍 Predicción de 1 Caso Individual"])

# =========================================================================
# PESTAÑA 1: GENERACIÓN DE NUEVAS INSTANCIAS Y PREDICCIÓN POR LOTES
# =========================================================================
with tab1:
    st.subheader("Generar nuevas instancias y predecir")
    st.write("Simula lecturas de sensores basadas en la distribución estadística real de tu dataset o sube un archivo CSV propio.")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    
    with col_ctrl1:
        n_instancias = st.number_input("Cantidad de nuevas instancias a generar:", min_value=1, max_value=5000, value=20, step=5)
        metodo_gen = st.radio(
            "Método de generación:",
            ["Muestreo con ruido gaussiano (Realista)", "Aleatorio uniforme (Min - Max)"]
        )
        btn_generar = st.button("🚀 Generar y Predecir", type="primary", use_container_width=True)

    if btn_generar:
        # Generar datos sintéticos nuevos
        if metodo_gen == "Muestreo con ruido gaussiano (Realista)":
            # Toma filas al azar y añade una ligera variación estadística (10% de std)
            muestra_base = X_ref.sample(n=n_instancias, replace=True).reset_index(drop=True)
            nuevos_datos = muestra_base.copy()
            for col in columnas_features:
                if pd.api.types.is_numeric_dtype(X_ref[col]):
                    std = X_ref[col].std()
                    ruido = np.random.normal(0, std * 0.15, size=n_instancias)
                    nuevos_datos[col] = nuevos_datos[col] + ruido
                    # Si la columna original era entera, redondear
                    if pd.api.types.is_integer_dtype(X_ref[col]):
                        nuevos_datos[col] = nuevos_datos[col].round().astype(int)
        else:
            # Valores uniformes entre mínimo y máximo de cada columna
            data_dict = {}
            for col in columnas_features:
                if pd.api.types.is_integer_dtype(X_ref[col]):
                    data_dict[col] = np.random.randint(int(X_ref[col].min()), int(X_ref[col].max()) + 1, size=n_instancias)
                else:
                    data_dict[col] = np.random.uniform(X_ref[col].min(), X_ref[col].max(), size=n_instancias)
            nuevos_datos = pd.DataFrame(data_dict)

        # Realizar predicciones
        probabilidades = modelo.predict_proba(nuevos_datos)[:, 1]
        predicciones = (probabilidades >= umbral).astype(int)

        # Construir tabla de resultados
        df_resultado = nuevos_datos.copy()
        df_resultado.insert(0, "Probabilidad_Falla (%)", (probabilidades * 100).round(2))
        df_resultado.insert(0, "Prediccion", np.where(predicciones == 1, "⚠️ FALLA (1)", "✅ Normal (0)"))

        # Métricas resumen
        total_fallas = int(predicciones.sum())
        total_normales = int(n_instancias - total_fallas)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Instancias Evaluadas", n_instancias)
        m2.metric("Equipos Normales (0)", total_normales)
        m3.metric("Fallas Detectadas (1)", total_fallas, delta=f"{(total_fallas/n_instancias)*100:.1f}%", delta_color="inverse")

        st.dataframe(df_resultado, use_container_width=True)

        # Botón de descarga CSV
        csv_descarga = df_resultado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar nuevas instancias con predicciones (.csv)",
            data=csv_descarga,
            file_name="nuevas_instancias_predichas.csv",
            mime="text/csv"
        )

# =========================================================================
# PESTAÑA 2: PREDICCIÓN DE 1 SOLO CASO (FORMULARIO INTERACTIVO)
# =========================================================================
with tab2:
    st.subheader("Evaluar un caso individual")
    st.write("Ajusta los valores de cada sensor o variable manualmente para evaluar el riesgo de falla de un solo equipo.")

    # Botón auxiliar para rellenar con un caso aleatorio del dataset
    if "valores_caso" not in st.session_state:
        st.session_state.valores_caso = X_ref.median().to_dict()

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    if col_btn1.button("🎲 Cargar ejemplo aleatorio"):
        fila_azar = X_ref.sample(1).iloc[0]
        st.session_state.valores_caso = fila_azar.to_dict()
        st.rerun()

    if col_btn2.button("🔄 Restablecer a valores medianos"):
        st.session_state.valores_caso = X_ref.median().to_dict()
        st.rerun()

    # Formulario dinámico que se adapta automáticamente a las columnas de tu CSV
    with st.form("form_caso_individual"):
        inputs_usuario = {}
        cols_grid = st.columns(3)  # Organiza los inputs en 3 columnas

        for idx, col in enumerate(columnas_features):
            col_actual = cols_grid[idx % 3]
            val_defecto = st.session_state.valores_caso.get(col, float(X_ref[col].median()))

            if pd.api.types.is_integer_dtype(X_ref[col]):
                inputs_usuario[col] = col_actual.number_input(
                    label=f"{col}",
                    value=int(val_defecto),
                    step=1
                )
            else:
                inputs_usuario[col] = col_actual.number_input(
                    label=f"{col}",
                    value=float(val_defecto),
                    format="%.4f"
                )

        submit_individual = st.form_submit_button("🔍 Predecir este caso", type="primary", use_container_width=True)

    if submit_individual:
        df_caso = pd.DataFrame([inputs_usuario])
        prob_caso = modelo.predict_proba(df_caso)[0, 1]
        pred_caso = int(prob_caso >= umbral)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Probabilidad de Falla Calculada",
                value=f"{prob_caso * 100:.2f}%",
                delta=f"Umbral activo: {umbral * 100:.0f}%",
                delta_color="off"
            )

        with res_col2:
            if pred_caso == 1:
                st.error(f"### ⚠️ ALERTA DE FALLA DETECTADA (Clase 1)\nLa probabilidad ({prob_caso*100:.2f}%) supera o iguala el umbral establecido ({umbral*100:.0f}%).")
            else:
                st.success(f"### ✅ OPERACIÓN NORMAL (Clase 0)\nLa probabilidad de falla ({prob_caso*100:.2f}%) está por debajo del umbral establecido ({umbral*100:.0f}%).")
                
