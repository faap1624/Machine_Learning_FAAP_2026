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
st.markdown("Carga tu modelo guardado, ajusta el umbral de decisión, genera nuevas instancias o sube tu propio archivo CSV para predecir.")

# -------------------------------------------------------------------------
# 1. Funciones para cargar datos y modelos
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
st.sidebar.markdown("---")
umbral = st.sidebar.slider(
    "Umbral de decisión para Falla (%):",
    min_value=1, max_value=99, value=50, step=1,
    help="Define el porcentaje mínimo de probabilidad para declarar una falla (Clase 1)."
) / 100.0

# Cargar estructura de columnas desde el CSV de referencia
X_ref = cargar_datos_referencia()

if X_ref is None:
    st.error("No se encontró 'mantenimiento_predictivo_desbalanceado.csv' en el directorio actual.")
    st.stop()

columnas_features = list(X_ref.columns)

# -------------------------------------------------------------------------
# 3. Pestañas Principales
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Generación o Carga de Lotes", "🔍 Predicción de 1 Caso Individual"])

# =========================================================================
# PESTAÑA 1: GENERACIÓN DE INSTANCIAS O SUBIR CSV PROPIO
# =========================================================================
with tab1:
    st.subheader("Evaluación por Lotes (Sintético o CSV)")
    
    modo_lote = st.radio(
        "Selecciona el origen de los datos:",
        ["📁 Subir mi propio archivo CSV", "🤖 Generar nuevas instancias sintéticas"]
    )

    datos_a_predecir = None

    if modo_lote == "📁 Subir mi propio archivo CSV":
        st.markdown("Sube un archivo `.csv` que contenga las columnas: `" + ", ".join(columnas_features) + "`")
        archivo_subido = st.file_uploader("Elige un archivo CSV", type=["csv"])
        
        if archivo_subido is not None:
            try:
                df_user = pd.read_csv(archivo_subido)
                faltantes = [col for col in columnas_features if col not in df_user.columns]
                if faltantes:
                    st.error(f"⚠️ El CSV no tiene las columnas requeridas. Faltan: {faltantes}")
                else:
                    datos_a_predecir = df_user[columnas_features].copy()
                    st.success(f"¡Archivo cargado con éxito! Se encontraron {len(datos_a_predecir)} registros.")
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")

    else:
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            n_instancias = st.number_input("Cantidad de nuevas instancias a generar:", min_value=1, max_value=5000, value=20, step=5)
            metodo_gen = st.radio(
                "Método de generación:",
                ["Muestreo con ruido gaussiano (Realista)", "Aleatorio uniforme (Min - Max)"]
            )
            btn_generar = st.button("🚀 Generar sintéticos", type="primary", use_container_width=True)

        if btn_generar:
            if metodo_gen == "Muestreo con ruido gaussiano (Realista)":
                muestra_base = X_ref.sample(n=n_instancias, replace=True).reset_index(drop=True)
                nuevos_datos = muestra_base.copy()
                for col in columnas_features:
                    if pd.api.types.is_numeric_dtype(X_ref[col]):
                        std = X_ref[col].std()
                        ruido = np.random.normal(0, std * 0.15, size=n_instancias)
                        nuevos_datos[col] = nuevos_datos[col] + ruido
                        if pd.api.types.is_integer_dtype(X_ref[col]):
                            nuevos_datos[col] = nuevos_datos[col].round().astype(int)
            else:
                data_dict = {}
                for col in columnas_features:
                    if pd.api.types.is_integer_dtype(X_ref[col]):
                        data_dict[col] = np.random.randint(int(X_ref[col].min()), int(X_ref[col].max()) + 1, size=n_instancias)
                    else:
                        data_dict[col] = np.random.uniform(X_ref[col].min(), X_ref[col].max(), size=n_instancias)
                nuevos_datos = pd.DataFrame(data_dict)
            
            st.session_state["datos_sinteticos"] = nuevos_datos

        if "datos_sinteticos" in st.session_state:
            datos_a_predecir = st.session_state["datos_sinteticos"]

    if datos_a_predecir is not None:
        st.markdown("---")
        st.subheader("📋 Resultados de la Predicción")

        # Cálculo basado en el umbral personalizado
        probabilidades = modelo.predict_proba(datos_a_predecir)[:, 1]
        predicciones = (probabilidades >= umbral).astype(int)

        df_resultado = datos_a_predecir.copy()
        df_resultado.insert(0, "Probabilidad_Falla (%)", (probabilidades * 100).round(2))
        df_resultado.insert(0, "Prediccion", np.where(predicciones == 1, "⚠️ FALLA (1)", "✅ Normal (0)"))

        total_instancias = len(datos_a_predecir)
        total_fallas = int(predicciones.sum())
        total_normales = int(total_instancias - total_fallas)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Instancias Evaluadas", total_instancias)
        m2.metric("Equipos Normales (0)", total_normales)
        m3.metric("Fallas Detectadas (1)", total_fallas, delta=f"{(total_fallas/total_instancias)*100:.1f}% (Umbral: {int(umbral*100)}%)", delta_color="inverse")

        st.dataframe(df_resultado, use_container_width=True)

        csv_descarga = df_resultado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados completos (.csv)",
            data=csv_descarga,
            file_name="lote_predicho.csv",
            mime="text/csv"
        )

# =========================================================================
# PESTAÑA 2: PREDICCIÓN DE 1 SOLO CASO
# =========================================================================
with tab2:
    st.subheader("Evaluar un caso individual")
    st.write(f"Ajusta los valores de cada variable. (El umbral activo actual es del **{int(umbral*100)}%**).")

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

    with st.form("form_caso_individual"):
        inputs_usuario = {}
        cols_grid = st.columns(3)

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
                label="Probabilidad de Falla",
                value=f"{prob_caso * 100:.2f}%",
                delta=f"Umbral activo: {int(umbral*100)}%",
                delta_color="off"
            )

        with res_col2:
            if pred_caso == 1:
                st.error(f"### ⚠️ ALERTA DE FALLA DETECTADA (Clase 1)\nLa probabilidad ({prob_caso*100:.2f}%) supera o iguala el umbral establecido ({int(umbral*100)}%).")
            else:
                st.success(f"### ✅ OPERACIÓN NORMAL (Clase 0)\nLa probabilidad de falla ({prob_caso*100:.2f}%) está por debajo del umbral establecido ({int(umbral*100)}%).")
