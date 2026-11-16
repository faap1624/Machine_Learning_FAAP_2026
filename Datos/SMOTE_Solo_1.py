from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# 1. Carga de datos
df = pd.read_csv('mantenimiento_predictivo_desbalanceado.csv')

X = df.drop(columns=["falla"])
y = df["falla"] 

# 2. División de datos (¡Fundamental hacerlo antes de aplicar cualquier balanceo!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== ANTES DE SMOTE (Conjunto de Entrenamiento) ===")
print(f"Total de registros: {len(X_train)}")
print("Distribución de clases:")
print(y_train.value_counts())

# 3. Aplicar SMOTE globalmente solo al conjunto de entrenamiento
smote = SMOTE(sampling_strategy=0.025, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n=== DESPUÉS DE SMOTE (Conjunto de Entrenamiento) ===")
print(f"Total de registros: {len(X_train_resampled)}")
print("Distribución de clases:")
print(Counter(y_train_resampled))

# ------------------------------------------------------------------------------------------#
# 4. Evaluación de Modelos
# ------------------------------------------------------------------------------------------#
resultados = {}

# A. Modelos entrenados directamente con el X_train_resampled global
# 1. XGBoost
model_xgb = xgb.XGBClassifier(random_state=42)
model_xgb.fit(X_train_resampled, y_train_resampled)
y_pred = model_xgb.predict(X_test)
y_prob = model_xgb.predict_proba(X_test)[:, 1]

resultados['XGBoost (con SMOTE previo)'] = {
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (Acierto Falla) (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}

# 2. Random Forest (con los datos ya balanceados)
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_resampled, y_train_resampled)
y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)[:, 1]

resultados['Random Forest (con SMOTE previo)'] = {
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (Acierto Falla) (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}

# B. Modelos que usan los datos originales (X_train, y_train) 
# 3. Pipeline con SMOTE + Random Forest 
pipeline_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_smote.fit(X_train, y_train)
y_pred = pipeline_smote.predict(X_test)
y_prob = pipeline_smote.predict_proba(X_test)[:, 1]

resultados['Pipeline (SMOTE + RF)'] = {
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (Acierto Falla) (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}

# 4. Balanced Random Forest
model_brf = BalancedRandomForestClassifier(random_state=42)
model_brf.fit(X_train, y_train)
y_pred = model_brf.predict(X_test)
y_prob = model_brf.predict_proba(X_test)[:, 1]

resultados['Balanced Random Forest'] = {
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (Acierto Falla) (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}

# 5. Visualización final de resultados
df_res = pd.DataFrame(resultados).T
print("\n=== COMPARATIVA DE MODELOS ===")
print(df_res.round(2))

# ------------------------------------------------------------------------------------------#
# 6. Guardado de Modelos Entrenados y Resultados
# ------------------------------------------------------------------------------------------#
carpeta_salida = 'modelos_guardados'
os.makedirs(carpeta_salida, exist_ok=True)

# Guardar cada modelo entrenado en formato .joblib
joblib.dump(model_xgb, os.path.join(carpeta_salida, 'modelo_xgboost_smote.joblib'))
joblib.dump(model_rf, os.path.join(carpeta_salida, 'modelo_random_forest_smote.joblib'))
joblib.dump(pipeline_smote, os.path.join(carpeta_salida, 'pipeline_smote_rf.joblib'))
joblib.dump(model_brf, os.path.join(carpeta_salida, 'modelo_balanced_rf.joblib'))

# Guardar la tabla comparativa de métricas en CSV y Excel
df_res.round(2).to_csv(os.path.join(carpeta_salida, 'comparativa_modelos.csv'))

print(f"\n[OK] Todos los modelos y la tabla de resultados se han guardado en la carpeta: '{carpeta_salida}/'")