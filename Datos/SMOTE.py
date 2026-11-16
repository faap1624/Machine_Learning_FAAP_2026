from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

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

# 3. Calcular la proporción original automáticamente
train_counts = y_train.value_counts()
original_ratio = train_counts.min() / train_counts.max()
print(f"\nProporción original detectada: {original_ratio:.5f} (~{original_ratio*100:.2f}%)")

# Generar lista de estrategias multiplicándose por 1.25 progresivamente
strategies = []
curr = original_ratio * 1.1  
for _ in range(100):  
    if curr >= 1.0:
        strategies.append(1.0)
        break
    strategies.append(curr)
    curr += 0.01

if 1.0 not in strategies:
    strategies.append(1.0)

# Lista para acumular los resultados globales
todos_resultados = []

# ------------------------------------------------------------------------------------------#
# 3.1 Evaluamos primero SIN SMOTE (Datos originales puros como línea base)
# ------------------------------------------------------------------------------------------#
print(f"\n" + "="*60)
print(f"Evaluando SIN SMOTE (Distribución Original: {original_ratio*100:.2f}%)")
print("="*60)

resultados_baseline = []
for modelo_nombre, clf in [('XGBoost', xgb.XGBClassifier(random_state=42)), ('Random Forest', RandomForestClassifier(random_state=42))]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    res_dict = {
        'Modelo': f'{modelo_nombre} (Sin SMOTE)',
        'Sampling Strategy': round(original_ratio, 5),
        'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
        'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
        'Recall (%)': recall_score(y_test, y_pred) * 100,
        'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
        'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
    }
    resultados_baseline.append(res_dict)
    todos_resultados.append(res_dict)

# Imprimir tabla específica de la línea base
print(pd.DataFrame(resultados_baseline).to_string(index=False))

# ------------------------------------------------------------------------------------------#
# 4. Bucle de Evaluación con Diferentes 'sampling_strategy' incrementados
# ------------------------------------------------------------------------------------------#
for strategy in strategies:
    print(f"\n" + "="*60)
    print(f"Evaluando con sampling_strategy = {strategy:.5f} (~{strategy*100:.2f}%)")
    print("="*60)
    
    # Aplicar SMOTE con la estrategia actual
    smote = SMOTE(sampling_strategy=strategy, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Total registros tras SMOTE: {len(X_train_resampled)} | Distribución: {Counter(y_train_resampled)}")

    resultados_iteracion = []

    # A. XGBoost con SMOTE previo
    model_xgb = xgb.XGBClassifier(random_state=42)
    model_xgb.fit(X_train_resampled, y_train_resampled)
    y_pred = model_xgb.predict(X_test)
    y_prob = model_xgb.predict_proba(X_test)[:, 1]

    res_xgb = {
        'Modelo': 'XGBoost (SMOTE previo)',
        'Sampling Strategy': round(strategy, 5),
        'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
        'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
        'Recall (%)': recall_score(y_test, y_pred) * 100,
        'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
        'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
    }
    resultados_iteracion.append(res_xgb)
    todos_resultados.append(res_xgb)

    # B. Random Forest con SMOTE previo
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train_resampled, y_train_resampled)
    y_pred = model_rf.predict(X_test)
    y_prob = model_rf.predict_proba(X_test)[:, 1]

    res_rf = {
        'Modelo': 'Random Forest (SMOTE previo)',
        'Sampling Strategy': round(strategy, 5),
        'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
        'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
        'Recall (%)': recall_score(y_test, y_pred) * 100,
        'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
        'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
    }
    resultados_iteracion.append(res_rf)
    todos_resultados.append(res_rf)

    # Imprimir la tabla correspondiente a esta distribución específica de SMOTE
    print(f"\n--- TABLA COMPARATIVA (Estrategia: {strategy*100:.2f}%) ---")
    print(pd.DataFrame(resultados_iteracion).to_string(index=False))

# ------------------------------------------------------------------------------------------#
# 5. Modelos de Referencia
# ------------------------------------------------------------------------------------------#
print(f"\n" + "="*60)
print("Evaluando Modelos de Referencia (Pipeline & Balanced RF)")
print("="*60)

resultados_referencia = []

pipeline_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_smote.fit(X_train, y_train)
y_pred = pipeline_smote.predict(X_test)
y_prob = pipeline_smote.predict_proba(X_test)[:, 1]

res_pipe = {
    'Modelo': 'Pipeline (SMOTE auto + RF)',
    'Sampling Strategy': 'Auto (1.0)',
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}
resultados_referencia.append(res_pipe)
todos_resultados.append(res_pipe)

model_brf = BalancedRandomForestClassifier(random_state=42)
model_brf.fit(X_train, y_train)
y_pred = model_brf.predict(X_test)
y_prob = model_brf.predict_proba(X_test)[:, 1]

res_brf = {
    'Modelo': 'Balanced Random Forest',
    'Sampling Strategy': 'N/A (Interno)',
    'Accuracy Global (%)': accuracy_score(y_test, y_pred) * 100,
    'Precision Falla (%)': precision_score(y_test, y_pred, zero_division=0) * 100,
    'Recall (%)': recall_score(y_test, y_pred) * 100,
    'F1-Score (%)': f1_score(y_test, y_pred, zero_division=0) * 100,
    'PR-AUC (%)': average_precision_score(y_test, y_prob) * 100
}
resultados_referencia.append(res_brf)
todos_resultados.append(res_brf)

print(pd.DataFrame(resultados_referencia).to_string(index=False))

# 6. Visualización final en DataFrame global
df_res = pd.DataFrame(todos_resultados)
print("\n" + "="*60)
print("=== TABLA COMPARATIVA GENERAL ACUMULADA ===")
print("="*60)
print(df_res.to_string(index=False))
df_res.to_csv("TABLA COMPARATIVA GENERAL ACUMULADA.csv",index=False)

# ------------------------------------------------------------------------------------------#
# 7. Gráfica Final de Resultados
# ------------------------------------------------------------------------------------------#
df_plot = df_res[df_res['Sampling Strategy'].apply(lambda x: isinstance(x, (int, float)))].copy()

plt.figure(figsize=(10, 6))

config_estilos = {
    'XGBoost (SMOTE previo)': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
    'Random Forest (SMOTE previo)': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'}
}

for modelo, estilos in config_estilos.items():
    datos_modelo = df_plot[df_plot['Modelo'] == modelo]
    
    x = datos_modelo['Sampling Strategy'] * 100
    y = datos_modelo['F1-Score (%)']
    
    plt.plot(x, y, 
             label=modelo, 
             color=estilos['color'], 
             linestyle=estilos['linestyle'], 
             marker=estilos['marker'],
             linewidth=2, 
             markersize=6)

plt.title('Efecto del Sampling Strategy de SMOTE en el F1-Score (%) por Modelo', fontsize=13, fontweight='bold')
plt.xlabel('Sampling Strategy (%) [Proporción de clase minoritaria vs mayoritaria]', fontsize=11)
plt.ylabel('F1-Score (%)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(title='Modelos', fontsize=10)
plt.tight_layout()
plt.show()