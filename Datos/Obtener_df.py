import numpy as np
import pandas as pd

def generar_datos_desbalanceados(n_muestras=1000):
    np.random.seed(42)
    
    # 1. Clase Mayoritaria: Operación Normal (~95% de los datos)
    n_normal = int(n_muestras * 0.9)
    temp_normal = np.random.normal(loc=70.0, scale=6.0, size=n_normal)
    vibracion_normal = np.random.normal(loc=2.0, scale=0.6, size=n_normal)
    presion_normal = np.random.normal(loc=10.0, scale=1.5, size=n_normal)
    horas_uso_normal = np.random.uniform(50, 800, size=n_normal)
    
    df_normal = pd.DataFrame({
        'temperatura': temp_normal,
        'vibracion': vibracion_normal,
        'presion': presion_normal,
        'horas_uso': horas_uso_normal,
        'falla': 0
    })
    
    n_falla = n_muestras - n_normal
    # Falla con zonas que se solapan con la operación normal (casos difíciles)
    temp_falla = np.random.normal(loc=78.0, scale=7.0, size=n_falla) 
    vibracion_falla = np.random.normal(loc=2.3, scale=0.9, size=n_falla) 
    presion_falla = np.random.normal(loc=12.0, scale=2.0, size=n_falla)
    horas_uso_falla = np.random.uniform(300, 800, size=n_falla)
    
    df_falla = pd.DataFrame({
        'temperatura': temp_falla,
        'vibracion': vibracion_falla,
        'presion': presion_falla,
        'horas_uso': horas_uso_falla,
        'falla': 1
    })
    
    df = pd.concat([df_normal, df_falla], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Generar y guardar dataset
df_industrial = generar_datos_desbalanceados()
df_industrial.to_csv('mantenimiento_predictivo_desbalanceado.csv', index=False)
print(df_industrial['falla'].value_counts())