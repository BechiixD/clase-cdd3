# 📊 datos_lib

Una libreria de Python simple y modular para realizar analisis de datos clasicos.
Incluye:
- Analisis descriptivo
- Generacion de histogramas y densidades
- Regresion lineal
- Regresion logistica

---

## Instalacion
Instalar directo desde GitHub:
```bash
!pip install git+https://github.com/BechiixD/datos_lib
```

---

## Uso basico
```python
from datos_lib import AnalisisDescriptivo, RegresionLineal

# --- Análisis Descriptivo ---
data = [1, 2, 3, 4, 5, 6]
ad = AnalisisDescriptivo(data)

# Histograma
bins, densidad = ad.genera_histograma(bins=3)
print("Bins:", bins)
print("Densidad:", densidad)

# --- Regresión Lineal ---
X = [[1], [2], [3], [4]]
y = [2, 3, 4, 5]

modelo = RegresionLineal(X, y)
modelo.ajustar_modelo()

predicciones = modelo.predecir([[5]])
print("Predicción para x=5:", predicciones)
```

---
## Requisitos
- numpy>=1.19.0
- pandas>=1.1.0
- scipy>=1.5.0
- matplotlib>=3.3.0
- statsmodels>=0.12.0
- sklearn>=0.24.0

Instalalos con:
```bash
pip install -r requirements.txt