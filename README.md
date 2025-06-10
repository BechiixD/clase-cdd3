# 📊 datos-lib

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
- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Instalalos con:
```bash
pip install -r requirements.txt