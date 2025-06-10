import numpy as np
import pandas as pd
import statsmodels.api
from statsmodels.discrete.discrete_model import Logit
from datos_lib.regresion.base import Regresion
import matplotlib.pyplot as plt
import random


class RegresionLogistica(Regresion):
    def modelo(self, X: pd.DataFrame, y: np.ndarray) -> Logit:
        """
        Devuelve el modelo de regresión logística de statsmodels.

        Args:
            X (pd.DataFrame): Matriz de variables predictoras (incluye constante).
            y (np.ndarray): Array de la variable respuesta (0/1 para logística).

        Returns:
            Logit: Modelo de regresión logística de statsmodels listo para ajustar.

        Ejemplo:
            >>> reg = RegresionLogistica(X, y)
            >>> model = reg.modelo(pd.DataFrame(X), np.array(y))
            >>> results = model.fit()
        """
        X = pd.DataFrame(X) if X is not None else statsmodels.api.add_constant(self.X)
        y = np.array(y) if y is not None else self.y
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y debe contener solo 0 y 1 para regresión logística.")
        return Logit(y, X)
    
    def graficar_dispersion(self, column=0) -> None:
        if self.adjusted_model is None:
            self.ajustar_modelo()

        x_data = self.X.iloc[:, column]
        col_name = self.X.columns[column]

        x_vals = np.linspace(x_data.min(), x_data.max(), 300)
        X_pred = pd.DataFrame({col_name: x_vals})
        X_pred = statsmodels.api.add_constant(X_pred)

        # Predecir probabilidades
        y_vals = self.adjusted_model.predict(X_pred)

        # Ordenar x_vals y y_vals en conjunto
        orden = np.argsort(x_vals)
        x_vals_ordenado = x_vals[orden]
        y_vals_ordenado = y_vals[orden]

        # Graficar
        plt.scatter(x_data, self.y, color='blue', label='Datos reales', alpha=0.6)
        plt.plot(x_vals_ordenado, y_vals_ordenado, color='red', label='Curva logística')
        plt.xlabel(col_name)
        plt.ylabel("Probabilidad de y = 1")
        plt.title("Regresión Logística")
        plt.legend()
        plt.grid(True)
        plt.show()

    def ajustar_modelo(self):
        return super().ajustar_modelo()

    def calcular_matriz_confusion(self, p: float, test_ratio: float = 0.2, seed: int = 1):
        """
        Calcula y muestra la matriz de confusión para la regresión logística.

        Args:
            p (float): Umbral de probabilidad para clasificación.
            test_ratio (float): Proporción de datos para prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.

        Returns:
            None
        """
        n = self.X.shape[0]
        n_train = int(n * (1 - test_ratio))
        random.seed(seed)
        train_indices = random.sample(range(n), n_train)
        test_indices = list(set(range(n)) - set(train_indices))

        X_train = self.X.iloc[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X.iloc[test_indices]
        y_test = self.y[test_indices]

        # Ajustar modelo con datos de entrenamiento
        X_train_const = statsmodels.api.add_constant(X_train)
        model = self.modelo(X_train_const, y_train)
        results = model.fit()

        # Predecir con datos de prueba
        X_test_const = statsmodels.api.add_constant(X_test)
        y_pred_prob = results.predict(X_test_const)
        y_pred = (y_pred_prob >= p).astype(int)

        # Calcular matriz de confusión (asumiendo y es 0/1)
        si_si = sum((y_test == 1) & (y_pred == 1))
        si_no = sum((y_test == 1) & (y_pred == 0))
        no_si = sum((y_test == 0) & (y_pred == 1))
        no_no = sum((y_test == 0) & (y_pred == 0))

        sensibilidad = si_si / (si_si + si_no)
        falso_negativo = si_no / (si_si + si_no)
        falso_positivo = no_si / (no_si + no_no)
        especificidad = no_no / (no_si + no_no)
        error_clasificacion = (si_no + no_si) / n

        return {
            "sensibilidad": sensibilidad,
            "falso_negativo": falso_negativo,
            "falso_positivo": falso_positivo,
            "especificidad": especificidad,
            "error_clasificacion": error_clasificacion
        }
    