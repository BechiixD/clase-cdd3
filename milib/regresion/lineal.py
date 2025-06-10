import numpy as np
import pandas as pd
import statsmodels.api
from statsmodels.regression.linear_model import OLS
from milib.regresion.base import Regresion
import matplotlib.pyplot as plt

class RegresionLineal(Regresion):
    def modelo(self, X: pd.DataFrame, y: np.ndarray) -> OLS:
        """
        Devuelve el modelo correspondiente a la regresión lineal.
        Args:
            X (array): El array de variables predictoras.
            y (array): El array de la variable respuesta.

        Returns:
            statsmodels.api.OLS: El modelo de regresión lineal ajustado.

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.modelo(X,y)
            # Esto graficará la columna 1 de X contra y con la recta de predicción.
        """
        X = pd.DataFrame(X) if X is not None else statsmodels.api.add_constant(self.X)
        y = np.array(y) if y is not None else self.y
        return OLS(y, X)

    def graficar_dispersion(self, column: int = 0) -> None:
        """
        Grafica un diagrama de dispersión de la columna especificada de X contra y,
        y superpone la recta de predicción del modelo de regresión lineal para esa variable,
        manteniendo las demás variables en sus valores medios.

        Args:
            column (int, optional): Índice de la columna de X a graficar. Por defecto es 0.

        Returns:
            None

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.graficar_dispersion(column=1)
            # Esto graficará la columna 1 de X contra y con la recta de predicción.
        """
        if self.adjusted_model is None:
            self.ajustar_modelo()

        col_name = self.X.columns[column]
        x_data = self.X[col_name]
        x_vals = np.linspace(x_data.min(), x_data.max(), 100)  # 100 puntos para una recta suave

        # Calcular y_pred con la fórmula del efecto parcial
        y_bar = np.mean(self.y)
        x1_bar = np.mean(self.X[col_name])
        beta1 = self.adjusted_model.params[col_name]
        y_pred = y_bar + beta1 * (x_vals - x1_bar)

        plt.scatter(x_data, self.y)
        plt.plot(x_vals, y_pred, color='red')
        plt.xlabel(col_name)
        plt.ylabel("y")
        plt.title("Regresión Lineal")
        plt.show()

    def calcular_coeficiente_correlacion(self) -> float:
        """
        Calcula el coeficiente de correlacion entre X y y.

        Returns:
            Float: Coeficiente de correlación de Pearson.

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.calcular_coeficiente_correlacion()
            # Esto devuelve el coeficiente de correlacion.
        """

        return np.corrcoef(self.X.iloc[:, 0], self.y)[0, 1]

    def ajustar_modelo(self):
        """
        Llama a la funcion ajustar_modelo en la clase Regresion

        Returns:
            None

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.ajustar_modelo(column=1)
        """
        return super().ajustar_modelo()

    def analisis_residuales(self) -> float:
        """
        Grafica un QQPlot de statsmodels.api con los residuos y la recta de predicción.
        Y grafica los residuos contra los valores predichos.

        Returns:
            Float: Valores residuales y ajustados.

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.analisis_residuales()
            # Esto grafica el QQPlot y residuos y devuelve los valores.
        """

        if self.adjusted_model is None:
            self.ajustar_modelo()
        res = self.adjusted_model
        fitted, resid = res.fittedvalues, res.resid
        # QQ-plot
        statsmodels.api.qqplot(resid, line='45')
        plt.title("QQ-Plot de residuos"); plt.show()
        # residuales vs ajustados
        plt.scatter(fitted, resid)
        plt.axhline(0, linestyle='--')
        plt.xlabel("Fitted"); plt.ylabel("Residuales")
        plt.title("Residuales vs Ajustados"); plt.show()
        return resid, fitted

    def calcular_coeficiente_determinacion(self) -> float:
        """
        Si el modelo no esta ajustado lo ajusta.
        Calcula el coeficiente de determinación.

        Returns:
            float: Coeficiente de determinación

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.calcular_coeficiente_determinacion()
        """

        if self.adjusted_model is None:
            self.ajustar_modelo()
        return self.adjusted_model.rsquared

    def calcular_R2_ajustado(self) -> float:
        """
        Si el modelo no esta ajustado lo ajusta.
        Calcula el coeficiente de determinación ajustado.

        Returns:
            float: Coeficiente de determinación ajustado

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.calcular_R2_ajustado()
        """

        if self.adjusted_model is None:
            self.ajustar_modelo()
        return self.adjusted_model.rsquared_adj
