import numpy as np
import pandas as pd
import statsmodels.api

class Regresion:
    def __init__(self, X, y):
        """
        Inicializa la clase con los datos predictivos y la variable respuesta.

        Args:
            X: Datos predictivos (puede ser lista, array, dict, etc.).
            y: Variable respuesta (array-like).

        Raises:
            ValueError: Si X no se puede convertir a DataFrame o está vacío.
        """
        try:
            self.X = pd.DataFrame(X)
            if self.X.empty or self.X.shape[1] == 0:
                raise ValueError("El DataFrame resultante está vacío o no tiene columnas.")
        except Exception as e:
            raise ValueError(f"No se pudo convertir X a DataFrame: {str(e)}. Asegúrate de que X sea un diccionario, lista, array de numpy, etc.")

        self.y = y
        self.adjusted_model = None

    def modelo(self, X: pd.DataFrame, y: np.ndarray) -> 'statsmodels.base.model.Model':
        """
        Este método se sobreescribe en subclases
        Guarda el modelo elegido mediante herencia
        """
        raise NotImplementedError

    def ajustar_modelo(self) -> dict:
        """
        Ajusta el modelo usando los datos de X y y.

        Returns:
            dict: Diccionario con parámetros, errores estándar, valores t y p-valores.

        Ejemplo:
            >>> reg = RegresionLineal(X, y)
            >>> reg.ajustar_modelo()
            # Ajusta el modelo y devuelve métricas.
        """
        X = statsmodels.api.add_constant(self.X)
        model = self.modelo(X, self.y)
        results = model.fit()
        self.adjusted_model = results

        return {
            "params": results.params.tolist(),
            "bse": results.bse.tolist(),
            "t obs": results.tvalues.tolist(),
            "p values": results.pvalues.tolist(),
            "results": results
        }

    def predecir(self, X_new) -> dict:
        """
        Realiza predicciones con el modelo ajustado.

        Args:
            X_new (pd.DataFrame): Nuevos datos para predecir.

        Returns:
            dict: Predicciones con intervalo de confianza y predicción.
        """
        if self.adjusted_model is None:
            self.ajustar_modelo()
        X_new = pd.DataFrame(X_new)
        X_new = statsmodels.api.add_constant(X_new)
        pred = self.adjusted_model.get_prediction(X_new)
        summary = pred.summary_frame()

        return {
            "res": summary["mean"].iloc[0],
            "int_conf": [summary["mean_ci_lower"].iloc[0], summary["mean_ci_upper"].iloc[0]],
            "int_pred": [summary["obs_ci_lower"].iloc[0], summary["obs_ci_upper"].iloc[0]]
        }
