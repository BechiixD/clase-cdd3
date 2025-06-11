import numpy as np
import pandas as pd
import statsmodels.api

class Regresion:
    def __init__(self, X, y):
        # 1) forzar X limpio y con dummies si hay categorías
        self.X = self._prepare_X(X)
        # 2) y siempre Series
        self.y = pd.Series(y).reset_index(drop=True)
        self.adjusted_model = None

    def _prepare_X(self, X):
        # convierte X a DataFrame, quita constant, genera dummies y resetea índices
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame(X)
        else:
            arr = np.array(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])

        # detectar y convertir categóricas a dummies
        for col in df.select_dtypes(include=['object', 'category']).columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = df.drop(columns=col).join(dummies)

        # quitar si ya existe constante
        if 'const' in df.columns:
            df = df.drop(columns='const')

        return df.reset_index(drop=True)
    
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
