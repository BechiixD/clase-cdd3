import numpy as np
import pandas as pd
import statsmodels.api as sm

class Regresion:
    def __init__(self, X, y):
        # X siempre DataFrame limpio y sin 'const'
        self.X = self._prepare_X(X)
        # y siempre Series alineada
        self.y = pd.Series(y).reset_index(drop=True)
        if len(self.X) != len(self.y):
            raise ValueError(f"X e y deben tener igual número de filas. got {len(self.X)} vs {len(self.y)}")
        self.adjusted_model = None

    def _prepare_X(self, X):
        # convierte X a DataFrame, quita constante y genera dummies
        if isinstance(X, pd.DataFrame):
            df = X.copy().reset_index(drop=True)
        elif isinstance(X, dict):
            df = pd.DataFrame(X)
        else:
            arr = np.array(X)
            # si es lista de arrays columnas, stack
            if arr.dtype == 'object' and isinstance(X, (list, tuple)):
                arr = np.column_stack(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
        # dummies para categóricas
        cats = df.select_dtypes(include=['object', 'category']).columns
        for col in cats:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = df.drop(columns=col).join(dummies)
        # quitar 'const'
        df = df.drop(columns=[c for c in df.columns if c.lower()=='const'], errors='ignore')
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
        Xc = sm.add_constant(self.X)
        model = self.modelo(Xc, self.y)
        results = model.fit()
        self.adjusted_model = results

        return {
            "params": results.params.tolist(),
            "bse": results.bse.tolist(),
            "t obs": results.tvalues.tolist(),
            "p values": results.pvalues.tolist(),
            "results": results
        }

    def predecir(self, X_new, alpha=0.05) -> dict:
        """
        Realiza predicciones con el modelo ajustado.

        Args:
            X_new (pd.DataFrame): Nuevos datos para predecir.

        Returns:
            dict: Predicciones con intervalo de confianza y predicción.
        """
        Xn = self._prepare_X(X_new)
        if len(Xn.columns) != len(self.X.columns):
            raise ValueError("X_new debe tener mismas columnas que X original después de dummies.")
        Xc = sm.add_constant(Xn, has_constant='add')
        pred = self.adjusted_model.get_prediction(Xc).summary_frame(alpha=alpha)
        summary = pred.summary_frame()

        return {
            "res": summary["mean"].iloc[0],
            "int_conf": [summary["mean_ci_lower"].iloc[0], summary["mean_ci_upper"].iloc[0]],
            "int_pred": [summary["obs_ci_lower"].iloc[0], summary["obs_ci_upper"].iloc[0]]
        }
