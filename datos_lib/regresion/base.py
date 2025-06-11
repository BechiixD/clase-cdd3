import numpy as np
import pandas as pd
import statsmodels.api as sm

class Regresion:
    def __init__(self, X, y):
        # 1) Preparo X como DataFrame
        self.X = self._make_df(X)
        # 2) Chequeo que X e y tengan la misma longitud
        if len(self.X) != len(y):
            raise ValueError(f"X e y deben tener el mismo número de filas: {len(self.X)} vs {len(y)}")
        # 3) Guardo y como Series
        self.y = pd.Series(y).reset_index(drop=True)
        self.adjusted_model = None

    def _make_df(self, X):
        # a) Si ya es DataFrame, lo copio
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        # b) Si es dict, lo paso a DataFrame
        elif isinstance(X, dict):
            df = pd.DataFrame(X)
        # c) Si es lista/tupla o array, stackeo columnas
        else:
            arr = np.array(X)
            if arr.ndim == 1:
                arr = np.column_stack([arr])
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(1, arr.shape[1]+1)])
        # d) Quito cualquier columna 'const' residual
        if "const" in df.columns:
            df = df.drop(columns="const")
        # e) Convierto categóricas u object en dummies
        df = df.infer_objects() 
        cat = df.select_dtypes(include=["object", "category"]).columns
        if len(cat):
            df = pd.get_dummies(df, columns=cat, drop_first=True, dtype=int)
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
