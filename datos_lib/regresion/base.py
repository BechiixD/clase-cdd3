import numpy as np
import pandas as pd
import statsmodels.api as sm

class Regresion:
    def __init__(self, X, y):
        '''
        Inicializa la clase de regresión con los datos X e y y los guarda.
        Args:
            X (pd.DataFrame, list, tuple, np.ndarray): Datos de ntrada (variables predictoras).
                Deben tener un formato de matriz 2D (n filas (datos), m columnas (columnas)).
            y (pd.Series, np.ndarray): Variable objetivo (respuesta).
        '''
        self.X = self._make_df(X)
        self.y = pd.Series(y).reset_index(drop=True)

        if len(self.X) != len(self.y):
            raise ValueError(f"X e y deben tener el mismo número de filas: {len(self.X)} vs {len(self.y)}")

        self.adjusted_model = None

    def _make_df(self, X):
        # Si es DataFrame, lo uso directo
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, pd.Series):
            X = X.to_frame()
        # Si es lista/array de 2D, lo paso a DataFrame
        elif isinstance(X, (list, tuple, np.ndarray)):
            arr = np.array(X)
            if arr.ndim != 2:
                raise ValueError("X debe ser una matriz 2D (n filas, m columnas)")
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(1, arr.shape[1]+1)])
        else:
            raise TypeError("X debe ser un DataFrame, lista de listas o array 2D")

        # Elimino 'const' si está
        if "const" in df.columns:
            df = df.drop(columns="const")

        # Convierte variables categóricas en dummies
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
        Xn = self._make_df(X_new)
        if len(Xn.columns) != len(self.X.columns):
            raise ValueError("X_new debe tener mismas columnas que X original después de dummies.")
        Xc = sm.add_constant(Xn, has_constant='add')
        summary = self.adjusted_model.get_prediction(Xc).summary_frame(alpha=alpha)

        return {
            "res": summary["mean"].iloc[0],
            "int_conf": [summary["mean_ci_lower"].iloc[0], summary["mean_ci_upper"].iloc[0]],
            "int_pred": [summary["obs_ci_lower"].iloc[0], summary["obs_ci_upper"].iloc[0]]
        }
