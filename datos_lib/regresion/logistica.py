import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from datos_lib.regresion.base import Regresion
import matplotlib.pyplot as plt
import random


class RegresionLogistica(Regresion):
    def modelo(self, Xc: pd.DataFrame, y: np.ndarray) -> Logit:
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
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y debe contener solo 0 y 1 para regresión logística.")
        return Logit(y, Xc)
    
    def graficar_dispersion(self, column=0) -> None:
        if self.adjusted_model is None:
            self.ajustar_modelo()

        col = self.X.columns[column]
        x = self.X[col]
        plt.scatter(x, self.y, color='blue', alpha=0.6, label='Datos')
        xs = np.linspace(x.min(), x.max(), 300)
        # preparar con medias en otras variables
        X_pred = pd.DataFrame({c: np.repeat(self.X[c].mean(), len(xs)) for c in self.X.columns})
        X_pred[col] = xs
        Xc = sm.add_constant(X_pred, has_constant='add')
        ys = self.adjusted_model.predict(Xc)
        idx = np.argsort(xs)
        plt.plot(xs[idx], ys[idx], color='red', label='Curva logística')
        plt.xlabel(col); plt.ylabel('Probabilidad y=1'); plt.title('Regresión Logística')
        plt.legend(); plt.grid(True); plt.show()

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
        n = len(self.y)
        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)
        cut = int(n*(1-test_ratio))
        train, test = indices[:cut], indices[cut:]
        X_tr, y_tr = self.X.iloc[train], self.y.iloc[train]
        X_te, y_te = self.X.iloc[test], self.y.iloc[test]
        Xc_tr = sm.add_constant(X_tr, has_constant='add')
        res = self.modelo(Xc_tr, y_tr).fit()
        Xc_te = sm.add_constant(X_te, has_constant='add')
        y_prob = res.predict(Xc_te)
        y_pred = (y_prob>=p).astype(int)
        tp = sum((y_te==1)&(y_pred==1)); tn = sum((y_te==0)&(y_pred==0))
        fp = sum((y_te==0)&(y_pred==1)); fn = sum((y_te==1)&(y_pred==0))
        return {'sens':tp/(tp+fn), 'spec':tn/(tn+fp), 'error':(fp+fn)/len(y_te)}
    