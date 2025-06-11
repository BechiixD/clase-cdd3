import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from datos_lib.regresion.base import Regresion
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score


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

    def dividir_datos(self, test_ratio: float = 0.2, seed: int = 1):
        '''
        Divide los datos en conjuntos de entrenamiento y prueba.
        Args:
            test_ratio (float): Proporción de datos para el conjunto de prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.
        
        Returns:
            tuple: Índices de entrenamiento y prueba.
        '''
        n = len(self.y)
        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)
        cut = int(n*(1-test_ratio))
        train, test = indices[:cut], indices[cut:]
        return train, test

    def calcular_matriz_confusion(self, p: float, test_ratio: float = 0.2, seed: int = 1) -> dict:
        """
        Calcula y muestra la matriz de confusión para la regresión logística.

        Args:
            p (float): Umbral de probabilidad para clasificación.
            test_ratio (float): Proporción de datos para prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.

        Returns:
            None
        """
        train, test = self.dividir_datos(test_ratio, seed)
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
    
    def corte_optimo(self, n: int = 100, test_ratio: float = 0.2, seed: int = 1) -> dict:
        '''
        Calcula el corte óptimo para maximizar la sensibilidad y especificidad.
        Args:
            n (int): Número de puntos a evaluar. Por defecto 100.
            test_ratio (float): Proporción de datos para prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.
        Returns:
            dict: Diccionario con sensibilidad, especificidad y corte óptimo.
        '''
        train, test = self.dividir_datos(test_ratio, seed)

        X_tr, y_tr = self.X.iloc[train], self.y.iloc[train]
        X_te, y_te = self.X.iloc[test], self.y.iloc[test]
        
        p_values = np.linspace(0, 1, n)

        # Ajustar el modelo con los datos de entrenamiento
        Xc_tr = sm.add_constant(X_tr, has_constant='add')
        temp_model = self.modelo(Xc_tr, y_tr).fit()
        # Inicializar listas para almacenar sensibilidad y especificidad
        sensibilidad = []
        especificidad = []
        max_p = 0
        max_j = 0
        for p in p_values:
            Xc_te = sm.add_constant(X_te, has_constant='add')
            y_pred = (temp_model.predict(Xc_te) >= p).astype(int)
            
            si_si = sum((y_te == 'Yes') & (y_pred == 1))
            si_no = sum((y_te == 'Yes') & (y_pred == 0))
            no_si = sum((y_te == 'No') & (y_pred == 1))
            no_no = sum((y_te == 'No') & (y_pred == 0))

            sens = si_si / (si_si + si_no) if (si_si + si_no) > 0 else 0
            espec = no_no / (no_no + no_si) if (no_no + no_si) > 0 else 0

            sensibilidad.append(sens)
            especificidad.append(espec)

            current = sens + espec - 1
            if current > max_j:
                max_j = current
                max_p = p
                max_sensibilidad = sens
                max_especificidad = espec

        return {
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'corte_optimo': max_p,
            'sensibilidad_optima': max_sensibilidad,
            'especificidad_optima': max_especificidad
        }
        
    def graficar_corte_optimo(self, n: int = 100, test_ratio: float = 0.2, seed: int = 1) -> None:
        """
        Grafica la sensibilidad y especificidad en función del umbral de probabilidad.

        Args:
            n (int): Número de puntos a evaluar. Por defecto 100.
            test_ratio (float): Proporción de datos para prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.

        Returns:
            None
        """
        resultados = self.corte_optimo(n, test_ratio, seed)
        plt.plot(np.linspace(0, 1, n), resultados['sensibilidad'], label='Sensibilidad')
        plt.plot(np.linspace(0, 1, n), resultados['especificidad'], label='Especificidad')
        plt.axvline(x=resultados['corte_optimo'], color='red', linestyle='--', label='Corte óptimo')
        plt.xlabel('Umbral de probabilidad')
        plt.ylabel('Tasa')
        plt.title('Sensibilidad y Especificidad vs Umbral de Probabilidad')
        plt.legend()
        plt.grid(True)
        plt.show()

    def curva_ROC(self, test_ratio: float = 0.2, seed: int = 1) -> dict:
        """
        Grafica la curva ROC, calcula el área bajo la curva (AUC) y evalúa el clasificador.

        Args:
            test_ratio (float): Proporción de datos para prueba. Por defecto 0.2.
            seed (int): Semilla para reproducibilidad. Por defecto 1.

        Returns:
            dict: Diccionario con fpr (false positives ratio), tpr (true positives ratio), thresholds (), auc y evaluación.
        """

        train, test = self.dividir_datos(test_ratio, seed)
        X_tr, y_tr = self.X.iloc[train], self.y.iloc[train]
        X_te, y_te = self.X.iloc[test], self.y.iloc[test]

        Xc_tr = sm.add_constant(X_tr, has_constant='add')
        model = self.modelo(Xc_tr, y_tr).fit()
        Xc_te = sm.add_constant(X_te, has_constant='add')
        y_prob = model.predict(Xc_te)

        fpr, tpr, umbrales = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)

        # Evaluación del clasificador según el AUC (tabla teórica)
        if auc < 0.6:
            evaluacion = "Malo"
        elif auc < 0.7:
            evaluacion = "Regular"
        elif auc < 0.8:
            evaluacion = "Aceptable"
        elif auc < 0.9:
            evaluacion = "Bueno"
        else:
            evaluacion = "Excelente"

        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Azar')
        plt.xlabel('Tasa de Falsos Positivos (1-Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"AUC: {auc:.3f} - Evaluación: {evaluacion}")

        return {
            'fpr': fpr,
            'tpr': tpr,
            'umbrales': umbrales,
            'auc': auc,
            'evaluacion': evaluacion
        }