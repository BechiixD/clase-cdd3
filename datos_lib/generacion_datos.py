import numpy as np
from scipy.stats import norm, uniform
from typing import Union

class GeneradorDatos:
  def __init__(self, n: int) -> None:
    '''
    Inicializa la clase con el número de datos a generar.
    Args:
      n (int): Número de datos a generar.
    '''
    if not isinstance(n, int) or n <= 0:
      raise ValueError("n debe ser un entero positivo.")
    self.n = n

  def generar_datos_dist_norm(self, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    '''
    Genera datos de una distribución normal.
    Args:
      mu (float): Media de la distribución normal. Por defecto 0.
      sigma (float): Desviación estándar de la distribución normal. Por defecto 1.
    Returns:
      np.ndarray: Array de datos generados de la distribución normal.
    '''
    return np.random.normal(loc=mu, scale=sigma, size=self.n)

  def generar_datos_dist_bs(self, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    '''
    Genera datos de una distribución bimodal.
    Args:
      mu (float): Media de la distribución normal. Por defecto 0.
      sigma (float): Desviación estándar de la distribución normal. Por defecto 1.
    Returns:
      np.ndarray: Array de datos generados de la distribución bimodal.
    '''
    u = np.random.uniform(size=(self.n,))
    y = np.zeros_like(u)
    ind = np.where(u > 0.5)[0]
    y[ind] = np.random.normal(loc=0, scale=1, size=len(ind))
    for j in range(5):
        ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
        y[ind] = np.random.normal(loc=(j/2 - 1), scale=(1/10), size=len(ind))
    return y

  def generar_datos_dist_uniform(self, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return np.random.uniform(low=a, high=b, size=self.n)

  def generar_datos_dist_exp(self, beta: float) -> np.ndarray:
    return np.random.exponential(scale=beta, size=self.n)

  def pdf_norm(self, x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return norm.pdf(x, loc=mu, scale=sigma)

  def pdf_uniform(self, x: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return uniform.pdf(x, loc=a, scale=b-a)

  def pdf_bs(self, x: np.ndarray) -> np.ndarray:
    '''
    Densidad de probabilidad de una distribución bimodal.
    '''
    densidad = 0.5 * norm.pdf(x, loc=0, scale=1)
    for j in range(5):
      densidad += 0.1 * norm.pdf(x, loc=j/2 - 1, scale=1/10)
    return densidad