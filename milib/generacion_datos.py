import numpy as np
from scipy.stats import norm, uniform

class GeneradorDatos:
  def __init__(self, n):
    self.n = n

  def generar_datos_dist_norm(self, mu=0, sigma=1):
    return np.random.normal(loc=mu, scale=sigma, size=self.n)

  def generar_datos_dist_bs(self, mu=0, sigma=1):
    u = np.random.uniform(size=(self.n,))
    y = np.zeros_like(u)
    ind = np.where(u > 0.5)[0]
    # y[ind] = np.random.normal(loc=mu, scale=sigma, size=len(ind))
    y[ind] = np.random.normal(loc=0, scale=1, size=len(ind))
    for j in range(5):
        ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
        y[ind] = np.random.normal(loc=(j/2 - 1), scale=(1/10), size=len(ind))
    return y

  def generar_datos_dist_uniforme(self, a=0, b=1):
    return np.random.uniform(low=a, high=b, size=self.n)

  def generar_datos_dist_exp(self, beta):
    return np.random.exponential(scale=beta, size=self.n)

  def pdf_norm(self, x, mu=0, sigma=1):
    return norm.pdf(x, loc=mu, scale=sigma)

  def pdf_uniform(self, x, a, b):
    return uniform.pdf(x, a=a, b=b)

  def pdf_bs(self, x):
    densidad = 0.5 * norm.pdf(x, loc=0, scale=1)
    for j in range(5):
      densidad += 0.1 * norm.pdf(x, loc=j/2 - 1, scale=1/10)
    return densidad