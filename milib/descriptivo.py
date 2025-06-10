import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class AnalisisDescriptivo:
  def __init__(self, data):
    self.data = data

  def genera_histograma(self, h):
    self.h = h
    max_val = np.max(self.data)
    min_val = np.min(self.data)

    bins = np.arange(min_val, max_val + self.h, self.h)
    frecuencias = np.zeros(len(bins) - 1)

    for i in range(1, len(bins)):
      frecuencias[i-1] = np.sum((self.data >= bins[i-1]) & (self.data < bins[i]))

    densidad = frecuencias / (len(self.data) * self.h)
    # frecuencia en escala de densidad
    return bins, densidad

  def evalua_histograma(self, h, x):
    self.h = h
    bins, densidad = self.genera_histograma(h)
    densidad_x = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
      # Encontrar en qué intervalo cae el valor x[i]
      for j in range(1, len(bins)):
        if bins[j-1] <= x[i] < bins[j]:
          densidad_x[i] = densidad[j-1]
          break

    return densidad_x

  def kernel_gaussiano(self, x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp((-1/2) * x**2)

  def kernel_uniforme(self, x):
    return np.where(((x > -1/2) & (x < 1/2)), 1, 0)

  def kernel_cuadratico(self, x):
    return 3 / 4 * (1 - x**2) * ((x >= -1) & (x <= 1))

  def kernel_triangular(self, x):
    return (1 + x) * ((x >= -1) & (x <= 0)) + (1-x) * ((x >= 0) & (x <= 1))

  def densidad_nucleo(self, h, kernel, x_vals):

    n = len(self.data)
    density = np.zeros(len(x_vals), dtype=float)

    for i, x in enumerate(x_vals):
      u = (self.data - x) / h
      if (kernel == 'gaussiano'):
        density[i] = np.sum(self.kernel_gaussiano(u))
      elif (kernel == 'uniforme'):
        density[i] = np.sum(self.kernel_uniforme(u))
      elif (kernel == 'cuadratico'):
        density[i] = np.sum(self.kernel_cuadratico(u))
      elif (kernel == 'triangular'):
        density[i] = np.sum(self.kernel_triangular(u))

    density /= (n * h)

    return density

  def qqplot(self, newdata=None):

    if (newdata == None):
      newdata = self.data

    ## Completar
    data_sorted = sorted(newdata)
    data_mean = newdata.mean()
    data_std = newdata.std()

    cuantiles_muestrales = (data_sorted - data_mean) / data_std

    probabilidades = np.arange(1/(len(newdata) + 1), 1, 1/(len(newdata) + 1))
    cuantiles_teoricos = norm.ppf(probabilidades)

    plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos,cuantiles_teoricos , linestyle='-', color='red')
    plt.show()

    return