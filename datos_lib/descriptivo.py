import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class AnalisisDescriptivo:
  '''
  Clase para realizar análisis descriptivo de datos univariantes.
  Permite generar histogramas, evaluar densidades, aplicar diferentes
  kernels y realizar QQ plots.
  Atributos:
    data: array-like, datos a analizar.
  Métodos:
    genera_histograma(h): Genera un histograma con el ancho de bin h.
    evalua_histograma(h, x): Evalúa la densidad del histograma en puntos x.
    kernel_gaussiano(x): Aplica el kernel gaussiano.
    kernel_uniforme(x): Aplica el kernel uniforme.
    kernel_cuadratico(x): Aplica el kernel cuadrático.
    kernel_triangular(x): Aplica el kernel triangular.
    densidad_nucleo(h, kernel, x_vals): Calcula la densidad usando un kernel específico.
    qqplot(newdata=None): Genera un QQ plot de los datos.
  '''
  def __init__(self, data: np.ndarray) -> None:
    self.data = data

  def genera_histograma(self, h: float) -> tuple:
    '''
    Genera un histograma de los datos con un ancho de bin h.
    Args:
      h (float): Ancho del bin para el histograma.
    Returns:
      bins (np.ndarray): Bordes de los bins del histograma.
      densidad (np.ndarray): Densidad de probabilidad en cada bin.
    '''
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

  def evalua_histograma(self, h: float, x: np.ndarray) -> np.ndarray:
    '''
    Evalúa la densidad del histograma en puntos x.
    Args:
      h (float): Ancho del bin para el histograma.
      x (np.ndarray): Puntos donde se evalúa la densidad.
    Returns:
      densidad_x (np.ndarray): Densidad del histograma en los puntos x.
    '''
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
  
  def kernel_gaussiano(self, x: np.ndarray) -> np.ndarray:
    return (1 / np.sqrt(2 * np.pi)) * np.exp((-1/2) * x**2)

  def kernel_uniforme(self, x: np.ndarray) -> np.ndarray:
    return np.where(((x > -1/2) & (x < 1/2)), 1, 0)

  def kernel_cuadratico(self, x: np.ndarray) -> np.ndarray:
    return 3 / 4 * (1 - x**2) * ((x >= -1) & (x <= 1))

  def kernel_triangular(self, x: np.ndarray) -> np.ndarray:
    return (1 + x) * ((x >= -1) & (x <= 0)) + (1-x) * ((x >= 0) & (x <= 1))

  def densidad_nucleo(self, h: float, kernel: str, x_vals: np.ndarray) -> np.ndarray:
    '''
    Calcula la densidad usando un kernel específico.
    Args:
      h (float): Ancho del kernel.
      kernel (str): Tipo de kernel a usar ('gaussiano', 'uniforme', 'cuadratico', 'triangular').
      x_vals (np.ndarray): Puntos donde se evalúa la densidad.
    Returns:
      density (np.ndarray): Densidad estimada en los puntos x_vals.
    '''
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

  def qqplot(self, newdata: np.ndarray = None) -> None:
    '''
    Genera un QQ plot de los datos.
    Args:
      newdata (array-like, optional): Datos nuevos para comparar. Si es None, usa self.data.
    Returns:
      None: Muestra el gráfico del QQ plot.
    '''

    if newdata is not None:
      if not isinstance(newdata, (list, np.ndarray)):
        raise TypeError("newdata debe ser una lista o un array de numpy.")
      data = newdata
    else:
      data = self.data

    ## Completar
    data_sorted = sorted(data)
    data_mean = data.mean()
    data_std = data.std()

    cuantiles_muestrales = (data_sorted - data_mean) / data_std

    probabilidades = np.arange(1/(len(data) + 1), 1, 1/(len(data) + 1))
    cuantiles_teoricos = norm.ppf(probabilidades)

    plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos,cuantiles_teoricos , linestyle='-', color='red')
    plt.show()

    return