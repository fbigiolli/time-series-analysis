import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

from src import TimeSeries

class TimeSeriesPlot:
    def __init__(self, ts: "TimeSeries"):
        self.ts = ts
        self.fig, self.ax = plt.subplots(figsize=(14, 6))

    # --- --- Adición de plots en Dominio de Tiempo --- ---

    def add_original(self, color='orange'):
        self.ax.plot(self.ts.dates, self.ts.values, label="Original", color=color)
        self._set_axes_for_time_domain()

    def add_detrended(self, grade=1, color='gray'):
        ts_detrended = self.ts.detrend_with_regression_fitting(grade)
        self.ax.plot(self.ts.dates, ts_detrended.values, label="Detrended", color=color)
        self._set_axes_for_time_domain()

    def add_tendency(self, grade=1, color='black', linestyle='--'):
        tendency = self.ts.tendency_with_regression_fitting(grade)
        self.ax.plot(self.ts.dates, tendency, label=f'Tendencia (grado {grade})', color=color, linestyle=linestyle)
        self._set_axes_for_time_domain()

    def add_filtered(self, serie, corte, color='blue'):
        self.ax.plot(self.ts.dates, serie, label=f'Filtrada (corte={corte})', color=color, linewidth=2)
        self._set_axes_for_time_domain()

    def add_grid(self):
        self.ax.grid(True)

    def set_title(self, title):
        self.ax.set_title(title)
        
    def show(self):
        self.ax.legend()
        plt.show()

    # --- --- Adición de plots en Dominio de Frecuencia --- ---
    # Crear nuevo TimeSeriesPlot para esto, no usar el mismo que para dom de tiempo.

    def add_yearly_frequency_spectrum(self, low_freqs_limit=50, ticks_freq= 1, color='purple'):
        frecuencias_pos, magnitudes_pos = self.ts.yearly_frequency_spectrum(365)

        # Crear nueva figura para este plot
        self.ax.plot(
            frecuencias_pos[:low_freqs_limit],
            magnitudes_pos[:low_freqs_limit],
            color=color,
            label=f'Frecuencia'
        )
        self.ax.set_title('Espectro de Frecuencias (ciclos/año)')
        self.ax.xaxis.set_major_locator(MultipleLocator(ticks_freq))
        self._set_axes_for_frequency_domain()
        self.ax.grid(True)
        plt.show()

    # --- --- --- Privados --- --- ---

    def _set_axes_for_frequency_domain(self):
        self.ax.set_xlabel('Frecuencia (ciclos/año)')
        self.ax.set_ylabel('Magnitud')

    def _set_axes_for_time_domain(self):
        self.ax.set_xlabel("Fecha")
        self.ax.set_ylabel("Precio")