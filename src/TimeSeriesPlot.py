import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator
import mplfinance as mpf

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

    def add_low_pass_filtered(self, cutoff_freq: float, sampling_rate: float = 365):
        """
        Agrega la serie suavizada con un filtro pasa bajos FFT.
        """
        filtered_values = self.ts.low_pass_filter(cutoff_freq, sampling_rate)

        self.ax.plot(
            self.ts.dates,
            filtered_values,
            label=f'Filtrada (corte={cutoff_freq:.2f})'
        )
        self._set_axes_for_time_domain()

    def add_ema(self, window):
        """
        Agrega una media móvil exponencial (EMA).
        """
        ema_values = self.ts.ema(window)

        # Asegurar que tenga la misma longitud que las fechas
        min_len = min(len(ema_values), len(self.ts.dates))
        ema_values = ema_values[:min_len]
        dates = self.ts.dates[:min_len]

        self.ax.plot(dates, ema_values, linestyle='--', label=f'EMA ({window})')
        self._set_axes_for_time_domain()

    def add_candlestick(self):
        """
        Agrega un gráfico de velas japonesas.
        Se espera que TimeSeries tenga atributos: dates, open, high, low, close.
        """
        df = self.ts.df_for_candlestick()

        mpf.plot(df, type='candle', style='charles', ax=self.ax, show_nontrading=True)
        self.ax.set_title('Gráfico de Velas Japonesas')
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
        frecuencias_pos, magnitudes_pos, _, _ = self.ts.yearly_frequency_spectrum(365)

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