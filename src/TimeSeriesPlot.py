import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator
import mplfinance as mpf

from src import TimeSeries

class TimeSeriesPlot:
    def __init__(self, ts: "TimeSeries", figsize=(14, 6)):
        self.ts = ts
        self.fig, self.ax = plt.subplots(figsize=figsize)

    # --- --- Dominio de Tiempo --- ---

    def add_original(self, **kwargs):
        self.ax.plot(self.ts.dates, self.ts.values, label=self.ts.name, **kwargs)
        self._set_axes_for_time_domain()

    def add_another(self, other_ts: "TimeSeries", **kwargs):
        """
        Agrega otra serie temporal al gráfico actual.
        """
        self.ax.plot(other_ts.dates, other_ts.values, label=other_ts.name, **kwargs)
        self._set_axes_for_time_domain()


    def add_detrended(self, grade, **kwargs):
        ts_detrended = self.ts.detrend_with_regression_fitting(grade)
        self.ax.plot(self.ts.dates, ts_detrended.values, label=ts_detrended.name, **kwargs)
        self._set_axes_for_time_domain()

    def add_tendency(self, grade, **kwargs):
        tendency = self.ts.tendency_with_regression_fitting(grade)
        self.ax.plot(self.ts.dates, tendency, label=f'Tendencia de {self.ts.name} (grado {grade})', **kwargs)
        self._set_axes_for_time_domain()

    def add_low_pass_filtered(self, cutoff_freq: float, sampling_rate: float = 365, **kwargs):
        """
        Agrega la serie suavizada con un filtro pasa bajos FFT.
        """
        filtered_values = self.ts.low_pass_filter(cutoff_freq, sampling_rate)

        self.ax.plot(self.ts.dates, filtered_values, **kwargs, label=f'{self.ts.name} filtrada pasa-bajo hasta {cutoff_freq}')
        self._set_axes_for_time_domain()

    def add_band_pass_filtered(self, low_cutoff: float, high_cutoff: float, freq_per_year: int = 365,  **kwargs):
        """
        Agrega la serie suavizada con un filtro pasa bajos FFT.
        """
        filtered_values = self.ts.band_pass_filter(low_cutoff, high_cutoff, freq_per_year)

        self.ax.plot(
            self.ts.dates,
            filtered_values,
            label=f'{self.ts.name} filtrada (cortes entre [{low_cutoff:.2f}, {high_cutoff:.2f}])',
            **kwargs
        )
        self._set_axes_for_time_domain()

    def add_ema(self, window, **kwargs):
        """
        Agrega una media móvil exponencial (EMA).
        """
        ema_values = self.ts.ema(window)

        # Asegurar que tenga la misma longitud que las fechas
        min_len = min(len(ema_values), len(self.ts.dates))
        ema_values = ema_values[:min_len]
        dates = self.ts.dates[:min_len]

        self.ax.plot(dates, ema_values, label=f'{self.ts.name} EMA ({window})', **kwargs)
        self._set_axes_for_time_domain()

    def add_candlestick(self):
        """
        Agrega un gráfico de velas japonesas.
        Se espera que TimeSeries tenga atributos: dates, open, high, low, close.
        """
        df = self.ts.df_for_candlestick()

        mpf.plot(df, type='candle', style='charles', ax=self.ax, show_nontrading=True)
        self.ax.plot([], [], label='Velas Japonesas', color='black')

        self.ax.set_title('Gráfico de Velas Japonesas')
        self._set_axes_for_time_domain()

    def add_grid(self):
        self.ax.grid(True)

    def set_title(self, title):
        self.ax.set_title(title)

    def set_lims(self, xlim=None, ylim=None):
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        
    def show(self):
        self.ax.legend()
        plt.show()

    # --- --- Correlación cruzada --- ---
    def add_cross_correlation_plot(self, other: "TimeSeries", max_lag: int = 30):
        """
        Plotea la correlación cruzada con otra serie temporal
        """
        corr, lags = self.ts.cross_correlation(other, max_lag)
        N = len(self.ts.values)
        signif = 2 / np.sqrt(N)

        self.ax.bar(lags, corr, width=0.6, color='tab:blue', edgecolor='black', linewidth=0.5, alpha=0.8, label='Correlaciones de Pearson')

        # Bandas de significancia sombreadas
        self.ax.fill_between(lags, signif, -signif, color='purple', alpha=0.2, label='Umbral de significancia')

        self.ax.set_title(f'Correlación cruzada ({self.ts.name} vs {other.name})')
        self.ax.set_xlabel('Lag')
        self.ax.set_ylabel('Correlación')
        self.ax.grid(True)
        self.ax.legend()

    # --- --- Dominio de Frecuencia --- ---
    # Crear nuevo TimeSeriesPlot para esto, no usar el mismo que para dom de tiempo.

    def add_yearly_frequency_spectrum(self, low_freqs_limit=50, ticks_freq= 1, color='purple'):
        frecuencias_pos, magnitudes_pos, _, _ = self.ts.yearly_frequency_spectrum(365)

        # Crear nueva figura para este plot
        self.ax.plot(
            frecuencias_pos[:low_freqs_limit],
            magnitudes_pos[:low_freqs_limit],
            color=color,
            label=f'Frecuencia {self.ts.name} (ciclos/año)'
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