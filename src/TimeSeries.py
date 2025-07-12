import warnings
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import correlate

class TimeSeries:
    def __init__(self, name, dates, values, open=None, high=None, low=None):
        self.name = name
        self.values = values
        self.dates = dates
        self.time = np.arange(len(dates))

        self.open = open
        self.high = high
        self.low = low
        self.close = self.values # alias para mayor claridad

    # --- --- --- Trend and Regression --- --- ---

    def detrend_with_regression_fitting(self, grade: int = 1) -> "TimeSeries":
        """
        Eliminación de tendencia mediante modelo de regresión linael o polinomial.
        """
        tendency = self.tendency_with_regression_fitting(grade)
        detrended = self.values - tendency
        return TimeSeries(f'{self.name} detrended', self.dates, detrended)

    def differentiate(self, times: int = 1) -> "TimeSeries":
        """
        Hacer diferencia de cada valor con su anterior, para hacer seire estacionaria
        (por ej. para obtener el parámetro `d` del modelo ARIMA)
        """
        detrended = self.values.copy()
        for _ in range(times):
            detrended = np.diff(detrended)
        new_date = self.dates[times:] # porque np.diff lo acorta
        return TimeSeries(f'{self.name} detrended', new_date, detrended)

    def tendency_with_regression_fitting(self, grade: int = 1) -> np.array:
        """
        Línea de regresión lineal o polinomial.
        """
        coef = np.polyfit(self.time, self.values, grade)
        return np.polyval(coef, self.time)

    def normalize_base100(self) -> "TimeSeries":
        """
        Devuelve una nueva TimeSeries con los valores normalizados, base 100
        """
        base_value = self.values.iloc[0]
        normalized_values = (self.values / base_value) * 100

        return TimeSeries(name=f"{self.name} (Base 100)", dates=self.dates, values=normalized_values,)

    # --- --- --- Filtros --- --- ---

    def low_pass_filter(self, cutoff_freq: float, freq_per_year: int = 365) -> np.ndarray:
        """
        Filtro pasa bajos FFT: conserva solo frecuencias <= cutoff_freq (ciclos/año).
        """
        _, _, fft_result, fft_freq = self.yearly_frequency_spectrum(freq_per_year)

        mask = np.abs(fft_freq) <= cutoff_freq
        filtered_fft = fft_result * mask
        filtered_values = np.fft.ifft(filtered_fft).real

        return filtered_values

    def band_pass_filter(self, low_cutoff: float, high_cutoff: float, freq_per_year: int = 365) -> np.ndarray:
        """
        Filtro pasa bandas FFT: conserva frecuencias entre low_cutoff y high_cutoff (ciclos/año).
        """
        _, _, fft_result, fft_freq = self.yearly_frequency_spectrum(freq_per_year)

        # Crear máscara de pasa banda
        mask = (np.abs(fft_freq) >= low_cutoff) & (np.abs(fft_freq) <= high_cutoff)
        filtered_fft = fft_result * mask
        filtered_values = np.fft.ifft(filtered_fft).real

        return filtered_values

    def ema(self, window):
        """
        Retorna una nueva lista o array con la media móvil exponencial (EMA).
        """
        return pd.Series(self.values).ewm(span=window, adjust=False).mean().values

    def df_for_candlestick(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'Date': self.dates,
            'Open': self.open,
            'High': self.high,
            'Low': self.low,
            'Close': self.close
        })
        df.set_index('Date', inplace=True)
        return df

    # --- --- --- Operaciones con otras series --- --- ---
    def cross_correlation(self, other: "TimeSeries", max_lag: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula la correlación cruzada normalizada entre esta serie y otra.
        """
        x = self.values
        y = other.values
        N = min(len(x), len(y))  # Asegurar misma longitud
        x = x[:N]
        y = y[:N]

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        corr = correlate(x_centered, y_centered, mode='full')
        corr_normalized = corr / (np.std(x) * np.std(y) * N)
        lags = np.arange(-N + 1, N)

        if max_lag is not None:
            mask = (lags >= -max_lag) & (lags <= max_lag)
            corr_normalized = corr_normalized[mask]
            lags = lags[mask]

        return corr_normalized, lags

    # --- --- --- Frequency Domain operations --- --- ---
    def yearly_frequency_spectrum(self, freq_per_year=365):
        """
        Calcula el espectro de frecuencias (en ciclos por año) tomando `freq_per_year` como
        el número de observaciones por año (365 si es diaria, 12 si es mensual, etc.).
        """
        n = len(self.values)
        dt = 1 / freq_per_year  # intervalo en años entre observaciones

        # FFT y frecuencias en ciclos por año
        fft_result = np.fft.fft(self.values)
        fft_freq = np.fft.fftfreq(n, d=dt)
        magnitudes = np.abs(fft_result) / n

        # Solo frecuencias positivas
        mask = fft_freq > 0
        frecuencias_pos = fft_freq[mask]
        magnitudes_pos = magnitudes[mask]

        return frecuencias_pos, magnitudes_pos, fft_result, fft_freq

    # --- --- --- Stationarity check --- --- ---

    def is_stationary(self) -> bool:
        # Test ADF
        adf_result = adfuller(self.values)
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('')

        # Test KPSS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InterpolationWarning)
            kpss_result = kpss(self.values, regression='ct', nlags='auto')
        print('KPSS Statistic:', kpss_result[0])
        print('p-value:', kpss_result[1])
        print('')

        adf_p = adf_result[1]
        kpss_p = kpss_result[1]

        is_stationary = False

        # Decisión sobre estacionariedad
        if adf_p < 0.05 and kpss_p >= 0.05:
            decision = "Estacionaria"
            is_stationary = True
        elif adf_p < 0.05 and kpss_p < 0.05:
            decision = "Indefinida"
        elif adf_p >= 0.05 and kpss_p >= 0.05:
            decision = "Indefinida"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            decision = "No estacionaria"
        else:
            decision = "Indefinida"

        print("Decisión:", decision)

        return is_stationary
