import warnings
import numpy as np
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

class TimeSeries:
    def __init__(self, dates, values):
        self.values = values
        self.dates = dates
        self.time = np.arange(len(dates))

    # --- --- --- Time Domain operations --- --- ---

    def detrend_with_regression_fitting(self, grade: int = 1) -> "TimeSeries":
        """
        Eliminación de tendencia mediante modelo de regresión linael o polinomial.
        """
        tendency = self.tendency_with_regression_fitting(grade)
        detrended = self.values - tendency
        return TimeSeries(self.dates, detrended)

    def detrend_with_differencing(self, times: int = 1) -> "TimeSeries":
        """
        Eliminación de tendencia mediante diferenciación de `times` veces, con el objetivo de
        hacer estacionaria una serie (por ej. para obtener el parámetro `d` del modelo ARIMA)
        """
        detrended = self.values.copy()
        for _ in range(times):
            detrended = np.diff(detrended)
        new_date = self.dates[times:] # porque np.diff lo acorta
        return TimeSeries(new_date, detrended)

    def tendency_with_regression_fitting(self, grade: int = 1) -> np.array:
        """
        Línea de regresión lineal o polinomial.
        """
        coef = np.polyfit(self.time, self.values, grade)
        return np.polyval(coef, self.time)

    def low_pass_filter(self, cutoff_freq: float, freq_per_year: int = 365) -> np.ndarray:
        """
        Filtro pasa bajos FFT: conserva solo frecuencias <= cutoff_freq (ciclos/año).
        """
        _, _, fft_result, fft_freq = self.yearly_frequency_spectrum(freq_per_year)

        mask = np.abs(fft_freq) <= cutoff_freq
        filtered_fft = fft_result * mask
        filtered_values = np.fft.ifft(filtered_fft).real

        return filtered_values

    # --- --- --- Frequency Domain operations --- --- ---
    def yearly_frequency_spectrum(self, freq_per_year=365, on_detrended = True, on_detrended_grade = 2):
        """
        Calcula el espectro de frecuencias (en ciclos por año) tomando `freq_per_year` como
        el número de observaciones por año (365 si es diaria, 12 si es mensual, etc.).

        El parámetro `on_detrended` determina si se aplica sobre la serie hecha estacionaria
        """
        n = len(self.values)
        dt = 1 / freq_per_year  # intervalo en años entre observaciones

        # FFT y frecuencias en ciclos por año
        fft_result = np.fft.fft(
            self.detrend_with_regression_fitting(on_detrended_grade).values if on_detrended else self.values
        )
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
