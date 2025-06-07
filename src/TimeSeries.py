from typing import Tuple
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

class TimeSeries:
    def __init__(self, dates, values):
        self.values = values
        self.dates = dates
        self.time = np.arange(len(dates))

    def detrend(self, grade: int = 1) -> Tuple["TimeSeries", np.ndarray]:
        coef = np.polyfit(self.time, self.values, grade)
        tendency = np.polyval(coef, self.time)
        detrended = self.values - tendency
        return TimeSeries(self.time, detrended), tendency

    def is_stationary(self) -> bool:
        # Test ADF
        adf_result = adfuller(self.values)
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('')

        # Test KPSS
        kpss_result = kpss(self.values, regression='c', nlags='auto')
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
