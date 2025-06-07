import matplotlib.pyplot as plt
from src import TimeSeries

class TimeSeriesPlot:
    def __init__(self, ts: "TimeSeries"):
        self.ts = ts
        self.fig, self.ax = plt.subplots(figsize=(14, 6))

    def addOriginal(self, color='orange'):
        self.ax.plot(self.ts.dates, self.ts.values, label="Original", color=color)
        self._setAxesForTimeDomain()

    def addTendency(self, grade=1, color='black', linestyle='--'):
        _, tendency = self.ts.detrend(grade)
        self.ax.plot(self.ts.dates, tendency, label=f'Tendencia (grado {grade})', color=color, linestyle=linestyle)
        self._setAxesForTimeDomain()

    def addDetrended(self, grade=1, color='gray'):
        ts_detrended, _ = self.ts.detrend(grade)
        self.ax.plot(self.ts.dates, ts_detrended.values, label="Detrended", color=color)
        self._setAxesForTimeDomain()

    def addFiltered(self, serie, corte, color='blue'):
        self.ax.plot(self.ts.dates, serie, label=f'Filtrada (corte={corte})', color=color, linewidth=2)
        self._setAxesForTimeDomain()

    def addGrid(self):
        self.ax.grid(True)

    def setTitle(self, title):
        self.ax.set_title(title)
        
    def show(self):
        self.ax.legend()
        plt.show()

    # --- --- --- --- --- --- --- --- ---

    def _setAxesForTimeDomain(self):
        self.ax.set_xlabel("Fecha")
        self.ax.set_ylabel("Precio")