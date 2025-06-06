import yfinance as yf
import pandas as pd
import mplfinance as mpf


class CandlestickChart:
    def __init__(self, ticker: str, start="2023-01-01", end="2024-12-31"):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end
        self.df = self._download_data()
        self.additional_plots = []

        self.style = mpf.make_mpf_style(
            base_mpf_style='charles',
            rc={'font.size': 12},
            mavcolors=['blue', 'orange', 'green', 'purple'],
            facecolor='white',
            gridcolor='lightgray',
        )

    def _download_data(self):
        df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        return df

    def add_ema(self, span: int, color='blue'):
        """Agrega una EMA de periodo `span` al gráfico y prepara la leyenda."""
        ema_column = f'EMA{span}'
        self.df[ema_column] = self.df['Close'].ewm(span=span, adjust=False).mean()
        ema_plot = mpf.make_addplot(
            self.df[ema_column],
            color=color,
            secondary_y=False,
            label=f'EMA{span}'
        )
        self.additional_plots.append(ema_plot)

    def add_rsi(self, period=14, color='purple'):
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.df[f'RSI_{period}'] = rsi

        rsi_plot = mpf.make_addplot(rsi, panel=1, color=color, ylabel='RSI')
        self.additional_plots.append(rsi_plot)

    def add_bollinger_bands(self, window=20, num_std=2, color='gray'):
        rolling_mean = self.df['Close'].rolling(window=window).mean()
        rolling_std = self.df['Close'].rolling(window=window).std()
        upper_band = rolling_mean + num_std * rolling_std
        lower_band = rolling_mean - num_std * rolling_std

        self.df[f'BB_upper_{window}'] = upper_band
        self.df[f'BB_lower_{window}'] = lower_band

        self.additional_plots.append(mpf.make_addplot(upper_band, color=color, linestyle='--', label=f'BB+{num_std}σ'))
        self.additional_plots.append(mpf.make_addplot(lower_band, color=color, linestyle='--', label=f'BB-{num_std}σ'))

    def plot(self, save_path=None, dpi=600):
        """Plotea el gráfico con las EMAs agregadas, incluyendo leyenda."""
        kwargs = {
            'type': 'candle',
            'style': self.style,
            'volume': True,
            'addplot': self.additional_plots if self.additional_plots else None,
            'title': f'{self.ticker} - Candlestick Chart',
            'ylabel': 'Precio (USD)',
            'ylabel_lower': 'Volumen',
            'figratio': (16, 9),
            'figscale': 1.5,
            'tight_layout': True,
            'returnfig': True
        }

        fig, axes = mpf.plot(self.df, **kwargs)

        self._add_ema_labels_to_plot(axes)
        self._save_plot_to_disk(dpi, fig, save_path)

    def _save_plot_to_disk(self, dpi, fig, save_path):
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)
        else:
            fig.show()

    def _add_ema_labels_to_plot(self, axes):
        if self.additional_plots:
            ax = axes[0]
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper left')