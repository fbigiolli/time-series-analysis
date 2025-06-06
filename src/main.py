import os
from CandlestickChart import CandlestickChart

def main():
    print("Obteniendo datos del ticker...")
    ticker = "SPY"

    chart = CandlestickChart(ticker, start="2024-01-01", end="2025-06-06")

    print("Agregando indicadores extra...")
    chart.add_ema(20, color='blue')
    chart.add_ema(100, color='orange')
    chart.add_rsi()
    chart.add_bollinger_bands()

    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"candlestick_{ticker}.png")

    print("Generando grafico...")
    chart.plot(save_path=output_file)
    print(f"Grafico generado y guardado en {output_file}")

if __name__ == "__main__":
    main()
