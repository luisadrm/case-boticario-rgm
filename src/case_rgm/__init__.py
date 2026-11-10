"""Pipeline de previsão de demanda do case RGM.

Etapas (cada uma em um módulo):
    limpeza    → base limpa a partir do CSV bruto
    features   → variáveis de calendário, promoção dos irmãos, lags e médias recentes
    modelo     → LightGBM (P10/P50/P90), backtest e calibração conformal da faixa
    confianca  → semáforo de confiança (confiável / atenção)
    picos      → Prophet para a previsão de picos (opcional)
    pipeline   → orquestra tudo: python -m case_rgm.pipeline --dados <csv>
"""

__version__ = "1.0.0"
