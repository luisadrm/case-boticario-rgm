"""Parâmetros do pipeline. Os valores são os mesmos do notebook notebooks/case_rgm_v2.ipynb."""

CHAVE = ["sku_id", "canal", "semana"]
SERIE = ["sku_id", "canal"]
FLAGS = ["flag_desconto_direto", "flag_combo", "flag_pague_e_leve"]

H = 4  # horizonte de previsão, em semanas
CATS = ["sku_id", "canal", "categoria_id", "subcategoria_id"]
EVENTOS = ["maes", "namorados", "pais", "black_friday", "natal"]
LAGS = [4, 5, 6, 8, 13, 52]

PARAMS_LGBM = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=42,
)

COBERTURA_ALVO = 0.80     # faixa P10–P90 calibrada para conter o real em 80% dos casos
N_JANELAS_RECENTES = 6    # backtest "normal": últimas 6 janelas de 4 semanas
N_JANELAS_Q4 = 4          # backtest de pico: 4 janelas a partir de outubro do último Q4 completo

# Semáforo: cada condição verdadeira soma um alerta (histórico curto soma dois)
LIMITE_LARGURA = 0.50     # (P90 − P10) ÷ P50
LIMITE_ERRO = 0.18        # WAPE histórico da série no backtest
LIMITE_CV = 0.35          # desvio-padrão ÷ média das últimas 8 semanas
LIMITE_IDADE = 26         # semanas de histórico
ALERTAS_ATENCAO = 2       # a partir de quantos alertas a previsão fica em "atenção"
