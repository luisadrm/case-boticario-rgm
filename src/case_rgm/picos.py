"""Prophet por série, usado na previsão para semanas de pico (combinação com o LightGBM)."""

import logging

import numpy as np
import pandas as pd

from .features import semana_do_evento

REGRESSORES = ["log_preco_rel", "flag_combo", "flag_pague_e_leve"]


def prever_prophet(base_f, ultima):
    """Ajusta um Prophet por série com toda a história e prevê as semanas posteriores a `ultima`."""
    from prophet import Prophet  # import tardio: só é necessário quando a combinação é pedida

    for nome in ["cmdstanpy", "prophet"]:
        logging.getLogger(nome).disabled = True
    feriados = pd.DataFrame([{"holiday": ev, "ds": s, "lower_window": -7 if ev == "natal" else 0, "upper_window": 0}
                             for ev in ["black_friday", "natal", "maes", "namorados"]
                             for s in semana_do_evento(base_f["semana"], ev)])
    tipos = {c: float for c in REGRESSORES}
    saida = []
    for _, g in base_f.groupby("serie", observed=True):
        tr = g[g["semana"] <= ultima].assign(y=np.log1p(g["demanda_interp"]))
        fu = g[g["semana"] > ultima]
        m = Prophet(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False,
                    holidays=feriados, uncertainty_samples=0)
        for c in REGRESSORES:
            m.add_regressor(c)
        m.fit(tr[["semana", "y"] + REGRESSORES].rename(columns={"semana": "ds"}).astype(tipos))
        fc = m.predict(fu[["semana"] + REGRESSORES].rename(columns={"semana": "ds"}).astype(tipos))
        saida.append(pd.Series(np.expm1(fc["yhat"].values).clip(0), index=fu.index))
    return pd.concat(saida)
