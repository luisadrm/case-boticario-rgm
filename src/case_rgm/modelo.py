"""LightGBM com alvo relativo, quantis P10/P50/P90, backtest e calibração conformal."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import COBERTURA_ALVO, H, N_JANELAS_Q4, N_JANELAS_RECENTES, PARAMS_LGBM
from .features import colunas_evento, lista_features


def wape(y, p):
    return np.abs(y - p).sum() / y.sum()


def vies(y, p):
    return (p - y).sum() / y.sum()


def treinar(treino, features=None):
    features = features or lista_features()
    X, y = treino[features], treino["y"]
    return {
        "p50": lgb.LGBMRegressor(objective="regression", **PARAMS_LGBM).fit(X, y),
        "p10": lgb.LGBMRegressor(objective="quantile", alpha=0.10, **PARAMS_LGBM).fit(X, y),
        "p90": lgb.LGBMRegressor(objective="quantile", alpha=0.90, **PARAMS_LGBM).fit(X, y),
    }


def prever(modelos, X, features=None):
    """Devolve P10, P50 e P90 em unidades, com os quantis em ordem."""
    features = features or lista_features()
    nivel = np.log1p(X["media_8s"])
    out = pd.DataFrame({k: np.expm1(m.predict(X[features]) + nivel) for k, m in modelos.items()},
                       index=X.index).clip(lower=0)
    out = out[["p10", "p50", "p90"]]
    out[:] = np.sort(out.values, axis=1)
    return out


def treino_valido(base):
    return base[base["y"].notna() & base["lag_4"].notna()]


def backtest(base_m, inicios):
    """Origem móvel: para cada início, treina só com o passado e prevê as H semanas seguintes."""
    resultados = []
    for ini in inicios:
        ini = pd.Timestamp(ini)
        fim = ini + pd.Timedelta(weeks=H - 1)
        treino = treino_valido(base_m[base_m["semana"] < ini])
        teste = base_m[(base_m["semana"] >= ini) & (base_m["semana"] <= fim)
                       & base_m["demanda"].notna() & base_m["media_8s"].notna()].copy()
        teste[["p10", "p50", "p90"]] = prever(treinar(treino), teste)
        teste["origem"] = ini
        resultados.append(teste)
    return pd.concat(resultados)


def inicios_recentes(base_m, n=N_JANELAS_RECENTES):
    semanas = np.sort(base_m["semana"].unique())
    return list(semanas[-n * H::H])


def inicios_q4(base_m, n=N_JANELAS_Q4):
    """Janelas a partir da primeira segunda-feira de outubro do último Q4 completo nos dados."""
    sem = pd.to_datetime(base_m["semana"])
    anos = sorted(a for a in sem.dt.year.unique() if (sem >= pd.Timestamp(f"{a}-12-31") - pd.Timedelta(days=6)).any()
                  and (sem <= pd.Timestamp(f"{a}-10-07")).any())
    if not anos:
        return []
    a = anos[-1]
    primeira = pd.date_range(f"{a}-10-01", f"{a}-10-07", freq="W-MON")[0]
    return [primeira + pd.Timedelta(weeks=H * k) for k in range(n)]


def eh_pico(base):
    return base[colunas_evento()].any(axis=1) | (pd.to_datetime(base["semana"]).dt.quarter == 4)


def folga_conformal(bt, cobertura=COBERTURA_ALVO):
    """Quantil dos 'escapes' do real em relação à faixa P10–P90, em log."""
    ly, l10, l90 = np.log1p(bt["demanda"]), np.log1p(bt["p10"]), np.log1p(bt["p90"])
    s = np.maximum(l10 - ly, ly - l90)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * cobertura) / n)))


def aplicar_folga(base, q):
    base = base.copy()
    base["p10_cal"] = np.expm1(np.log1p(base["p10"]) - q).clip(lower=0)
    base["p90_cal"] = np.expm1(np.log1p(base["p90"]) + q)
    return base


def metricas(bt):
    return {
        "wape": float(wape(bt["demanda"], bt["p50"])),
        "vies": float(vies(bt["demanda"], bt["p50"])),
        "cobertura_bruta": float(((bt["demanda"] >= bt["p10"]) & (bt["demanda"] <= bt["p90"])).mean()),
        "n": int(len(bt)),
    }
