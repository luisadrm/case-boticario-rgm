"""Variáveis do modelo: calendário, promoção dos irmãos, lags e médias recentes."""

import numpy as np
import pandas as pd

from .config import CATS, CHAVE, EVENTOS, FLAGS, H, LAGS, SERIE


def semana_do_evento(datas, evento):
    """Segunda-feira da semana que contém o evento, para cada ano presente em `datas`."""
    out = []
    for a in sorted(set(pd.to_datetime(datas).dt.year)):
        if evento == "maes":            # 2º domingo de maio
            dia = pd.date_range(f"{a}-05-01", f"{a}-05-31", freq="W-SUN")[1]
        elif evento == "namorados":
            dia = pd.Timestamp(f"{a}-06-12")
        elif evento == "pais":          # 2º domingo de agosto
            dia = pd.date_range(f"{a}-08-01", f"{a}-08-31", freq="W-SUN")[1]
        elif evento == "black_friday":  # última sexta de novembro
            dia = pd.date_range(f"{a}-11-01", f"{a}-11-30", freq="W-FRI")[-1]
        elif evento == "natal":
            dia = pd.Timestamp(f"{a}-12-24")
        else:
            raise ValueError(f"Evento desconhecido: {evento}")
        out.append(dia - pd.Timedelta(days=dia.dayofweek))
    return out


def features_calendario(base):
    """Marca a semana de cada data comemorativa e a semana anterior, mais mês e semana do ano."""
    base = base.copy()
    for ev in EVENTOS:
        semanas = semana_do_evento(base["semana"], ev)
        base[f"ev_{ev}"] = base["semana"].isin(semanas).astype(int)
        base[f"ev_{ev}_pre"] = base["semana"].isin([s - pd.Timedelta(weeks=1) for s in semanas]).astype(int)
    base["mes"] = base["semana"].dt.month
    base["semana_ano"] = base["semana"].dt.isocalendar().week.astype(int)
    return base


def colunas_evento():
    return [f"ev_{ev}{suf}" for ev in EVENTOS for suf in ("", "_pre")]


def add_irmaos(base):
    """Fração dos irmãos (mesma subcategoria e canal) e de outras subcategorias em promoção na semana."""
    base = base.copy()
    k = ["subcategoria_id", "canal", "semana"]
    s = base.groupby(k)["em_promo"].agg(n_sub="size", p_sub="sum").reset_index()
    base = base.merge(s, on=k, how="left")
    base["irmao_promo"] = ((base["p_sub"] - base["em_promo"]) / (base["n_sub"] - 1)).where(base["n_sub"] > 1, 0)
    kc = ["categoria_id", "canal", "semana"]
    s2 = base.groupby(kc)["em_promo"].agg(n_cat="size", p_cat="sum").reset_index()
    base = base.merge(s2, on=kc, how="left")
    base["outra_sub_promo"] = ((base["p_cat"] - base["p_sub"]) / (base["n_cat"] - base["n_sub"])).where(
        base["n_cat"] > base["n_sub"], 0)
    return base.drop(columns=["n_sub", "p_sub", "n_cat", "p_cat"])


def preparar_base(df):
    """Base analítica: série, log do preço relativo, calendário e promoção dos irmãos."""
    d = df.copy()
    d["serie"] = d["sku_id"] + " | " + d["canal"]
    d["log_preco_rel"] = np.log(d["preco_modelo"] / d["preco_tabela"])
    d = features_calendario(d)
    return add_irmaos(d)


def lista_features():
    return (CATS
            + [f"lag_{L}" for L in LAGS] + ["media_8s", "dp_8s", "tendencia", "ruptura_recente", "idade_semanas"]
            + ["log_preco_rel", "desconto_modelo"] + FLAGS + ["n_mecanicas", "irmao_promo", "outra_sub_promo"]
            + colunas_evento() + ["mes", "semana_ano"])


def construir_features(base):
    """Lags e médias com defasagem ≥ H (nada do futuro entra) e alvo relativo à média recente."""
    b = base.sort_values(CHAVE).copy()
    g = b.groupby(SERIE)["demanda_interp"]
    for L in LAGS:
        b[f"lag_{L}"] = g.shift(L)
    b["media_8s"] = g.transform(lambda s: s.shift(H).rolling(8, min_periods=2).mean())
    b["dp_8s"] = g.transform(lambda s: s.shift(H).rolling(8, min_periods=2).std())
    b["tendencia"] = b["lag_4"] / b["media_8s"]
    b["ruptura_recente"] = (b.groupby(SERIE)["teve_ruptura"]
                              .transform(lambda s: s.astype(float).shift(H).rolling(8, min_periods=1).mean()))
    b["idade_semanas"] = b.groupby(SERIE).cumcount()
    for c in CATS:
        b[c] = b[c].astype("category")
    b["y"] = np.log1p(b["demanda"]) - np.log1p(b["media_8s"])
    return b


def base_futura(d, horizonte=H):
    """Linhas das próximas semanas no cenário sem promoção, com preço regular de cada série."""
    ultima = d["semana"].max()
    futuras = pd.date_range(ultima + pd.Timedelta(weeks=1), periods=horizonte, freq="7D")
    series = d.drop_duplicates(SERIE)[["sku_id", "canal", "categoria_id", "subcategoria_id", "serie"]]
    fut = series.merge(pd.DataFrame({"semana": futuras}), how="cross")
    sem_promo = d[~d["em_promo"] & d["preco_medio"].notna()]
    preco_regular = sem_promo.groupby("serie")["preco_medio"].median()
    fut["preco_tabela"] = fut["serie"].map(d.groupby("serie")["preco_tabela"].first())
    fut["preco_regular"] = fut["serie"].map(preco_regular)
    fut["log_preco_rel"] = np.log(fut["preco_regular"] / fut["preco_tabela"])
    fut["desconto_modelo"] = 100 * (1 - fut["preco_regular"] / fut["preco_tabela"])
    for f in FLAGS:
        fut[f] = False
    fut[["n_mecanicas", "irmao_promo", "outra_sub_promo"]] = 0
    fut[["em_promo", "teve_ruptura"]] = False
    return features_calendario(fut)
