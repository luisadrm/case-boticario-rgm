"""Limpeza da base bruta (seção 1 do notebook), sem gráficos nem prints."""

import numpy as np
import pandas as pd

from .config import CHAVE, FLAGS, SERIE


def ler_bruto(caminho):
    """Lê o CSV com 'semana' como texto, para tratar os dois formatos de data."""
    return pd.read_csv(caminho, dtype={"semana": str})


def limpar(bruto):
    """Aplica todos os tratamentos do notebook e devolve (base limpa, log de decisões)."""
    df = bruto.copy()
    log = []

    def registrar(etapa, problema, linhas, decisao):
        log.append({"etapa": etapa, "problema": problema, "linhas_afetadas": int(linhas), "decisao": decisao})

    # Datas em dois formatos
    iso = pd.to_datetime(df["semana"], format="%Y-%m-%d", errors="coerce")
    br = pd.to_datetime(df["semana"], format="%d/%m/%Y", errors="coerce")
    registrar("datas", "formato dd/mm/aaaa", iso.isna().sum(), "convertido para ISO")
    df["semana"] = iso.fillna(br)
    if df["semana"].isna().any():
        raise ValueError("Há datas em formato desconhecido na coluna 'semana'.")

    # Subcategoria faltante: o SKU tem uma única subcategoria
    mapa_sub = df.dropna(subset=["subcategoria_id"]).groupby("sku_id")["subcategoria_id"].first()
    registrar("subcategoria", "NaN", df["subcategoria_id"].isna().sum(), "preenchido pelo mapa SKU→subcategoria")
    df["subcategoria_id"] = df["subcategoria_id"].fillna(df["sku_id"].map(mapa_sub))

    # Duplicatas: exatas e conflitantes
    registrar("duplicatas", "linhas idênticas", df.duplicated().sum(), "removida a cópia")
    df = df.drop_duplicates()
    conf = df[df.duplicated(CHAVE, keep=False)]
    num_cols = df.select_dtypes("number").columns
    agg = {c: "mean" for c in num_cols}
    agg.update({c: "max" for c in FLAGS})
    agg.update({"categoria_id": "first", "subcategoria_id": "first"})
    df = df.groupby(CHAVE, as_index=False).agg(agg)
    chaves_conf = df.set_index(CHAVE).index.isin(conf.set_index(CHAVE).index)
    for c in ["qt_vendida", "qt_ruptura"]:
        df.loc[chaves_conf, c] = np.floor(df.loc[chaves_conf, c] + 0.5)   # média arredondada (,5 sobe)
    df.loc[chaves_conf, "receita"] = (df.loc[chaves_conf, "qt_vendida"] * df.loc[chaves_conf, "preco_medio"]).round(2)
    registrar("duplicatas", "mesma chave, valores diferentes", len(conf) // 2, "média arredondada; receita = qt × preço")

    # Quantidade negativa (sinal invertido)
    neg = df["qt_vendida"] < 0
    df["flag_qt_corrigida"] = neg
    df.loc[neg, "qt_vendida"] = df.loc[neg, "qt_vendida"].abs()
    registrar("qt_vendida", "valor negativo", neg.sum(), "abs(qt)")

    # Preço ≠ receita/qt (casa decimal)
    preco_calc = df["receita"] / df["qt_vendida"]
    erro_preco = ((preco_calc / df["preco_medio"] - 1).abs() > 0.01) & df["preco_medio"].notna()
    df["flag_preco_corrigido"] = erro_preco
    df.loc[erro_preco, "preco_medio"] = preco_calc[erro_preco].round(2)
    registrar("preco_medio", "≠ receita/qt", erro_preco.sum(), "recalculado como receita/qt")

    # Desconto ≥ 100%: recalculado pelo preço de tabela implícito da série
    ok = df["desconto_pct"].between(0, 100, inclusive="left") & df["preco_medio"].notna()
    impl = pd.Series(np.where(ok, df["preco_medio"] / (1 - df["desconto_pct"] / 100), np.nan), index=df.index)
    tabela = impl.groupby([df["sku_id"], df["canal"]]).transform("median")
    df["preco_tabela"] = tabela
    ruim = df["desconto_pct"] >= 100
    df["flag_desconto_corrigido"] = ruim
    df.loc[ruim, "desconto_pct"] = (100 * (1 - df["preco_medio"] / tabela))[ruim].clip(lower=0).round(2)
    registrar("desconto_pct", "> 100%", ruim.sum(), "recalculado = 1 − preço/preço de tabela")

    # Quantidade faltante recuperável por receita/preço
    recup = df["qt_vendida"].isna() & df["receita"].notna() & df["preco_medio"].notna()
    df["flag_qt_imputada"] = recup
    df.loc[recup, "qt_vendida"] = (df.loc[recup, "receita"] / df.loc[recup, "preco_medio"]).round()
    registrar("qt_vendida", "NaN com receita e preço", recup.sum(), "qt = receita/preço")

    # Zeros suspeitos: venda zero sem ruptura → dado faltante
    df = df.sort_values(CHAVE).reset_index(drop=True)
    zeros = df["qt_vendida"] == 0
    suspeito = zeros & (df["qt_ruptura"].fillna(0) == 0)
    df["flag_zero_suspeito"] = suspeito
    df.loc[suspeito, ["qt_vendida", "receita"]] = np.nan
    registrar("qt_vendida", "zero sem ruptura", suspeito.sum(), "NaN + flag")
    rz = (df["qt_vendida"] == 0) & df["receita"].isna()
    df.loc[rz, "receita"] = 0.0

    # Venda interpolada (só para médias recentes; o alvo continua com NaN)
    df["qt_vendida_interp"] = df.groupby(SERIE)["qt_vendida"].transform(lambda s: s.interpolate(limit_direction="both"))

    # Ruptura faltante → 0 + flag
    rnan = df["qt_ruptura"].isna()
    df["flag_ruptura_imputada"] = rnan
    df["qt_ruptura"] = df["qt_ruptura"].fillna(0)
    registrar("qt_ruptura", "NaN", rnan.sum(), "0 + flag")

    # Margem faltante → interpolação na série
    mnan = df["margem_pct"].isna() & df["preco_medio"].notna()
    df["flag_margem_imputada"] = mnan
    df["margem_pct"] = df.groupby(SERIE)["margem_pct"].transform(lambda s: s.interpolate(limit_direction="both"))
    registrar("margem_pct", "NaN com venda", mnan.sum(), "interpolação na série")

    # Preço e desconto para o modelo (semanas sem venda não têm preço observado)
    df["desconto_modelo"] = df["desconto_pct"]
    sem_preco = df["desconto_modelo"].isna()
    df.loc[sem_preco & ~df[FLAGS].any(axis=1), "desconto_modelo"] = 0.0
    df["desconto_modelo"] = df.groupby(SERIE)["desconto_modelo"].transform(lambda s: s.ffill().bfill())
    df["preco_modelo"] = df["preco_medio"].fillna(df["preco_tabela"] * (1 - df["desconto_modelo"] / 100))

    # Demanda = venda + ruptura
    df["demanda"] = df["qt_vendida"] + df["qt_ruptura"]
    df["demanda_interp"] = df["qt_vendida_interp"] + df["qt_ruptura"]
    df["teve_ruptura"] = df["qt_ruptura"] > 0
    df["n_mecanicas"] = df[FLAGS].sum(axis=1)
    df["em_promo"] = df["n_mecanicas"] > 0
    return df, pd.DataFrame(log)
