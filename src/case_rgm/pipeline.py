"""Pipeline de ponta a ponta: CSV bruto → previsão das próximas semanas com faixa e semáforo.

Uso:
    python -m case_rgm.pipeline --dados data/raw/vendas_sku_semanal.csv --saida outputs
    python -m case_rgm.pipeline --dados ... --sem-prophet     # pula a combinação para picos

Saídas em --saida:
    previsao_4_semanas.csv   previsão por série e semana, faixa provável e semáforo
    metricas_backtest.json   erro, viés e cobertura nos dois backtests e as folgas da faixa
    log_limpeza.csv          decisões de limpeza e linhas afetadas
    base_limpa.csv           base tratada (opcional, --salvar-base)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import H
from .confianca import contar_alertas, erro_historico, semaforo
from .features import base_futura, colunas_evento, construir_features, preparar_base
from .limpeza import ler_bruto, limpar
from .modelo import (aplicar_folga, backtest, eh_pico, folga_conformal, inicios_q4, inicios_recentes,
                     metricas, prever, treinar, treino_valido)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rodar(caminho_dados, pasta_saida="outputs", com_prophet=True, salvar_base=False):
    saida = Path(pasta_saida)
    saida.mkdir(parents=True, exist_ok=True)

    _log("1/6 Limpeza")
    df, log_limpeza = limpar(ler_bruto(caminho_dados))
    log_limpeza.to_csv(saida / "log_limpeza.csv", index=False)
    if salvar_base:
        df.to_csv(saida / "base_limpa.csv", index=False)

    _log("2/6 Features")
    d = preparar_base(df)
    base_m = construir_features(d)

    _log("3/6 Backtest (janelas recentes e último Q4)")
    bt_rec = backtest(base_m, inicios_recentes(base_m))
    ini_q4 = inicios_q4(base_m)
    bt_q4 = backtest(base_m, ini_q4) if ini_q4 else bt_rec.iloc[0:0]
    q_comum = folga_conformal(bt_rec)
    pico_rec = bt_rec[bt_rec[colunas_evento()].any(axis=1)]
    q_pico = folga_conformal(pd.concat([bt_q4, pico_rec])) if len(bt_q4) or len(pico_rec) else q_comum

    _log("4/6 Treino final e previsão")
    ultima = d["semana"].max()
    fut = base_futura(d)
    base_f = construir_features(pd.concat([d, fut], ignore_index=True))
    modelos = treinar(treino_valido(base_f[base_f["semana"] <= ultima]))
    X = base_f[base_f["semana"] > ultima].copy()
    X[["p10", "p50", "p90"]] = prever(modelos, X)
    X = aplicar_folga(X, np.where(eh_pico(X), q_pico, q_comum))

    if com_prophet:
        _log("5/6 Prophet para semanas de pico")
        from .picos import prever_prophet
        X["prophet"] = prever_prophet(base_f, ultima)
        X["combinacao"] = np.sqrt(X["p50"] * X["prophet"])
    else:
        _log("5/6 Prophet ignorado (--sem-prophet)")

    _log("6/6 Semáforo de confiança e gravação")
    idade = base_f[base_f["semana"] == ultima].set_index("serie")["idade_semanas"].astype(int) + 1
    X["erro_hist"] = X["serie"].map(erro_historico(aplicar_folga(bt_rec, q_comum)))
    X["largura"] = (X["p90_cal"] - X["p10_cal"]) / X["p50"]
    X["cv"] = X["dp_8s"] / X["media_8s"]
    X["alertas"] = contar_alertas(X["erro_hist"], X["largura"], X["cv"],
                                  X["serie"].map(idade), X[colunas_evento()].any(axis=1))
    X["confianca"] = semaforo(X["alertas"])

    colunas = {"serie": "serie", "categoria_id": "categoria", "semana": "semana", "p50": "previsao",
               "p10_cal": "minimo_provavel", "p90_cal": "maximo_provavel", "erro_hist": "erro_historico",
               "alertas": "alertas", "confianca": "confianca"}
    if com_prophet:
        colunas["combinacao"] = "previsao_picos_lgbm_prophet"
    entrega = (X[list(colunas)].rename(columns=colunas)
               .assign(semana=lambda x: x["semana"].dt.strftime("%Y-%m-%d"), serie=lambda x: x["serie"].astype(str),
                       categoria=lambda x: x["categoria"].astype(str))
               .round({"previsao": 0, "minimo_provavel": 0, "maximo_provavel": 0, "erro_historico": 3,
                       "previsao_picos_lgbm_prophet": 0})
               .sort_values(["semana", "serie"]))
    entrega.to_csv(saida / "previsao_4_semanas.csv", index=False)

    resumo = {
        "ultima_semana_observada": f"{ultima:%Y-%m-%d}",
        "horizonte_semanas": H,
        "backtest_recente": metricas(bt_rec),
        "backtest_q4": metricas(bt_q4) if len(bt_q4) else None,
        "folga_conformal": {"comum": q_comum, "pico": q_pico},
        "semaforo": entrega["confianca"].value_counts().to_dict(),
    }
    (saida / "metricas_backtest.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    _log(f"Pronto: {len(entrega)} previsões em {saida / 'previsao_4_semanas.csv'}")
    return entrega, resumo


def main(argv=None):
    p = argparse.ArgumentParser(description="Previsão de demanda do case RGM (LightGBM + faixa calibrada + semáforo).")
    p.add_argument("--dados", default="data/raw/vendas_sku_semanal.csv", help="CSV bruto do case")
    p.add_argument("--saida", default="outputs", help="pasta de saída")
    p.add_argument("--sem-prophet", action="store_true", help="não calcula a previsão combinada para picos")
    p.add_argument("--salvar-base", action="store_true", help="grava também a base limpa")
    a = p.parse_args(argv)
    rodar(a.dados, a.saida, com_prophet=not a.sem_prophet, salvar_base=a.salvar_base)


if __name__ == "__main__":
    main()
