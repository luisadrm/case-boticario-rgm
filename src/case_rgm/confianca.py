"""Semáforo de confiança: conta sinais de alerta em cada previsão."""

import numpy as np
import pandas as pd

from .config import ALERTAS_ATENCAO, LIMITE_CV, LIMITE_ERRO, LIMITE_IDADE, LIMITE_LARGURA

CONFIAVEL, ATENCAO = "confiável", "atenção"


def contar_alertas(erro, largura, cv, idade, evento):
    return ((np.asarray(largura) > LIMITE_LARGURA).astype(int)
            + (np.asarray(erro) > LIMITE_ERRO).astype(int)
            + (np.asarray(cv) > LIMITE_CV).astype(int)
            + 2 * (np.asarray(idade) < LIMITE_IDADE).astype(int)
            + np.asarray(evento).astype(int))


def semaforo(alertas):
    return np.where(np.asarray(alertas) >= ALERTAS_ATENCAO, ATENCAO, CONFIAVEL)


def erro_historico(bt):
    """WAPE de cada série no backtest."""
    erro = (bt["demanda"] - bt["p50"]).abs()
    g = pd.DataFrame({"serie": bt["serie"].astype(str), "erro": erro, "y": bt["demanda"]}).groupby("serie")
    return (g["erro"].sum() / g["y"].sum()).rename("erro_hist")
