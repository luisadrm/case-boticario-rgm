# Modelos de previsão: vantagens e desvantagens neste case

Resumo de consulta rápida. Os números, a metodologia e a discussão completa estão em [`relatorio_modelos.md`](relatorio_modelos.md).

**Contexto que pesa na escolha:** 40 séries semanais curtas (até 105 semanas), demanda movida por **promoções planejadas** e **datas comemorativas**, canibalização entre SKUs irmãos, **lançamentos** com pouco histórico e apenas **uma Black Friday** anterior nos dados.

## Placar (WAPE: erro em % do volume; menor é melhor)

| Modelo | Tipo | Mai–out/26 | Q4/25 | Mês da BF | Lançamentos (Q4) |
|---|---|---|---|---|---|
| LSTM | Global, rede neural | **15,2%** | **19,0%** | 29,4% | **13,6%** |
| LightGBM | Global, árvores | 15,5% | 21,5% | 32,7% | 18,2% |
| Prophet | Local, decomposição | 17,8% | 20,9% | **19,8%** | 51,7% |
| ARIMA (com regressores) | Local, estatístico | 17,9% | 24,0% | 23,4% | 62,8% |
| ETS *(descartado)* | Local, estatístico | 21,7% | 38,9% | 34,3% | 64,0% |
| LightGBM + LSTM | Combinação | **14,7%** | 19,2% | 31,1% | — |
| LightGBM + Prophet | Combinação | 15,8% | **19,1%** | 25,2% | — |

## ARIMA (com regressores)

| ✅ Vantagens | ❌ Desvantagens |
|---|---|
| Faixa de confiança mais bem calibrada do estudo | Efeitos lineares e fixos, sem interações (promo × evento × canal) |
| Usa promoção e eventos como variáveis externas | Estima cada série sozinha: instável com uma única BF |
| Teoria sólida, coeficientes interpretáveis | Não funciona em lançamentos (63% de erro) |
| Subestima a BF menos que os modelos globais | O mais lento; frágil com regressores sem variação |

**Uso recomendado:** "segundo parecer" e referência de incerteza.

## ETS: descartado

| ✅ Vantagens | ❌ Desvantagens |
|---|---|
| Simples, rapidíssimo, intervalos nativos | Não aceita variáveis externas: cego a promoção e evento |
| Ótimo para itens estáveis com histórico longo | Sem 2 anos de dados, não modela sazonalidade |
| | O pior no Q4 (38,9%); +54% de erro em janeiro (carrega o pico do Natal) |

**Uso recomendado:** nenhum neste case (RGM é justamente gerenciar promoção e calendário).

## Prophet

| ✅ Vantagens | ❌ Desvantagens |
|---|---|
| **Melhor nos picos:** datas comemorativas como "feriados" explícitos (viés de −2% no Q4) | Fraco em promoções: regressores aditivos, sem interações nem canibalização |
| Componentes fáceis de mostrar ao negócio | Faixa de confiança otimista (64–73% de cobertura para 80%) |
| Robusto a buracos e mudanças de patamar | Não funciona em lançamentos (52% de erro) |
| Ótimo parceiro de *ensemble* | 40 ajustes por janela; sazonalidade anual pouco confiável com < 2 anos |

**Uso recomendado:** semanas de pico, combinado com o LightGBM.

## LSTM

| ✅ Vantagens | ❌ Desvantagens |
|---|---|
| Menor erro individual nos dois períodos | Também subestima a BF (−28% no mês) |
| Melhor em lançamentos: aprende com todos os SKUs | Varia entre execuções (±0,3–0,5 p.p.): exige média de várias redes |
| Lê a sequência (aceleração, ressaca) sem lags manuais | Explicabilidade aproximada (DeepSHAP falha na LSTM; Expected Gradients é aproximado) |
| Faixa nativa razoavelmente calibrada (78%) | Hiperparâmetros não ajustados; ~10–15× mais lento que o LightGBM; exige PyTorch e especialização |

**Uso recomendado:** candidato à próxima versão, sozinho ou com o LightGBM, após validar em mais de um Q4.

## LightGBM: modelo principal

| ✅ Vantagens | ❌ Desvantagens |
|---|---|
| Precisão no nível do melhor; o melhor na CAT_A (promocional) | Subestima o mês da BF (−31%): árvores não extrapolam e só viram uma BF |
| **SHAP exato**: explicação por previsão em linguagem de negócio | Faixa crua otimista: precisa de calibração conformal |
| **Simulador de RGM** (preço, tipo de desconto, calendário como entrada) | Depende de engenharia de features e de cuidado com vazamento |
| Rápido, determinístico, fácil de manter | Associação, não causalidade (para precificar, usar as elasticidades) |
| Usa a promoção dos irmãos (canibalização) e funciona em lançamentos | |

**Uso recomendado:** previsão do dia a dia e simulação de promoções.

## Em uma frase cada

- **ARIMA:** confiável na incerteza, limitado na dinâmica.
- **ETS:** simples demais para um negócio movido a promoção.
- **Prophet:** o especialista em datas comemorativas.
- **LSTM:** o mais preciso, mas o mais difícil de explicar e de manter.
- **LightGBM:** o melhor equilíbrio entre precisão, explicação e operação.
- **Lição geral:** **modelo global + alvo relativo + calendário futuro** pesou mais que a escolha do algoritmo.
