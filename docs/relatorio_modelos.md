# Relatório de modelagem: previsão de demanda semanal por SKU × canal

**Case técnico Data Science Espec I, RGM (Grupo Boticário)** · notebook de referência: `notebooks/case_rgm_v2.ipynb`

---

## Sumário executivo

- Foram comparados **cinco modelos** de previsão de demanda com o mesmo protocolo: **ARIMA** (com regressores), **ETS**, **Prophet**, **LSTM** e **LightGBM** e duas combinações.
- **No período "normal" (mai–out/26)**, LSTM e LightGBM empatam na liderança, com **~15% de erro** (WAPE), à frente dos modelos por série (~18%). A combinação dos dois chega a **14,7%**, o menor erro do estudo.
- **No 4º trimestre (Q4/25, com Black Friday e Natal)**, todos pioram. O melhor sozinho é o **LSTM (19,0%)**, empatado com as combinações LightGBM + Prophet (19,1%) e LightGBM + LSTM (19,2%).
- **No mês da Black Friday**, os modelos globais (LightGBM, LSTM) **subestimam a demanda em ~30%**. O **Prophet é o melhor** nesse mês (19,8% de erro, viés de −11%), graças ao tratamento explícito das datas comemorativas.
- **Em lançamentos** com pouco histórico, a vantagem se inverte: os modelos **globais** erram 14–18%, contra 52–63% dos modelos **por série** (ARIMA, Prophet).
- O **ETS foi o pior** e foi descartado: sem variáveis externas, não enxerga promoção nem evento, e no Q4 foi o que mais errou (38,9%).
- **Recomendação:** **LightGBM como modelo principal** (preciso, explicável por SHAP exato, serve de simulador de promoções, estável e rápido); **LightGBM + Prophet nas semanas de pico**; **LSTM como candidato** à próxima versão, após validação em mais de um Q4.

---

## 1. Protocolo de comparação

| Item | Definição |
|---|---|
| Unidade de previsão | SKU × canal × semana (40 séries: 20 SKUs × Loja/Venda Direta) |
| Alvo | **Demanda** = quantidade vendida + quantidade perdida por ruptura (evita perpetuar a falta de estoque) |
| Horizonte | 4 semanas à frente |
| Informação disponível | Cada modelo usa apenas dados anteriores à semana de origem. Preço, tipos de desconto, promoção dos irmãos e calendário futuros são tratados como **conhecidos** (são planejados pelo time de RGM) |
| Validação | **Backtest com origem móvel** em dois períodos: **mai–out/26** (6 janelas de 4 semanas, 929 previsões) e **Q4/25** (out/25–jan/26, 4 janelas, 548 previsões, com BF e Natal) |
| Métricas | **WAPE** = Σ\|real − previsto\| ÷ Σ real (erro em % do volume); **viés** = Σ(previsto − real) ÷ Σ real (negativo = subestima, risco de ruptura); **cobertura** da faixa de 80% |

**Por que dois períodos?** O primeiro backtest (mai–out) não continha nenhum Q4, mas a previsão final cai em novembro, com Black Friday. Validar o modelo numa época e usá-lo em outra esconderia justamente o comportamento nos picos. O backtest do Q4/25 tem uma limitação estrutural: a base começa em out/24, então os modelos **viram no máximo uma Black Friday** no treino, e no caso do LightGBM essa BF ainda cai no período de "aquecimento" dos lags.

---

## 2. Resultados

### 2.1 Visão geral

| Modelo | WAPE mai–out | Viés mai–out | WAPE Q4 | Viés Q4 | WAPE promoções (mai–out) | Viés nas datas comemorativas (Q4) | Cobertura 80% (mai–out / Q4) |
|---|---|---|---|---|---|---|---|
| **LSTM** | **15,2%** | +0,3% | **19,0%** | −6,7% | **16,6%** | −14,3% | 78% / 62% |
| **LightGBM** | 15,5% | −0,2% | 21,5% | −5,6% | 17,5% | −15,7% | 70% / 56% * |
| Prophet | 17,8% | −0,9% | 20,9% | −1,1% | 20,6% | **−2,1%** | 73% / 64% |
| ARIMA (com regressores) | 17,9% | −2,0% | 24,0% | −2,1% | 20,6% | −7,3% | **81%** / **69%** |
| ETS (descartado) | 21,7% | +1,5% | 38,9% | −2,7% | 25,5% | −37,7% | 88% / 67% |
| **LightGBM + LSTM** | **14,7%** | −0,1% | 19,2% | −6,5% | 15,7% | −15,2% | — |
| **LightGBM + Prophet** | 15,8% | −0,9% | **19,1%** | −4,8% | 17,9% | −11,3% | — |

\* Faixa crua do LightGBM. Após a calibração conformal, a cobertura vai para 80% (mai–out) e 77% (Q4), medida na própria amostra de calibração.

> **Nota sobre a variação do LSTM:** redes neurais dão resultados ligeiramente diferentes a cada treino. Com a média de 3 redes e repetindo o experimento com outras sementes, o WAPE do LSTM variou de 15,2% a 15,5% (mai–out) e de 18,9% a 19,3% (Q4). As tabelas por janela abaixo vêm de uma execução independente, então os números do LSTM podem diferir ~0,5 p.p. dos da tabela acima.

### 2.2 Por janela do Q4: onde cada modelo erra

**WAPE por janela de 4 semanas (Q4/25):**

| Janela | ARIMA | ETS | Prophet | LSTM | LightGBM | Média 8s | LGBM + Prophet | LGBM + LSTM |
|---|---|---|---|---|---|---|---|---|
| out/25 (pré-BF) | 17,8% | 20,4% | 17,1% | 16,1% | **15,3%** | 17,9% | 15,4% | **14,9%** |
| nov/25 (**Black Friday**) | 23,4% | 34,3% | **19,8%** | 29,4% | 32,7% | 33,4% | 25,2% | 31,1% |
| dez/25 (**Natal**) | 25,6% | 38,8% | 22,5% | **14,3%** | 19,2% | 36,9% | 17,9% | **14,7%** |
| jan/26 (ressaca) | 27,9% | 62,0% | 23,2% | **15,8%** | 16,3% | 34,6% | 16,4% | **15,2%** |

**Viés por janela (Q4/25):**

| Janela | ARIMA | ETS | Prophet | LSTM | LightGBM | Média 8s |
|---|---|---|---|---|---|---|
| out/25 | +4,7% | +3,4% | +4,3% | +7,5% | +4,1% | +4,3% |
| nov/25 (BF) | −12,5% | −30,1% | **−11,3%** | −28,2% | −30,9% | −28,2% |
| dez/25 (Natal) | −6,6% | −20,1% | **−2,2%** | **−0,4%** | +9,0% | −33,0% |
| jan/26 | +11,9% | **+54,3%** | +8,5% | **−2,2%** | −4,8% | +27,3% |

**Leitura:**
- **Novembro é o mês crítico.** Os modelos globais (LightGBM e LSTM) erram ~30% e subestimam a demanda em ~30%. O Prophet e o ARIMA, que tratam as datas comemorativas como **efeitos explícitos** (feriados e regressores), subestimam bem menos (−11% a −13%).
- **Em dezembro e janeiro**, os globais voltam a liderar. O LSTM é o mais preciso no Natal (14,3%) e acerta a "ressaca" de janeiro (viés de −2%).
- **O ETS desmonta em janeiro** (+54% de viés): ele "carrega" o patamar alto de dezembro para janeiro, porque só olha o nível recente. É o exemplo mais claro do custo de ignorar o calendário.

### 2.3 Por categoria

| Período | Categoria | ARIMA | ETS | Prophet | LSTM | LightGBM | Média 8s |
|---|---|---|---|---|---|---|---|
| mai–out | CAT_A (promocional, elástica) | 26,5% | 36,0% | 25,8% | 19,9% | **19,3%** | 35,8% |
| mai–out | CAT_B | 15,5% | 17,2% | 15,6% | **14,2%** | 14,6% | 17,0% |
| mai–out | CAT_C | 13,6% | 15,8% | 13,9% | **11,8%** | 13,1% | 15,3% |
| Q4 | CAT_A | 32,2% | 42,3% | 25,9% | **22,5%** | 22,8% | 40,9% |
| Q4 | CAT_B | 21,1% | 37,5% | 19,4% | **18,2%** | 22,0% | 28,5% |
| Q4 | CAT_C | 20,7% | 38,3% | 17,4% | **15,7%** | 16,8% | 28,4% |

A **CAT_A**, mais promocional, elástica a preço e com canibalização forte entre irmãos, é onde a distância é maior: os modelos globais erram ~19% contra ~26% dos modelos por série. É a categoria em que **saber da promoção planejada e da promoção do irmão** faz mais diferença, e só os modelos globais usam a promoção do irmão.

### 2.4 Lançamentos (*cold start*)

| Período | Grupo | ARIMA | ETS | Prophet | LSTM | LightGBM | Média 8s |
|---|---|---|---|---|---|---|---|
| mai–out | SKUs maduros | 18,3% | 22,4% | 17,7% | **15,6%** | 15,8% | 22,6% |
| mai–out | Lançamentos (SKU_02, 07, 13, 19) | 15,7% | 17,4% | 18,4% | **12,9%** | 13,7% | 14,9% |
| Q4 | SKUs maduros | 21,7% | 37,4% | **19,0%** | 19,3% | 21,7% | 31,7% |
| Q4 | Lançamentos (SKU_07, 19, com poucas semanas) | 62,8% | 64,0% | 51,7% | **13,6%** | 18,2% | 34,4% |

No Q4/25, os lançamentos tinham poucas semanas de história. **Modelos por série (ARIMA, ETS, Prophet) praticamente não têm o que aprender**, e o erro passa de 50%. **Modelos globais aprendem com os outros SKUs** como uma série reage à promoção e ao calendário, e erram 14–18%. Essa é uma das vantagens mais concretas do modelo global para um portfólio com inovação frequente.

### 2.5 Por canal

Em mai–out, a diferença de erro entre Loja e Venda Direta é de no máximo ~1 p.p. em qualquer modelo. No Q4, ela chega a ~2–3 p.p. (5 p.p. no ETS). A ordem entre os modelos se mantém nos dois canais, então **não há motivo para escolher modelos diferentes por canal**.

### 2.6 Custo computacional (backtest completo, 2 CPUs)

| Modelo | mai–out (6 janelas) | Q4 (4 janelas) | Observação |
|---|---|---|---|
| LightGBM | ~7 s | ~4 s | 1 modelo global (3 com os quantis) |
| LSTM | ~87 s | ~33 s | 3 redes × janela; ~10–15× o LightGBM |
| Prophet | ~63 s | ~40 s | 40 ajustes por janela (Stan) |
| ARIMA | ~128 s | ~60 s | busca automática de ordens em 40 séries por janela |
| ETS | ~2 s | ~1 s | trivial |

Para um portfólio real (milhares de SKUs), os modelos **por série** escalam linearmente com o número de séries. Os **globais** escalam com o volume total de dados, em um único treino.

---

## 3. Propriedades, vantagens e desvantagens de cada modelo

### 3.1 ARIMA com regressores (AutoARIMA, `statsforecast`)

**Propriedades:** modelo estatístico **local** (um por série). A demanda é uma combinação linear dos valores passados (parte AR), dos erros passados (parte MA) e das variáveis externas (preço relativo, tipos de desconto, datas comemorativas). O AutoARIMA escolhe as ordens por critério de informação. Por não haver 2 ciclos anuais completos, a sazonalidade semanal não foi modelada como componente SARIMA, e o calendário entrou só via dummies de evento.

| Vantagens neste case | Desvantagens neste case |
|---|---|
| **Faixa de confiança mais bem calibrada** de todos (81% de cobertura para 80% prometidos em mai–out; 69% no Q4, também a maior) | Efeitos **lineares e fixos**: um pague-e-leve tem o mesmo efeito em março e no Natal; não há interação com canal ou com a promoção do irmão |
| Usa promoção e eventos: no mês da BF, subestima bem menos que os modelos globais (−12,5% contra ~−30%) | Cada série estima sozinha o efeito de BF a partir de **uma** observação: estimativas instáveis |
| Base teórica sólida, coeficientes interpretáveis, intervalos analíticos | **Inviável em lançamentos** (63% de erro no Q4): sem histórico, não há o que modelar |
| Comportamento previsível e auditável | **O mais lento** do estudo; frágil com regressores sem variação (exigiu remover colunas série a série) |
| | Erro 2–3 p.p. maior que os globais no ano, e muito maior na CAT_A (26% contra 19%) |

**Quando usar:** como referência estatística e "segundo parecer" para o planejador, especialmente quando a prioridade é uma faixa de incerteza honesta.

### 3.2 ETS, suavização exponencial (AutoETS, `statsforecast`), descartado

**Propriedades:** modelo **local** que atualiza recursivamente nível, tendência e sazonalidade, com peso exponencialmente maior no passado recente. **Não aceita variáveis externas.** Com menos de 2 anos de dados, não há como estimar sazonalidade semanal (52 parâmetros).

| Vantagens | Desvantagens neste case |
|---|---|
| Simples, rapidíssimo, quase sem decisões de modelagem | **Cego a promoção e evento**, que é exatamente o que RGM gerencia |
| Intervalos de previsão nativos | Na prática vira uma média ponderada do passado: 21,7% de erro em mai–out e **38,9% no Q4**, o pior de todos |
| Excelente para séries estáveis e longas | Subestima os picos em −38% e **erra +54% em janeiro**, porque carrega o patamar do Natal |
| | Faixa larga demais em mai–out (88% de cobertura): honesta por ser vaga |

**Quando usar:** itens de linha, sem promoção e com vários anos de histórico. **Não adequado a RGM.**

### 3.3 Prophet (`prophet`)

**Propriedades:** modelo **local**, aditivo e decomponível: tendência linear por partes + sazonalidade anual (Fourier) + efeitos de **feriados** com janela configurável + regressores lineares. Ajustado em log, o que torna os efeitos multiplicativos. Aqui, as datas comemorativas (BF, Natal com a véspera, Dia das Mães, Dia dos Namorados) entraram como feriados, e o preço relativo, o combo e o pague-e-leve como regressores.

| Vantagens neste case | Desvantagens neste case |
|---|---|
| **O melhor nos picos:** no mês da BF, erro de 19,8% e viés de −11%, contra ~30% dos modelos globais; no Q4 inteiro, o menor viés nas datas comemorativas (−2%) | **Fraco em promoções** (20,6% contra 16–17% dos globais): os regressores são aditivos e não interagem com canal, evento ou promoção do irmão |
| Componentes (tendência, sazonalidade, feriados) fáceis de mostrar a não técnicos | **Faixa otimista** (cobre 64–73% para 80% prometidos) |
| Robusto a buracos e mudanças de patamar; pouca configuração | **Inviável em lançamentos** (52% de erro no Q4) |
| Ótimo parceiro de *ensemble*: combinado com o LightGBM, chega ao melhor erro do Q4 (19,1%) | Sazonalidade anual com < 2 anos é pouco confiável (desligada automaticamente nas séries curtas); 40 ajustes por janela |

**Quando usar:** planejamento de **semanas de pico** e como componente de *ensemble* com um modelo global.

### 3.4 LSTM (rede neural recorrente, PyTorch)

**Propriedades:** modelo **global** (uma rede para as 40 séries). Uma camada LSTM (32 unidades) lê as últimas 16 semanas de cada série (demanda relativa à média recente, preço, tipos de desconto, promoção dos irmãos e eventos), e uma camada densa combina essa "memória" com o plano das 4 semanas seguintes (preço, tipos de desconto, eventos) e com canal e categoria. A saída são **3 quantis** (P10, P50, P90) para cada semana, treinados com perda quantílica. A previsão final é a média de 3 redes com sementes diferentes.

| Vantagens neste case | Desvantagens neste case |
|---|---|
| **Menor erro individual** nos dois períodos (15,2% mai–out; 19,0% Q4); o melhor no Natal (14,3%) e em janeiro (15,8%) | **Também subestima a BF** (−28% no mês de novembro): uma única BF no histórico não basta para nenhum modelo aprendê-la |
| **Melhor em lançamentos** (13,6% no Q4): aprende com os demais SKUs | **Variância entre execuções** (±0,3–0,5 p.p.); exige média de várias redes |
| Lê a **sequência** (aceleração, ressaca) sem lags escolhidos à mão | **Explicabilidade aproximada**: o DeepSHAP falha na checagem de soma para LSTM em PyTorch (erro de −16%); o *Expected Gradients* funciona, mas é aproximado e depende da referência e da semente |
| **Faixa nativa mais bem calibrada** que a do LightGBM (78% contra 70% em mai–out) | **Hiperparâmetros não ajustados** (janela, tamanho, épocas): não há dados para uma validação separada, e o bom desempenho pode ser em parte sorte de configuração |
| Combinado com o LightGBM, dá o **menor erro do estudo** (14,7% em mai–out) | ~10–15× mais lento que o LightGBM; dependência de PyTorch; manutenção exige especialização |

**Por que funcionou, contra a expectativa inicial:** a rede herdou as três escolhas que fazem o LightGBM funcionar (**modelo global**, **alvo relativo ao nível recente** e **calendário futuro de promoções e eventos**) e foi mantida pequena. Uma LSTM "pura", só com o histórico de cada série, provavelmente se comportaria como o ETS. A lição é que **o desenho do problema pesou mais que o algoritmo**.

**Quando usar:** candidato forte a substituir ou complementar o LightGBM, **depois de** validar em mais de um Q4 e de ter uma forma de explicação aceita pelo negócio.

### 3.5 LightGBM (*gradient boosting* de árvores)

**Propriedades:** modelo **global** de árvores de decisão em sequência. Alvo: log da demanda **relativa à média das 8 últimas semanas conhecidas** (aprende multiplicadores, como "BF ≈ ×1,5", em vez de níveis). Features: lags ≥ 4 semanas (sem vazamento), médias e volatilidade recentes, ruptura recente, idade da série, preço relativo, tipos de desconto, promoção dos irmãos e de outras subcategorias, calendário, identidade do SKU/canal/categoria. Faixa de confiança por **regressão quantílica** (P10/P90) com **calibração conformal** (folga maior no Q4 e em datas comemorativas).

| Vantagens neste case | Desvantagens neste case |
|---|---|
| Empata com o LSTM em mai–out (15,5%) e é **o melhor na CAT_A** (19,3%) | **Subestima o mês da BF** (−31%), o pior viés entre os modelos com covariáveis: só viu uma BF útil, e árvores não extrapolam além do que viram |
| **Explicabilidade exata** com TreeSHAP: "parte da média recente de 222 un.; BF +52%; sem promoção −7%" | **Faixa crua otimista** (70% / 56%): precisa de calibração conformal |
| **Simulador de RGM**: basta mudar preço, tipo de desconto ou calendário na entrada | Depende de engenharia de features e de cuidado com vazamento |
| **Rápido e determinístico**: o backtest inteiro roda em segundos, sempre com o mesmo resultado | Mede associação, não causalidade: para precificar, usar as elasticidades econométricas da EDA |
| Bom em lançamentos (13,7–18,2%) e com a canibalização (usa a promoção dos irmãos) | "Caixa mais preta" para o negócio que ARIMA ou Prophet, embora bem menos que o LSTM, graças ao SHAP |

**Quando usar:** **modelo principal** de previsão e simulação.

### 3.6 Combinações (*ensembles*)

**Propriedades:** média geométrica das previsões P50 de dois modelos. Funciona porque os modelos erram em semanas diferentes.

| Combinação | Resultado | Leitura |
|---|---|---|
| **LightGBM + LSTM** | **14,7%** mai–out (melhor do estudo); 19,2% Q4 | Dois globais com arquiteturas diferentes: os erros se compensam no dia a dia, mas compartilham a fraqueza na BF |
| **LightGBM + Prophet** | 15,8% mai–out; **19,1%** Q4; viés de −11% nas datas comemorativas | O Prophet "puxa" o LightGBM para cima nos picos (no mês da BF: 25,2% contra 32,7% do LightGBM sozinho) |

**Desvantagem comum:** dois modelos para manter, monitorar e explicar. A explicação passa a ser "metade de cada um".

---

## 4. Matriz de decisão

| Critério | ARIMA | ETS | Prophet | LSTM | LightGBM |
|---|---|---|---|---|---|
| Precisão no ano (mai–out) | ●●○○ | ●○○○ | ●●○○ | ●●●● | ●●●● |
| Precisão no Q4 | ●●○○ | ○○○○ | ●●●○ | ●●●● | ●●●○ |
| Picos (mês da BF) | ●●●○ | ●○○○ | ●●●● | ●○○○ | ●○○○ |
| Promoções / CAT_A | ●○○○ | ○○○○ | ●○○○ | ●●●● | ●●●● |
| Lançamentos (*cold start*) | ○○○○ | ○○○○ | ○○○○ | ●●●● | ●●●○ |
| Calibração da faixa de 80% | ●●●● | ●●○○ (larga) | ●●○○ | ●●●○ | ●●○○ (crua) / ●●●● (calibrada) |
| Explicabilidade para o negócio | ●●●○ | ●●●● | ●●●● | ●○○○ | ●●●○ |
| Uso como simulador de promoção | ●●○○ | ○○○○ | ●●○○ | ●●●○ | ●●●● |
| Custo e manutenção | ●●○○ | ●●●● | ●●○○ | ●○○○ | ●●●● |
| Estabilidade (reprodutibilidade) | ●●●● | ●●●● | ●●●● | ●●○○ | ●●●● |

---

## 5. Recomendação

| Uso | Modelo | Justificativa |
|---|---|---|
| Previsão do dia a dia e simulação de promoções | **LightGBM** | Precisão equivalente à melhor, explicação exata (SHAP), simulador, rápido e estável |
| Semanas de pico (Q4, datas comemorativas) | **LightGBM + Prophet** | Empata com as melhores opções no Q4 e é a que menos subestima os picos |
| Lançamentos com < 6 meses de histórico | LightGBM (marcado como "atenção" no semáforo) | Modelos por série não funcionam nesse caso |
| Referência para o planejador | ARIMA | Faixa de incerteza mais bem calibrada |
| Próxima versão | **LSTM ou LightGBM + LSTM** | Menor erro do estudo; falta validar em mais de um Q4 e resolver a explicação |
| Não usar | ETS | Ignora promoção e calendário |

**Para a semana de Black Friday:** planejar pela coluna de **combinação LightGBM + Prophet** e dimensionar o estoque pelo **máximo provável** da faixa, já que todos os modelos tendem a subestimar o pico.

---

## 6. Limitações e próximos passos

**Limitações do estudo:**
1. **Um único Q4 de validação**, com uma única BF anterior no treino. As conclusões sobre picos são indícios fortes, não certezas.
2. **Hiperparâmetros não ajustados** em nenhum modelo, para não "olhar o teste". Com mais dados, um ajuste com validação temporal aninhada seria o correto.
3. **Diferenças de 1–2 p.p.** entre os melhores modelos estão dentro da variação esperada (veja a variação do LSTM entre sementes).
4. **Cenário de covariáveis futuras:** os modelos assumem que o plano de promoções é conhecido e cumprido. Se o plano mudar, a previsão muda.

**Próximos passos:**
- Retestar com **3+ anos de histórico** (mais de uma BF no treino) e validar no Q4/26 real.
- **Ajuste de hiperparâmetros** com validação temporal aninhada (LightGBM e LSTM).
- **Modelos diretos por horizonte** (t+1 com lag 1 etc.) para ganhar precisão nas semanas mais próximas.
- **Reconciliação hierárquica** (SKU → subcategoria → categoria).
- Em produção: **monitorar WAPE, viés e cobertura por janela** (*concept drift*), as distribuições de preço e mix de tipos de desconto (*data drift*), e retreinar por gatilho.
