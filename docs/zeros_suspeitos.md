# Arquivo de apoio: os 71 “zeros suspeitos”

Documento de apoio ao slide 3 da apresentação e à seção 1.6 do notebook `notebooks/case_rgm_v2.ipynb`. Explica como os zeros foram encontrados, como foi testada a plausibilidade de cada um e por que foram tratados como dado faltante.

---

## Resumo

- A base tem **79 semanas com venda zero**. Em **8** delas houve ruptura (faltou estoque), e esses zeros são reais: foram mantidos.
- Nas outras **71**, o SKU não vendeu nada **sem** ruptura, **sem** promoção e **sem** receita, em séries que vendem normalmente de 8 a 306 unidades por semana.
- Pelo teste de plausibilidade, a chance de cada um desses zeros acontecer por acaso fica entre **0,03%** (Poisson) e **2,3%** (Binomial Negativa, mais conservadora). Na base inteira, o acaso explicaria **~1 zero**, mas há **71**.
- Os zeros se espalham **aleatoriamente** no tempo e entre canais, e não há sinal de venda migrando para outro canal ou para a semana seguinte.
- **Conclusão:** são falhas pontuais de registro. Foram convertidos em “sem informação” (NaN, com flag), retirados do alvo do modelo e interpolados apenas para calcular as médias recentes.

---

## 1. O problema: quando um zero não é um zero

Uma semana com venda zero pode significar três coisas diferentes:

| Situação | O zero é… | Exemplo |
|---|---|---|
| **Ruptura total** | real: faltou produto | `qt_ruptura > 0` |
| **Baixa demanda** | real: ninguém quis comprar | plausível em itens de giro muito baixo |
| **Registro faltante** | falso: o sistema não carregou a semana | gravado como zero por padrão |

Confundir o terceiro caso com os dois primeiros tem custo. O modelo aprenderia quedas de demanda que não existem, a média recente das séries ficaria artificialmente baixa e a previsão tenderia a subestimar, o que, no planejamento, vira ruptura.

## 2. Passo 1: identificação

Critério: `qt_vendida == 0`. Em seguida, a separação pelo campo de ruptura.

| Grupo | Semanas | Tratamento |
|---|---|---|
| Zero **com** ruptura (`qt_ruptura > 0`) | 8 | **Mantidos**: são demanda censurada real, e a demanda da semana = ruptura |
| Zero **sem** ruptura | **71** | Investigados nos passos seguintes |

Características dos 71:
- **Nenhum** tinha promoção ativa (desconto direto, combo ou pague-e-leve).
- A receita era **zero** em 61 casos e **vazia** em 10. Preço e desconto também vinham vazios, o que é coerente com “nenhuma transação registrada”.
- Aparecem em **15 dos 20 SKUs**, nos dois canais. Os mais afetados: SKU_01 (12), SKU_05 (7), SKU_09 (7), SKU_10 (6) e SKU_18 (6).

## 3. Passo 2: plausibilidade (“esse zero poderia acontecer por acaso?”)

### 3.1 O nível normal de cada série

Para cada zero, estimei quanto aquele SKU, naquele canal, costuma vender **naquela época**: a **mediana das vendas nas 4 semanas antes e nas 4 depois** (janela centrada de 9 semanas), **ignorando os próprios zeros**. A mediana não é distorcida por picos de promoção ou de datas comemorativas.

| Estatística do nível normal (71 zeros) | Unidades por semana |
|---|---|
| Mínimo | 8 (SKU_04, Loja) |
| 25% | 19,5 |
| Mediana | 28 |
| 75% | 68 |
| Máximo | 306 (SKU_01) |

### 3.2 A chance de zero por acaso: Poisson

Vendas semanais são **contagens**. O modelo mais simples para contagens é a distribuição de **Poisson**: se a venda esperada é λ unidades, a chance de não vender nenhuma é

> P(0) = e^(−λ)

| Venda normal (λ) | Chance de zero por acaso |
|---|---|
| 8 un./semana (o caso mais fraco) | 0,03% |
| 28 un./semana (a mediana) | menos de 1 em 1 trilhão |
| 225 un./semana (SKU_01) | praticamente zero |

**Intuição:** é como uma loja que vende ~28 unidades toda semana e, numa semana qualquer, sem faltar produto e sem nada de diferente, não vende nenhuma. Não é impossível, mas é tão improvável que uma falha de registro explica muito melhor.

### 3.3 Teste de robustez: Binomial Negativa

A Poisson assume que a variância da venda é igual à média. A venda real costuma variar **mais** que isso (sobredispersão), e com mais variabilidade os zeros ficam mais prováveis. Para não depender dessa premissa:

1. Medi a dispersão (variância ÷ média) de cada série **nas semanas normais**, sem promoção e sem ruptura. Ela vai de 0,2 a 12,4 (mediana 1,0).
2. Recalculei P(0) com uma **Binomial Negativa** que usa a dispersão de cada série.

| Medida | Poisson | Binomial Negativa |
|---|---|---|
| Maior P(0) entre os 71 zeros | 0,03% | **2,3%** (SKU_05 Loja, a série mais volátil) |
| Zeros esperados **por acaso** em toda a base (semanas sem promoção e sem ruptura) | ~0,07 | **~1** |
| Zeros **observados** | 71 | 71 |

Mesmo com a premissa mais conservadora, o acaso explicaria cerca de **1** zero, e a base tem **71**. Os zeros não vêm da variação normal da demanda.

## 4. Passo 3: padrões (“é um problema sistemático?”)

Se os zeros viessem de algo sistemático, como uma troca de sistema numa data, um canal fora do ar, venda migrando de canal ou lançamento atrasado, deixariam rastro. Procurei quatro rastros:

**a) Concentração no tempo:** nenhuma.

| Trimestre | 24T4 | 25T1 | 25T2 | 25T3 | 25T4 | 26T1 | 26T2 | 26T3 | 26T4 |
|---|---|---|---|---|---|---|---|---|---|
| Loja | 1 | 4 | 3 | 4 | 4 | 7 | 4 | 5 | 3 |
| Venda Direta | 1 | 1 | 5 | 5 | 3 | 9 | 5 | 6 | 1 |

Um teste qui-quadrado compara essa distribuição com a esperada se os zeros caíssem ao acaso, na proporção do número de linhas de cada trimestre: **p ≈ 0,30**, sem evidência de concentração. Os 71 zeros se distribuem por **50 semanas diferentes**.

**b) Migração entre canais:** não há. Na semana em que o SKU zera num canal, **o outro canal vende normalmente** (1,0× o seu nível, na mediana). Se a venda tivesse migrado, o outro canal estaria acima do normal. Só em **2 semanas** o mesmo SKU zerou nos dois canais ao mesmo tempo.

**c) Venda represada:** não há. Na **semana seguinte** ao zero, o SKU vende o normal (1,0×). Se a venda tivesse sido lançada com atraso, a semana seguinte estaria acima do normal.

**d) Ligação com promoção, evento ou ruptura:** nenhuma. Nenhum zero tinha promoção; a quantidade em datas comemorativas é a esperada pelo acaso (6, contra ~6,6 esperados).

**Leitura:** o padrão é de **ausência completamente aleatória** (MCAR, *missing completely at random*): falhas pontuais, sem relação com o tempo, o canal ou o próprio valor que falta. Isso é importante para o tratamento: quando a ausência é aleatória, **retirar** essas semanas do alvo **não introduz viés**.

## 5. Passo 4: tratamento e por quê

### 5.1 O que foi feito

| Ação | Detalhe |
|---|---|
| `qt_vendida` e `receita` viram **NaN** | O valor é desconhecido, não zero |
| Flag `flag_zero_suspeito = True` | Permite refazer qualquer análise com ou sem o tratamento |
| **Fora do alvo** do modelo | Semanas sem alvo não entram no treino nem na avaliação: nenhum valor é inventado para ser “previsto” |
| **Interpolação** só para features | `qt_vendida_interp` preenche o buraco linearmente dentro da série, apenas para calcular lags e médias recentes, que não podem ter buracos |
| Zeros **com** ruptura **mantidos** | São demanda censurada real; a demanda dessas semanas é a quantidade de ruptura |

### 5.2 Por que não manter como zero

- **Distorce o nível das séries:** nas 28 séries afetadas, os zeros baixam a média de venda em **2,7%** na mediana e em até **7,6%** (SKU_01, Venda Direta).
- **Contamina a “média recente”**, que é a base do alvo relativo do LightGBM e do LSTM: por até 8 semanas depois de cada zero, a previsão partiria de um patamar errado.
- **Ensina quedas que não existem:** o modelo tentaria “explicar” um colapso de demanda sem causa, o que piora as previsões e as explicações (SHAP).
- **Viés para baixo vira ruptura:** subestimar a demanda leva a comprar pouco e faltar produto.

### 5.3 Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Manter como zero | Distorções da seção 5.2 |
| Excluir as linhas da base | Quebra a grade semanal: o “lag de 4 semanas” passaria a apontar para outra data |
| Imputar o alvo (média, interpolação) | Criaria valores inventados que o modelo seria cobrado para acertar, inflando artificialmente as métricas |
| Tratar como demanda censurada (como a ruptura) | Não há informação de quanto se perdeu; a ruptura é registrada, o zero suspeito não |
| **NaN + flag, fora do alvo, interpolado só nas features** | **Escolhida:** honesta (não inventa alvo), preserva a grade e é reversível |

## 6. Limitações

- **Zeros reais isolados não podem ser descartados por completo.** Um SKU pode ter ficado sem exposição numa loja ou sido retirado temporariamente da vitrine da consultora. Para uma semana isolada, porém, isso seria indistinguível de uma falha de registro com os dados disponíveis. A pergunta a levar ao negócio é: *“existe algum evento operacional que zere a venda de um SKU num canal sem gerar ruptura registrada?”*
- **O nível normal é uma estimativa:** a janela de 9 semanas pode ser afetada por promoções vizinhas. Usar a mediana, e não a média, reduz esse efeito, e o teste de robustez cobre a variabilidade.
- **A conclusão MCAR é estatística, não comprovada:** confirmar com a área de dados que houve falhas de carga nessas datas fecharia a questão.

## 7. Onde está no código

| Etapa | Notebook (`case_rgm_v2.ipynb`) | Colunas |
|---|---|---|
| Identificação e teste de Poisson | Seção 1.6 | `flag_zero_suspeito` |
| Padrões no tempo e entre canais | Seção 1.6.1 | — |
| Interpolação para features | Seção 1.7 | `qt_vendida_interp`, `demanda_interp` |

O teste de robustez com Binomial Negativa (seção 3.3 deste documento) foi feito à parte, para esta nota, com o mesmo critério de nível normal.

## 8. Em uma frase (para a apresentação)

> “Antes de tratar zero como zero, testei se ele era plausível: em séries que vendem de 8 a 306 unidades por semana, sem ruptura nem promoção, o acaso explicaria cerca de 1 zero, e havia 71, espalhados aleatoriamente no tempo e entre canais. São falhas de registro: virei-os ‘sem informação’, fora do alvo do modelo.”
