# Case RGM: previsão de demanda e análise promocional

Case técnico para a vaga **Data Science Especialista I, RGM**, no **Grupo Boticário**: análise exploratória de preço, promoção e canibalização, e previsão da demanda semanal por SKU e canal, com indicador de confiabilidade compreensível para áreas não técnicas.

## Entregas do case

| Pedido | Onde está |
|---|---|
| 1. Análise exploratória com insights (elasticidade promocional, canibalização e outros) | `notebooks/case_rgm_v2.ipynb`, seções 1 e 2 |
| 2. Modelo preditivo da quantidade vendida (recorte e horizonte próprios) | `notebooks/case_rgm_v2.ipynb`, seção 3 · `docs/relatorio_modelos.md` |
| 3. Explicabilidade, métricas e grau de confiabilidade das previsões | `notebooks/case_rgm_v2.ipynb`, seção 4 (SHAP, semáforo de confiança) |

## Estrutura do repositório

```
case-boticario-rgm/
├── README.md                          ← este arquivo
├── requirements.txt                   ← dependências (versões fixadas)
├── .gitignore
├── .gitlab-ci.yml                     ← verificação do notebook + execução manual
├── data/
│   ├── raw/                           ← coloque aqui vendas_sku_semanal.csv (não versionado)
│   └── processed/                     ← base limpa gerada pelo notebook (não versionado)
├── notebooks/
│   ├── case_rgm_v2.ipynb              ← NOTEBOOK DO CASE
│   └── case_rgm_v2.py                 ← mesmo notebook em texto (jupytext), para revisão de código no GitLab
├── docs/
│   ├── relatorio_modelos.md           ← relatório detalhado: performance e propriedades de cada modelo
│   ├── modelos_vantagens_desvantagens.md  ← resumo de vantagens e desvantagens
│   └── zeros_suspeitos.md             ← apoio: os 71 zeros (método, plausibilidade, tratamento)
└── outputs/                           ← previsões e artefatos gerados (não versionado)
```

## Como executar

**Pré-requisitos:** Python 3.11+ (testado em 3.13).

```bash
git clone <url-do-repositorio>
cd case-boticario-rgm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# coloque a base do case em data/raw/vendas_sku_semanal.csv
jupyter lab notebooks/case_rgm_v2.ipynb
```

- A execução completa leva **~10 minutos** em 2 CPUs. A parte mais lenta é o backtest de ARIMA, Prophet e LSTM (seção 3.4).
- O notebook grava `data/processed/vendas_limpo.csv` e `outputs/previsao_4_semanas.csv`.

**Notebook pareado (jupytext):** o `.ipynb` tem um `.py` equivalente em formato *percent*. Edite qualquer um e sincronize com `jupytext --sync notebooks/case_rgm_v2.ipynb`. Assim, as revisões de *merge request* no GitLab mostram diffs legíveis do código, sem o JSON do notebook.

## Abordagem

### 1. Limpeza (Dia 1)
Problemas encontrados e tratados, todos registrados num log com flags por linha:
- datas em dois formatos;
- duplicatas exatas e conflitantes;
- quantidades com sinal invertido;
- preço com erro de casa decimal;
- desconto acima de 100% (recalculado pelo preço de tabela implícito);
- **71 "zeros suspeitos"**: semanas sem venda, sem ruptura e sem promoção em SKUs de giro alto. Pelo teste de Poisson, são registros faltantes e viraram NaN. A distribuição é aleatória no tempo e entre canais.

**Alvo:** `demanda = venda + ruptura`, para não perpetuar a venda censurada pela falta de estoque.

### 2. Análise exploratória
- **Sazonalidade:** Dia das Mães e Natal dobram a demanda, e a Black Friday a aumenta em ~50%. Outubro é um vale (o consumidor espera a BF), dezembro fica num platô e janeiro é ressaca.
- **Ruptura:** custa 2,5–3,7% da receita, aumenta com promoção (12% → 19% das semanas) e quase dobra nas datas comemorativas (24%).
- **Elasticidade** (regressão em painel log-log, com efeitos fixos de série e semana; 40 séries SKU × canal, out/24–out/26): a CAT_A é a única sensível a preço (−1,0 a −1,7); B e C ficam perto de zero.
- **ROI por tipo de desconto** (desconto direto, combo, pague e leve): nenhuma promoção se paga em lucro bruto semanal. Com margem de ~44% e desconto de ~28%, seria preciso +175% de volume.
- **Canibalização:** forte na CAT_A (−38% de demanda com os irmãos em promoção) e igual nos dois canais. Os irmãos fazem promoção em **rodízio**, o que torna a separação entre elasticidade própria e canibalização sensível ao modelo. Os lançamentos foram incrementais (diferenças-em-diferenças).
- **Outliers:** detecção robusta (MAD) com explicação por contexto. 2/3 das quedas atípicas coincidem com o irmão em promoção.

### 3. Modelagem
Quatro modelos comparados com o mesmo backtest (mai–out/26 e Q4/25), horizonte de 4 semanas:

| Modelo | WAPE mai–out | WAPE Q4 |
|---|---|---|
| LSTM | 15,2% | 19,0% |
| LightGBM | 15,5% | 21,5% |
| Prophet | 17,8% | 20,9% |
| ARIMA | 17,9% | 24,0% |

Vantagens, desvantagens e a análise por janela, categoria e lançamentos estão em [`docs/relatorio_modelos.md`](docs/relatorio_modelos.md).

**Escolha:** LightGBM como modelo principal (preciso, explicável, simulador de promoções); combinação LightGBM + Prophet nas semanas de pico; LSTM como candidato à próxima versão.

### 4. Explicabilidade e confiança
- **SHAP** (exato para o LightGBM), agrupado em "motivos" de negócio, gera uma frase por previsão, como *"média recente 222 un.; Black Friday +52%; sem promoção −7% → 353 un."*. Para o LSTM, uma análise com DeepSHAP e Expected Gradients mostra explicações coerentes com as do LightGBM, mas aproximadas.
- **Faixa provável** (P10–P90) com calibração conformal, mais larga no Q4.
- **Semáforo de confiança** binário (🔵 confiável · 🔴 atenção), que conta sinais de alerta (faixa larga, erro histórico alto, volatilidade, histórico curto, data comemorativa). Foi validado no backtest: com 2 ou mais alertas (atenção, ~40% das previsões) o erro é ~19%, contra ~13% nas confiáveis.

## Limitações principais
- Só **uma Black Friday** anterior nos dados. Todos os modelos subestimam o pico, então a previsão de BF deve ser planejada pelo **máximo provável**.
- A margem informada não reage ao desconto. A conclusão de ROI usa um custo unitário estimado e precisa ser validada com Finanças.
- Elasticidades e efeitos de promoção são observacionais. Para decisões de preço, o ideal são **testes controlados** (A/B por região ou canal).

## Autoria
Luisa, case técnico para o processo seletivo do Grupo Boticário.
