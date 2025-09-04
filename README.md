# PCA → MSVM Pipeline (Vang / Synchrophasor)

**Autor:** Bryan Ambrósio  
**Data:** 2025-09-04  
**Versão:** 0.1.0  
**Licença:** MIT (sugerida)

Pipeline reproduzível para **EDA → PCA(2D) → MSVM linear por pares adjacentes** aplicado a *Vang* (ângulos de tensão) em quatro instantes (0.18s, 0.25s, 0.30s, 0.35s) e dois pontos de medição (XES e RioVTEST). 
Um **orquestrador `main.py`** executa os três scripts na ordem, aguardando 1s entre eles e **pausando no final** quando aberto por duplo-clique no Windows.

---

## Sumário

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Como Rodar](#como-rodar)
- [Expectativas de Dados](#expectativas-de-dados)
- [Descrição dos Scripts](#descrição-dos-scripts)
  - [1) `scripts/1-EDA.py` — Limpeza & Diagnóstico](#1-scripts1-edapy--limpeza--diagnóstico)
  - [2) `scripts/2-PCA.py` — Redução de Dimensão (PC1, PC2)](#2-scripts2-pcapy--redução-de-dimensão-pc1-pc2)
  - [3) `scripts/3-MSVM_com_ponto.py` — MSVM Linear por Pares](#3-scripts3-msvm_com_pontopy--msvm-linear-por-pares)
- [Orquestrador — `main.py`](#orquestrador--mainpy)
- [Saídas (onde os arquivos são gravados)](#saídas-onde-os-arquivos-são-gravados)
- [Reprodutibilidade & Dicas](#reprodutibilidade--dicas)
- [Solução de Problemas](#solução-de-problemas)
- [Licença](#licença)
- [Agradecimentos](#agradecimentos)

---

## Estrutura do Projeto

```text
project_root/
  main.py
  scripts/
    1-EDA.py
    2-PCA.py
    3-MSVM_com_ponto.py
  data_raw/
    tabelaestabilidade_minUG.xlsx
  data_cleaned/
  data_PCA/
  PCA_visualization/
  results_msvm/
  best_treshold_evaluation/
```
## Requisitos

Python 3.10+

## Bibliotecas:

numpy, pandas, matplotlib, scikit-learn, joblib

(opcional) pyarrow ou fastparquet para Parquet

## Instalação recomendada:
```text
pip install numpy pandas matplotlib scikit-learn joblib pyarrow
```
## Como Rodar

### Duplo-clique (Windows)

Basta dar duplo-clique em main.py

O terminal espera um pressionamento de tecla ao final

Expectativas de Dados

Entrada esperada: data_raw/tabelaestabilidade_minUG.xlsx

Coluna-alvo: min_UGs ∈ {0, 1, ..., 7}

Features obrigatórias (8 Vang):

Vang_XES_0.18s, Vang_XES_0.25s, Vang_XES_0.30s, Vang_XES_0.35s

Vang_RioVTEST_0.18s, Vang_RioVTEST_0.25s, Vang_RioVTEST_0.30s, Vang_RioVTEST_0.35s

Nomes longos como Vang_XES_018019s são aceitos e renomeados automaticamente

## Descrição dos Scripts
### 1) scripts/1-EDA.py — Limpeza & Diagnóstico

Lê o Excel com separador decimal ,

Remove:

Linhas com min_UGs inválido

Linhas com Estabilidade == "Instavel" e min_UGs == 0 (se aplicável)

Linhas com qualquer Vang negativo

Renomeia colunas longas → curtas

Salva:

CSV e Parquet em data_cleaned/

Gráficos em feature_vs_target_pre_cleaning/ e feature_vs_target_pos_cleaning/

### 2) scripts/2-PCA.py — Redução de Dimensão (PC1, PC2)

Detecta automaticamente os 8 Vang

Split train/test estratificado

Aplica StandardScaler + PCA(2D) apenas no treino

Salva:

train_pca.csv, test_pca.csv, all_pca.csv em data_PCA/

scaler.joblib, pca.joblib, features_used.txt

Figuras em PCA_visualization/

### 3) scripts/3-MSVM_com_ponto.py — MSVM Linear por Pares

Treina LinearSVC para cada par de classes adjacentes (0↔1, 1↔2, ..., 6↔7)

Para cada modelo:

Calcula decision_function

Define threshold como o menor score positivo da classe positiva (evita FN)

Avalia e salva:

thresholds.csv, metrics_train.csv, metrics_test.csv

Modelos .joblib por par

Figuras com as fronteiras, falsos negativos e ponto capturado (se aplicável)

parametrização/equacoes_fronteiras_vang.txt com as equações no espaço original

### Orquestrador — main.py

Executa os três scripts em sequência

Aguarda 1s entre cada script

Usa cwd=root para garantir caminhos corretos

Com --no-pause, não pausa ao final (útil para CI)

Saídas (onde os arquivos são gravados)

Dados limpos: data_cleaned/

PCA e artefatos: data_PCA/, PCA_visualization/

Modelos e métricas: results_msvm/

Equações originais: parametrização/equacoes_fronteiras_vang.txt

Figuras: results_msvm/plots/, best_treshold_evaluation/, caso_base_evaluation/

## Reprodutibilidade & Dicas

Sementes fixas nos splits

Parquet opcional — sempre há fallback para CSV

features_used.txt define a ordem das features, essencial para projeção

Pares com classes com poucos exemplos podem ser ignorados com aviso

Solução de Problemas

Erro de colunas: verifique os nomes no Excel e os logs do 1-EDA.py

Erro ao projetar ponto capturado: verifique se o dict tem as mesmas chaves de features_used.txt

Parquet falhando: instale pyarrow ou fastparquet
  caso_base_evaluation/
  parametrização/
