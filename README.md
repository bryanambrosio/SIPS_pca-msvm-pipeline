# PCA → MSVM Pipeline (Vang / Synchrophasor)

**Autor:** Bryan Ambrósio  
**Data:** 2025-09-04  
**Versão:** 0.1.0

Pipeline reproduzível para **EDA → PCA(2D) → MSVM linear por pares adjacentes** aplicado a *Vang* (ângulos de tensão) em quatro instantes (0.18s, 0.25s, 0.30s, 0.35s) e dois pontos de medição (XES e RioVTEST). 
Um **orquestrador `main.py`** executa os três scripts na ordem, aguardando 1s entre eles e **pausando no final** quando aberto por duplo-clique no Windows.

---

# Agredecimentos:

PPGEEL - UFSC

INESC P&D

---

# Estrutura de diretórios do Projeto

```text
project_root/
  main.py
  scripts/
    1-EDA.py
    2-PCA.py
    3-MSVM_com_ponto.py
  data_raw/
    tabelaestabilidade_minUG.xlsx
```
# Como Rodar

## Duplo-clique (Windows Explorer)

Basta dar duplo-clique em main.py

O terminal espera um pressionamento de tecla ao final

## Expectativas de Dados

Entrada esperada: data_raw/tabelaestabilidade_minUG.xlsx

Coluna-alvo: min_UGs ∈ {0, 1, ..., 7}

Features obrigatórias (8 Vang): Vang_XES_0.18s, Vang_XES_0.25s, Vang_XES_0.30s, Vang_XES_0.35s, Vang_RioVTEST_0.18s, Vang_RioVTEST_0.25s, Vang_RioVTEST_0.30s, Vang_RioVTEST_0.35s

Nomes longos como Vang_XES_018019s são aceitos e renomeados automaticamente

# O script main.py vai rodar os scripts abaixo:

---

## 1) scripts/1-EDA.py — Limpeza & Diagnóstico

Carrega a planilha bruta ../data_raw/tabelaestabilidade_minUG.xlsx, já tratando vírgula decimal (decimal=",").

Define o alvo e as features: o alvo é min_UGs (classes 0..7) e as oito features são ângulos de tensão (Vang) em quatro instantes (0.18s, 0.25s, 0.30s, 0.35s) para dois pontos (XES e RioVTEST), totalizando 8 colunas.

Reconhece nomes “longos” e renomeia para “curtos” para facilitar a leitura.
Ex.: Vang_XES_018019s → Vang_XES_0.18s, Vang_RioVTEST_035036s → Vang_RioVTEST_0.35s.

Roda diagnósticos de colunas para as selecionadas (alvo + Vang):

Quais estão faltando.

Quais estão vazias (100% NaN).

Contagem de NaN por coluna.

Mostra amostras de linhas para conferência.

Filtra linhas inválidas com regras explícitas:

Mantém apenas min_UGs entre 0 e 7.

Se existir a coluna Estabilidade, remove a combinação inconsistente
Estabilidade == "Instavel" e min_UGs == 0.

Descarta linhas que tenham qualquer valor negativo nas features Vang.

Gera visualizações pré-limpeza (para entender o estado original):

Dispersões (scatter) de cada feature vs min_UGs em ../feature_vs_target_pre_cleaning/.

Histogramas (por exemplo, para uma classe de interesse) em ../histograms_pre_cleaning/.

(Após a limpeza) volta a plotar as dispersões de cada feature vs min_UGs para comparação em
../feature_vs_target_pos_cleaning/ (nome usado no script).

Salva o dataset limpo em ../data_cleaned/:

CSV sempre.

Parquet se a engine estiver disponível (senão emite um aviso e segue só com CSV).

Loga o caminho dos arquivos gerados e encerra.

---

## 2) scripts/2-PCA.py — Redução de Dimensão (PC1, PC2)

Carrega o dataset limpo gerado pelo 1-EDA.py:

Arquivo esperado: ../data_cleaned/tabelaestabilidade_minUG_cleaned.csv.

Se não existir, gera erro pedindo para rodar o EDA antes.

Seleciona exatamente 8 features Vang e o alvo:

Alvo: min_UGs (classes 0..7).

As 8 colunas são os Vang em 0.18s, 0.25s, 0.30s, 0.35s para XES e RioVTEST.

Auto-detecção de nomes: aceita tanto o padrão curto (0.18s) quanto o longo (018019s). Se ambos existirem, prioriza o curto.

Mantém somente [min_UGs] + 8 Vang e ignora extras (ex.: Estabilidade).

Validação do alvo e limpeza defensiva:

Converte min_UGs para inteiro e remove rótulos fora de 0..7 (logando quantas linhas saíram).

Substitui ±inf por NaN e dropa linhas com NaN/inf em qualquer uma das 8 features ou no alvo (logando quantas foram removidas).

Split de treino/teste com segurança:

Tenta estratificar (mesma distribuição de classes nos conjuntos).

Se alguma classe tiver < 2 amostras, cai para split não-estratificado (com aviso).

Proporção padrão: 25% para teste (random_state=42).

Padronização e PCA(2D):

Ajusta StandardScaler apenas no treino e transforma treino e teste.

Ajusta PCA(n_components=2, random_state=42) apenas no treino e projeta ambos os conjuntos para PC1/PC2.

Imprime a variância explicada de PC1 e PC2 e a soma.

Figuras (../PCA_visualization/):

pca_train_scatter.png: dispersão do treino em (PC1, PC2), colorido por classe.

pca_test_scatter.png: dispersão do teste.

pca_explained_variance.txt: texto com as razões de variância explicada e a soma.

Dados transformados e artefatos (../data_PCA/):

CSVs: train_pca.csv, test_pca.csv, all_pca.csv (com PC1, PC2, min_UGs e a coluna split).

Parquet (opcional): tenta gravar *.parquet; se faltar pyarrow/fastparquet, avisa e segue só com CSV.

Modelos: scaler.joblib e pca.joblib (para reutilizar na etapa MSVM e na projeção de pontos novos).

Rastreabilidade: features_used.txt com timestamp, alvo, ordem exata das 8 features (e o “nome canônico” longo para cada uma), shapes de X_train/X_test e a variância explicada.

Logs finais:

Lista tudo que foi salvo em data_PCA/ e indica a pasta das figuras.

Finaliza com “Done.”

---

## 3) scripts/3-MSVM_com_ponto.py — MSVM Linear por Pares

Carrega os dados em PCA
Lê data_PCA/train_pca.csv e data_PCA/test_pca.csv (gerados pelo 2-PCA.py). Exige as colunas PC1, PC2 e min_UGs e converte o alvo para inteiro.
Se os arquivos não existirem ou faltarem colunas, lança erro claro pedindo para rodar o 2-PCA.py.

Define os pares adjacentes
Considera os pares (0_vs_1, 1_vs_2, …, 6_vs_7) e grava results_msvm/pairs_order.txt com essa ordem.

Treina um SVM linear por par (no plano PCA)
Para cada par k vs k+1:

Cria um subconjunto binário a partir do dataframe (min_UGs ∈ {k, k+1}), com rótulos 0/1 onde 1 ≡ k+1.

Ajusta um LinearSVC(C=1.0, class_weight="balanced") usando apenas os pontos de treino (PC1, PC2).

Obtém os scores contínuos via decision_function.

Calibra um threshold conservador (no TREINO)
Define o limiar como o menor score entre as amostras positivas (classe k+1) no train:
threshold = min(score | y_train == 1)
Objetivo: evitar FN no treino (quando possível). Se não houver positivos, usa -inf como fallback.
Predição binária: 1 se score >= threshold; caso contrário, 0.

Avalia no TREINO e no TESTE
Calcula TP, TN, FP, FN por par no train (com o mesmo threshold) e depois no test.
Consolida linhas de métricas para salvar em CSV.

Salva artefatos numéricos e modelos

results_msvm/thresholds.json — threshold por par.

results_msvm/metrics_train.csv e results_msvm/metrics_test.csv — métricas por par.

results_msvm/models_adjacent.joblib — dicionário com todos os LinearSVC treinados.

Plota figuras agregadas com todas as fronteiras
Gera dois gráficos (um para treino e outro para teste) em best_treshold_evaluation/:

Dispersão de todos os pontos em (PC1, PC2), coloridos por classe (min_UGs).

Todas as retas de decisão Dx(k)=0 (já com o threshold aplicado) para os pares treinados.

Marca falsos negativos (da classe k+1 classificados como k) com “x” vermelho.

Se houver um ponto destacado (ver abaixo), plota um asterisco e anota suas coordenadas.

(Opcional) Projeta um ponto Vang “capturado” para o PCA

Lê data_PCA/features_used.txt para garantir a mesma ordem de 8 features.

Valida que o CAPTURED_VANG (dict) tem exatamente as mesmas chaves.

Carrega data_PCA/scaler.joblib e data_PCA/pca.joblib, calcula z = (x−μ)/σ e PC = P[:2]·z.

Imprime diagnósticos de contribuição por feature em PC1/PC2 e confere consistência com o sklearn.

Plota o ponto no gráfico agregado de treino (asterisco).

Exporta as fronteiras para o espaço original (Vang)
Para cada par treinado, mapeia a reta no PCA
w_pca·PC + b_pca − thr = 0
de volta às 8 variáveis Vang usando PC = P·z, z = (x−μ)/σ:

w_z = w_pca @ P

w_raw = w_z / σ

c_raw = (b_pca − thr) − w_raw·μ
Resultado: uma equação linear em Vang do tipo
Dx(k): Σ a_j·Vang_j + c = 0,
mais um resumo com [PCA] w=(w1,w2), b, thr.
Tudo é salvo em parametrização/equacoes_fronteiras_vang.txt.

Mensagens especiais e edge cases

Se um par não tiver amostras suficientes nas duas classes, o par é pulado (log explícito).

O código imprime debugs úteis (ordem das features, z-scores, contribuições, etc.).

Paths são relativos à pasta do script; as saídas ficam em diretórios irmãos de data_PCA/.

Resumo das saídas principais

Modelos/thresholds/métricas: results_msvm/

Figuras agregadas: best_treshold_evaluation/all_pairs_train.png e all_pairs_test.png

Equações em Vang: parametrização/equacoes_fronteiras_vang.txt

Ordem dos pares: results_msvm/pairs_order.txt

---

## Orquestrador — main.py

Executa os três scripts em sequência

Aguarda 1s entre cada script

---

## Saídas (onde os arquivos são gravados)

Dados limpos: data_cleaned/

PCA e artefatos: data_PCA/, PCA_visualization/

Modelos e métricas: results_msvm/

Equações originais: parametrização/equacoes_fronteiras_vang.txt

Figuras: results_msvm/plots/, best_treshold_evaluation/, caso_base_evaluation/

