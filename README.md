# CHECKPOINT_02_SERS_1ccpj

# Avaliação: APIs, Energias Renováveis e Aprendizado de Máquina

Este repositório contém o desenvolvimento de duas tarefas independentes de Aprendizado de Máquina (Machine Learning) aplicadas ao setor de energia do Brasil, utilizando dados reais obtidos via consumo de APIs públicas (ANEEL e Open-Meteo).

---

##  Objetivos do Projeto

1. **Tarefa 1 (Classificação):** Predizer a fonte de geração renovável (*Solar, Eólica ou Hidráulica*) de empreendimentos outorgados pela ANEEL com base apenas na sua potência outorgada e localização geográfica (latitude e longitude).
2. **Tarefa 2 (Regressão):** Estimar a radiação solar global horizontal (\(W/m^2\)) na região de Petrolina (PE) utilizando variáveis meteorológicas horárias extraídas para o período de abril a junho de 2025.

---

##  Estrutura do Repositório

*   `aneel_classificacao_orange.csv`: Base de dados de empreendimentos de geração (ANEEL).
*   `meteo_regressao_orange.csv`: Base de dados meteorológicos horários (Open-Meteo).
*   `notebook_solucao.ipynb`: Jupyter Notebook contendo toda a pipeline de Ciência de Dados (Análise Exploratória, Pré-processamento, Treinamento, Avaliação de Modelos e Gráficos).
*   `README.md`: Este arquivo com o relatório final e orientações.

---

##  Instruções de Execução

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado e as seguintes bibliotecas obrigatórias:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests
```

### Como Executar
1. Clone o repositório para sua máquina local:
   ```bash
   git clone https://github.com
   ```
2. Abra e execute o notebook `notebook_solucao.ipynb` em seu ambiente de preferência (Jupyter Lab, VS Code ou Google Colab).
3. O notebook executará as requisições às APIs para gerar os arquivos `.csv` (caso não estejam locais) e rodará os treinamentos sequencialmente.

---

## Tarefa 1 — Classificação da Fonte Renovável

### Origem dos Dados
*   **Fonte:** SIGA — Agência Nacional de Energia Elétrica (ANEEL).
*   **Abordagem:** Dados cadastrais de empreendimentos (não mede energia gerada em tempo real).
*   **Alvo (\(y\)):** `fonte` (Solar - UFV, Eólica - EOL, Hidráulica - UHE/PCH/CGH).
*   **Entradas (\(X\)):** `potencia_kw`, `latitude`, `longitude`.

### Estratégia de Validação
*   **Divisão:** 80% Treino / 20% Teste.
*   **Divisão Estratificada:** Aplicada para manter a proporção original das classes em ambos os conjuntos.
*   **Semente Aleatória (Random State):** Fixada em `42` para garantir reprodutibilidade.
*   **Padronização:** Aplicado o `StandardScaler` nos dados de treino (e replicado no teste) para os algoritmos sensíveis à escala (ex: KNN, SVM ou Regressão Logística).

### Resultados e Comparação
As métricas multiclasse abaixo foram calculadas utilizando a média **macro**:

| Algoritmo | Acurácia (Accuracy) | Precisão (Precision) | Revocação (Recall) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Regressão Logística** | 0.7964 | 0.8003 | 0.7968 | 0.7933 |
| **KNN (k=5)** | 0.9652 | 0.9663 | 0.9636 | 0.9648 |
| **Random Forest** | 0.9742 | 0.9754 | 0.9727 | 0.9739 |

### Análise e Conclusões da Tarefa 1
*   **Matriz de Confusão:** No modelo Random Forest, a maior confusão identificada foi de 7 empreendimentos da classe **Solar** previstos como **Hidráulica**.
*   **Limitações do Modelo:** Prever a fonte energética utilizando exclusivamente coordenadas e potência possui limitações severas. O modelo ignora fatores fundamentais como a topografia e hidrografia (críticas para hidráulicas), o regime de ventos de alta altitude (essencial para eólicas) e o direcionamento/índice de irradiância local. Portanto, clusters puramente espaciais não capturam totalmente a complexidade da escolha da engenharia da usina.

---

## Tarefa 2 — Regressão da Radiação Solar

### Origem dos Dados
*   **Fonte:** API Histórica Open-Meteo (Coordenadas de Petrolina-PE: \(-9.39, -40.50\)).
*   **Período:** 01/04/2025 a 30/06/2025 (Fuso: `America/Recife`).
*   **Filtro:** Apenas registros diurnos (entre 7h e 17h).
*   **Alvo (\(y\)):** `radiacao_w_m2` (Radiação solar global horizontal média da hora anterior).
*   **Entradas (\(X\)):** `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh`, `hora`.

### Estratégia de Validação
*   **Divisão Temporal:** Primeiras 80% das horas cronológicas utilizadas para treino e as últimas 20% para teste. 
*   *Nota: Não foi utilizada divisão aleatória tradicional para evitar o vazamento de dados temporais (Data Leakage).*

### Resultados e Comparação

| Algoritmo | MAE (\(W/m^2\)) | MSE (\((W/m^2)^2\)) | \(R^2\) Score |
| :--- | :---: | :---: | :---: |
| **Regressão Linear** | 145.2049 | 30034.2011 | 0.3598 |
| **Random Forest** | 67.1746 | 7468.9354 | 0.8408 |
| **Gradient Boosting** | 67.6205 | 7524.8326 | 0.8396 |

### Análise e Conclusões da Tarefa 2
*   **Comparação dos Modelos:** O Random Forest apresentou R² de 0.8408, MAE de 67.1746 W/m² e MSE de 7468.9354 (W/m²)². Entre os modelos avaliados, esses resultados correspondem ao maior R² e ao menor MAE e MSE.
*   **O papel da Hora do Dia:** A variável `hora` é o preditor mais crítico do modelo, pois dita o ângulo zenital solar. Mesmo em dias completamente limpos ou nublados, o teto máximo de radiação disponível é rigidamente delimitado pelo horário (curva senoidal ao longo do dia, com pico por volta do meio-dia).
*   **Radiação Média vs. Geração Elétrica:** Estimar a radiação que atinge o solo **não equivale** a prever a geração elétrica de uma usina fotovoltaica. A conversão final depende de fatores físicos e operacionais não incluídos no dataset, tais como:
    *   Eficiência nominal dos módulos fotovoltaicos e inversores.
    *   Coeficiente de perda por temperatura (painéis perdem eficiência quando esquentam demais).
    *   Sujeira acumulada nas placas (soiling) e sombreamento local.
    *   O ângulo de inclinação dos painéis (fixo vs. rastreadores solares/trackers).
