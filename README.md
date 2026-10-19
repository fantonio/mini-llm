# Mini-LLM do Zero com PyTorch 🧠🤖

Este repositório contém a implementação completa de um **Mini-Modelo de Linguagem (Mini-LLM)** autoregressivo construído do absoluto zero utilizando **Python** e **PyTorch**. O projeto implementa os conceitos fundamentais da arquitetura **Transformer** (a mesma base de modelos como GPT-4, Llama e Claude) de forma simplificada e educacional.

O modelo é capaz de aprender relações de adjacência de caracteres através de auto-atenção mascarada (*Causal Masked Self-Attention*) e gerar sequências de texto a partir de um prompt inicial por meio de memorização e inferência de probabilidades.

## 🚀 Funcionalidades e Características

* **Arquitetura Transformer Pura:** Implementação manual de camadas de Embedding (Token + Posição), cabeças de Auto-Atenção Mascarada (*Multi-Head Attention*), conexões residuais, *LayerNorm* e camadas de computação linear (*Feed-Forward*).
* **Script Unificado e Inteligente:** Fluxo de execução automatizado que detecta a presença de um modelo pré-treinado no diretório local. Se existir, carrega os pesos instantaneamente para inferência; caso contrário, inicializa o loop de treinamento.
* **Persistência de Dados:** Salvamento automático dos pesos (parâmetros da rede) em um arquivo `.pth` para reuso.

## 📊 Arquitetura do Sistema

Abaixo está o diagrama detalhado que ilustra as etapas de tokenização, o fluxo interno dos blocos do Transformer e a mecânica de decisão (Treino vs. Geração Direta):

<img width="1408" height="768" alt="image_c4c5f704" src="https://github.com/user-attachments/assets/f5bcfc53-61de-46be-a4be-c5608df851d2" />


## 🛠️ Tecnologias Utilizadas

* **Python 3.11+**
* **PyTorch** (Processamento de tensores e cálculo automatizado de gradientes)
* **NumPy** (Suporte interno e manipulação de arrays)

## 🔧 Como Executar o Projeto

### Prerequisites

Certifique-se de ter o Python instalado e configure as dependências necessárias no seu terminal:

```bash
# Instalação do PyTorch
pip install torch

# Instalação do NumPy (versão estável compatível)
pip install "numpy<2"
```

### Executando

Clone o repositório e execute o script principal:

```bash
python mini_llm.py
```

* **Na primeira execução:** O script detectará que o arquivo `mini_gpt_treinado.pth` não existe, iniciará as 3.000 iterações de treinamento mostrando a redução da *Loss* (Perda) no terminal, salvará o arquivo de pesos e imprimirá a frase gerada completa.
* **Nas execuções seguintes:** O modelo carregará os pesos de forma instantânea em menos de um segundo, pulando o treinamento direto para a resposta de inferência.

## 📖 Conceitos Aprendidos no Projeto

1. **Iteração vs. Perda (Loss):** O loop de treinamento atualiza os pesos da rede de maneira puramente iterativa (linear), minimizando a *Loss* calculada pela função de entropia cruzada (*CrossEntropy*).
2. **Persistência Dinâmica (Pesos):** Em vez de manter um histórico recursivo, o modelo altera sua própria estrutura física interna ajustando seus milhares de parâmetros matemáticos através do otimizador `AdamW`.
3. **Geração Autoregressiva:** O modelo gera texto prevendo o próximo caractere com base em uma janela histórica limitada (`block_size`) e realimentando a si mesmo com a nova letra escolhida probabilisticamente através do operador *Softmax*.
