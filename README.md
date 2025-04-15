# Reconhecimento de Emoções Faciais

## Sobre o Projeto

- **Objetivo:**  
  Desenvolver um modelo que identifique 7 emoções (angry, disgust, fear, happy, neutral, sad, surprise) a partir de imagens faciais.

- **Tecnologias:**  
  - **PyTorch:** Framework escolhido para construir e treinar a rede neural.  
  - **OpenCV:** Para pré-processamento e manipulação das imagens.  
  - **Matriz de Confusão:** Métrica que mostra mais claramente onde o modelo erra e quais emoções estão sendo confundidas.

## Conteúdo do repo

- **Código Completo:**  
  Um script Python que cobre desde o carregamento e pré-processamento do dataset até a construção, treinamento e avaliação da CNN.

- **Dataset:**  
  Utilizei o [Facial Emotion Dataset](https://github.com/dilkushsingh/Facial_Emotion_Dataset) (já organizado com subpastas `train` e `test`).

- **Análise de Resultados:**  
  Além de exibir a acurácia, o código gera uma matriz de confusão e um relatório de classificação para entender melhor como melhorar o modelo.

## Análise de erros

- **Matriz:**
  Ferramenta visual que mostra como as previsões do seu modelo se comparam com as classes reais dos dados. Cada linha da matriz representa os valores reais e cada coluna mostra as previsões feitas pelo modelo. Os números na diagonal principal indicam quantos exemplos foram classificados corretamente para cada emoção, enquanto os números fora da diagonal mostram as confusões, ou seja, quantos exemplos de uma determinada classe foram incorretamente classificados como outra.

## Relatório dos erros:

Os resultados atuais demonstram que o classificador consegue captar alguns padrões relevantes, alcançando uma acurácia próxima de 50.61% nos dados de teste. A análise da matriz de confusão mostra que emoções como "happy" e "surprise" são reconhecidas de maneira mais consistente, enquanto outras, como "angry", "fear" e "sad", ainda apresentam alta taxa de confusão. Esses achados indicam que, apesar do progresso inicial, há espaço para melhorias—por exemplo, por meio de técnicas de data augmentation, ajuste dos hiperparâmetros e até mesmo explorando o transfer learning.
