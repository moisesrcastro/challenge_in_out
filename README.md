# Detecção de Bola "Dentro ou Fora" no Vôlei com Visão Computacional

Este projeto aplica técnicas de **visão computacional** e **física do impacto** para identificar automaticamente se uma bola de vôlei tocou **dentro ou fora** da quadra, simulando a análise exibida nos desafios eletrônicos da **Volleyball Nations League (VNL)**.

## Objetivo

A proposta é reconstruir o instante do impacto da bola com o solo a partir de um vídeo real, utilizando segmentação de imagem, interpolação espacial e modelos físicos para determinar a posição da bola, calcular sua velocidade e estimar a área de contato com o solo.

## Demonstração

Abaixo estão os vídeos com os resultados:

<table>
  <tr>
    <td><strong>Vídeo Original</strong></td>
    <td><strong>Reconstrução Estilo VNL</strong></td>
  </tr>
  <tr>
    <td align="center">📎 <a href="assets/saida_com_velocidade.mp4">Clique aqui para assistir</a></td>
    <td align="center">📎 <a href="assets/challenge.mp4">Clique aqui para assistir</a></td>
  </tr>
</table>

## Metodologia

- **Captura de vídeo**: Extração quadro a quadro dos frames relevantes.
- **Segmentação da bola**: Criação de máscaras HSV para detectar a bola em diferentes condições de cor e iluminação.
- **Rastreamento**: Cálculo do centro da bola em cada frame.
- **Análise física**:
  - Estimativa da velocidade da bola.
  - Cálculo da deformação no impacto com base no modelo de Hertz.
  - Determinação da área de contato real da bola com o solo.
- **Classificação**: A partir da projeção da área de contato, é verificado se a bola tocou a linha da quadra (dentro) ou fora.
- **Visualização final**: Recriação da animação usada pela VNL, com diferentes ângulos (lateral, angular, aérea) e textos explicativos.

## Tecnologias Utilizadas

- Python
- OpenCV (manipulação de imagem e vídeo)
- NumPy (operações numéricas)
- Modelagem física (Teoria de Hertz para contato elástico)

## Resultados

A sequência de imagens gerada simula com fidelidade a apresentação gráfica utilizada em jogos profissionais, permitindo verificar de forma visual e objetiva o ponto de contato da bola. Essa abordagem pode ser aplicada a sistemas de arbitragem automática, análise tática ou projetos de computação gráfica esportiva.

## Estrutura do Projeto
```
├── assets/
│ ├── challenge.mp4 # Animação gerada simulando o desafio VNL
│ ├── saida_com_velocidade.mp4 # Vídeo original com tracking da bola
│ ├── QUADRO1.jpg # Visão lateral da quadra
│ ├── QUADRO2.jpg # Visão angular do impacto
│ ├── QUADRO3.jpg # Visão aérea da marca
│ └── Bola.png # Imagem da bola com canal alpha
├── notebook/
│ └── analise_bola_volei.ipynb # Notebook principal com todo o processamento
└── README.md
```
