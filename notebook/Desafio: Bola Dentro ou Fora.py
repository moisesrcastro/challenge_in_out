# Databricks notebook source
import cv2
import time
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import math

# COMMAND ----------

# MAGIC %md
# MAGIC # Carregamento e Captura dos Frames do Vídeo
# MAGIC
# MAGIC Neste trecho, realizamos a abertura do vídeo localizado no caminho especificado (`video.mov`) utilizando a biblioteca OpenCV. A partir do vídeo, extraímos o número de frames por segundo (FPS) para determinar quantos frames iremos capturar.
# MAGIC
# MAGIC O código captura os frames do vídeo até atingir o número máximo igual ao FPS, ou seja, aproximadamente 1 segundo de vídeo. Cada frame capturado é armazenado em uma lista para processamento posterior.
# MAGIC
# MAGIC Esse procedimento permite trabalhar com uma quantidade controlada de frames, facilitando a análise da trajetória da bola em segmentos específicos do vídeo.
# MAGIC

# COMMAND ----------

video_path = "/Assets/video.mov"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
frame_count = 0
max_frames = fps  

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_count += 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Criação da Máscara para Identificação do Centro da Bola
# MAGIC
# MAGIC Esta função realiza a segmentação da bola quadro a quadro com base em características de cor no espaço HSV. O processo segue os passos abaixo:
# MAGIC
# MAGIC 1. **Conversão de Espaço de Cor**: O frame é convertido de RGB para HSV, facilitando a segmentação por cor.
# MAGIC
# MAGIC 2. **Definição de Máscaras de Cor**: São criadas máscaras para detectar áreas brancas, verdes e azuis, que correspondem às cores predominantes da bola ou elementos relevantes para a detecção.
# MAGIC
# MAGIC 3. **Combinação das Máscaras**: As máscaras individuais são combinadas para capturar todas as regiões de interesse.
# MAGIC
# MAGIC 4. **Limpeza da Máscara**: Operações morfológicas (abertura) são aplicadas para remover ruídos e pequenas imperfeições.
# MAGIC
# MAGIC 5. **Suavização e Binarização**: A máscara é suavizada com filtro Gaussiano e binarizada para reforçar as regiões detectadas.
# MAGIC
# MAGIC 6. **Detecção de Contornos**: Os contornos da máscara binarizada são extraídos, e o maior contorno é selecionado, presumidamente correspondendo à bola.
# MAGIC
# MAGIC 7. **Cálculo do Centro e Tamanho**: Com base no maior contorno, calcula-se o retângulo que o engloba e, a partir dele, obtém-se o centro (cx, cy) e o tamanho do lado do quadrado que cobre a bola.
# MAGIC
# MAGIC Se nenhum contorno é encontrado, a função retorna `[None, None]`, indicando que a bola não foi detectada naquele frame.
# MAGIC
# MAGIC Esse método permite rastrear a posição da bola em cada frame, fundamental para análises posteriores de trajetória e impacto.
# MAGIC

# COMMAND ----------

def PosicaoBola(frame):

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Máscara branca (baixa saturação, valor alto)
    mask_white = cv2.inRange(frame_hsv, np.array([0, 0, 255]), np.array([255, 255, 255]))

    # Máscara verde
    mask_green = cv2.inRange(frame_hsv, np.array([40, 40, 40]), np.array([85, 255, 255]))

    # Máscara azul
    mask_blue = cv2.inRange(frame_hsv, np.array([45, 45, 45]), np.array([0, 255, 255]))

    # Combinar todas
    combined_mask = cv2.bitwise_or(mask_white, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue)

    # Aplicar morfologia para limpar ruídos
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 1. Suavizar máscara para reduzir ruído
    mask_blur = cv2.GaussianBlur(mask_clean, (7,7), 0)

    # 2. Binarizar de novo (threshold adaptativo ou fixo)
    _, mask_bin = cv2.threshold(mask_blur, 127, 255, cv2.THRESH_BINARY)

    # 3. Operações morfológicas (abrir para tirar ruídos pequenos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_morph = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Encontrar contornos
    contornos, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        # 5. Pega maior contorno
        maior_contorno = max(contornos, key=cv2.contourArea)

        # Cria máscara só do maior contorno
        mask_maior = np.zeros_like(mask_clean)
        cv2.drawContours(mask_maior, [maior_contorno], -1, 255, thickness=-1)

        # 6. Calcula quadrado que engloba esse contorno
        ys, xs = np.where(mask_maior == 255)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        w, h = x_max - x_min, y_max - y_min
        lado = max(w, h)

        cx = x_min + w // 2
        cy = y_min + h // 2
        
        centro = [cx, cy]

    else:
        return [None, None]
    
    return [centro, lado]

# COMMAND ----------

centros = []
lados = []
for i in range(len(frames)):
    frame = frames[i]
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    centro, lado = PosicaoBola(frame)
    centros.append(centro)
    lados.append(lado)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cálculo da Velocidade da Bola
# MAGIC
# MAGIC Com base nas posições do centro da bola extraídas para cada frame, é possível estimar a velocidade da bola em metros por segundo.
# MAGIC
# MAGIC O cálculo segue os seguintes passos:
# MAGIC
# MAGIC 1. **Conversão de Escala**: Define-se um fator de conversão de pixels para metros, a partir de uma referência conhecida (ex: 0,65 metros equivalem a 437 pixels). Isso permite transformar deslocamentos em pixels para unidades reais.
# MAGIC
# MAGIC 2. **Intervalo de Tempo**: O tempo entre dois frames consecutivos (`dt`) é calculado como o inverso da taxa de frames por segundo (FPS) do vídeo.
# MAGIC
# MAGIC 3. **Cálculo da Velocidade Vertical**:
# MAGIC    - Para cada par de centros consecutivos, calcula-se a diferença na coordenada vertical (eixo *y*), assumindo que o movimento mais relevante ocorre nesse eixo.
# MAGIC    - A velocidade em pixels por segundo é dada por `v_px = dy / dt`.
# MAGIC    - Essa velocidade é então convertida para metros por segundo usando o fator de escala.
# MAGIC
# MAGIC 4. **Tratamento de Falhas de Detecção**: Se a posição da bola não foi detectada em algum dos frames (centro igual a `None`), a velocidade correspondente também é registrada como `None`.
# MAGIC
# MAGIC O resultado é uma lista de velocidades quadro a quadro, que pode ser utilizada para análises físicas da trajetória da bola, como desaceleração após o impacto ou estimativa da energia do movimento.
# MAGIC

# COMMAND ----------

escala = 0.65 / 437  # metros por pixel
dt = 1 / fps
velocidades = []

for i in range(len(centros) - 1):
    p1 = centros[i]
    p2 = centros[i + 1]

    if p1 is None or p2 is None:
        velocidades.append(None)
    else:
        dy = p2[1] - p1[1]   # diferença na coordenada y
        v_px = dy / dt       # velocidade em pixels/s (com sinal)
        v_m = v_px * escala  # velocidade em m/s (com sinal)
        velocidades.append(v_m)

# COMMAND ----------

velocidades.insert(0, None)

# COMMAND ----------

def encontrar_mudanca_sinal_com_none(lista):
    for i in range(len(lista) - 1):
        a = lista[i]
        b = lista[i + 1]
        if a is not None and b is not None:
            if a * b < 0:
                return i
    return None  

idx_impacto = encontrar_mudanca_sinal_com_none(velocidades)

# COMMAND ----------

# MAGIC %md
# MAGIC # Determinação do Impacto e Classificação "Dentro" ou "Fora"
# MAGIC
# MAGIC Nesta etapa, realizamos um ajuste refinado da posição da bola no frame de impacto, com o objetivo de determinar se o contato ocorreu dentro ou fora da quadra, com base na área de deformação da bola ao tocar o solo.
# MAGIC
# MAGIC ### 1. **Extrapolação da Posição no Frame de Impacto**
# MAGIC Utilizamos a posição da bola nos dois frames anteriores ao impacto para estimar sua posição exata no instante de contato com o solo (frame 5). Isso é feito por meio de interpolação linear, considerando a coordenada vertical da linha do chão. O raio da bola (em metros e convertido para pixels) é subtraído para ajustar o centro da bola à altura do contato real com o solo.
# MAGIC
# MAGIC ### 2. **Cálculo da Deformação e Área de Contato**
# MAGIC Com a velocidade vertical da bola no momento do impacto e o raio físico da bola, aplicamos um modelo simplificado baseado na teoria de Hertz para calcular a deformação (`δ`) e o raio da área de contato (`a`). Esses valores são fundamentais para representar com realismo o contato físico entre a bola e o solo.
# MAGIC
# MAGIC - **Deformação:**  
# MAGIC   \[
# MAGIC   \delta = \left( \frac{9 \cdot v^2 \cdot R}{16 \cdot E^*} \right)^{1/3}
# MAGIC   \]
# MAGIC - **Raio da Área de Contato:**  
# MAGIC   \[
# MAGIC   a = \sqrt{R \cdot \delta}
# MAGIC   \]
# MAGIC
# MAGIC ### 3. **Projeção da Área de Contato**
# MAGIC A partir do centro estimado no frame de impacto, projetamos horizontalmente a área de contato como um segmento entre dois pontos (`P_esq` e `P_dir`) sobre a linha do chão.
# MAGIC
# MAGIC ### 4. **Avaliação da Posição em Relação à Linha**
# MAGIC Com uma linha de referência inclinada representando o limite da quadra, utilizamos produto vetorial para verificar em que lado da linha cada ponto extremo da área de contato se encontra. Se pelo menos uma parte da área de contato estiver do lado "interno" da quadra, o toque é classificado como **dentro** (`"IN"`), caso contrário como **fora** (`"OUT"`).
# MAGIC
# MAGIC Esse processo integra dados físicos, geométricos e espaciais para reproduzir de forma confiável a decisão que seria tomada com base na imagem do impacto, simulando o funcionamento de tecnologias como o desafio eletrônico no vôlei profissional.
# MAGIC

# COMMAND ----------

# --- Cálculo do centro da bola no frame de impacto por extrapolação ---
cx3, cy3 = centros[idx_impacto-2]
cx4, cy4 = centros[idx_impacto-1]
lado3 = lados[idx_impacto-2]
lado4 = lados[idx_impacto-1]

# Parâmetros físicos e escala
y_linha = 2340             # linha do chão (pixels)
escala = 0.65 / 417        # metros por pixel (ajuste conforme seu setup)
R_m = 0.67 / 2             # raio da bola em metros

# Raio em pixels (para ajustar y do centro no solo)
R_px = int(R_m / escala)

# Cálculo do fator para extrapolar o centro da bola no frame 5 usando y da linha do chão
denominador = (cy4 - cy3)
if denominador != 0:
    fator = (y_linha - R_px - cy3) / denominador
else:
    fator = 2  # fallback caso denom = 0

# Extrapolação da posição e do tamanho da bola para o frame 5
cx5 = int(cx3 + (cx4 - cx3) * fator)
cy5 = int(cy3 + (cy4 - cy3) * fator)
lado5 = int(lado3 + (lado4 - lado3) * fator)

# --- Cálculo da deformação (delta) e raio da área de contato (a) ---
v_y = velocidades[idx_impacto]               # velocidade vertical no impacto (m/s)
E_star = 1e5               # módulo de Hertz efetivo (Pa)

delta = ((9 * v_y**2 * R_m) / (16 * E_star))**(1/3)
a = np.sqrt(R_m * delta)   # raio da área de contato em metros
a_px = a / escala          # conversão para pixels

cx_rot = cx5
cy_rot = cy5

# --- Define extremos da área de contato ---
P_esq = (int(cx_rot - a_px), y_linha)
P_dir = (int(cx_rot + a_px), y_linha)

# --- Linha inclinada perto de P_esq ---
angulo_graus = 45       # ângulo da linha inclinada (graus)
comprimento = 500       # comprimento da linha (pixels)
cor_linha = (0, 255, 255)  # ciano
espessura_linha = 10

theta = -math.radians(angulo_graus)

x0, y0 = P_esq[0] - 50, P_esq[1]
x1 = int(x0 - comprimento * math.cos(theta))
y1 = int(y0 - comprimento * math.sin(theta))

# --- Função para calcular lado do ponto em relação à linha ---
def lado_do_ponto(px, py, x0, y0, x1, y1):
    vx = x1 - x0
    vy = y1 - y0
    wx = px - x0
    wy = py - y0
    return vx * wy - vy * wx

# --- Verifica se bola toca dentro ou fora ---
lado_esq = lado_do_ponto(P_esq[0], P_esq[1], x0, y0, x1, y1)
lado_dir = lado_do_ponto(P_dir[0], P_dir[1], x0, y0, x1, y1)

toca_dentro = (lado_esq > 0) or (lado_dir > 0)

texto = "IN" if toca_dentro else "OUT"

# COMMAND ----------

# MAGIC %md
# MAGIC # Reconstrução Visual da Animação do Desafio
# MAGIC
# MAGIC Com os dados obtidos da análise quadro a quadro do vídeo — como posição da bola, momento do impacto, deformação e classificação do toque — esta etapa realiza a montagem visual da sequência de imagens, simulando a animação exibida durante os desafios eletrônicos da VNL (Volleyball Nations League).
# MAGIC
# MAGIC ### 1. **Carregamento das Imagens Base**
# MAGIC Três imagens de fundo são carregadas:
# MAGIC - `QUADRO1.jpg`: visão lateral da quadra.
# MAGIC - `QUADRO2.jpg`: visão angular para destacar o toque no solo.
# MAGIC - `QUADRO3.jpg`: visão aérea para marcar a área de contato.
# MAGIC Além disso, é carregada a imagem da bola com canal alpha (`Bola.png`), permitindo sobreposição com transparência.
# MAGIC
# MAGIC ### 2. **Interpolação e Cálculo da Posição no Impacto**
# MAGIC A posição final da bola é interpolada a partir de dois quadros anteriores ao impacto, garantindo consistência com os dados físicos já estimados. Com isso, determina-se a posição exata da bola no instante do toque.
# MAGIC
# MAGIC ### 3. **Aplicação de Sombra e Sobreposição da Bola**
# MAGIC Para reforçar a percepção visual do impacto:
# MAGIC - É desenhada uma sombra elíptica no chão, representando a deformação da bola no instante do toque (utilizada no quadro 1 e 2).
# MAGIC - No quadro aéreo, aplica-se uma sombra circular mais discreta, representando a marca deixada na quadra.
# MAGIC - A imagem da bola é sobreposta de forma realista sobre os quadros, respeitando a escala e posição.
# MAGIC
# MAGIC ### 4. **Inserção de Texto**
# MAGIC Textos com o resultado do desafio ("IN" ou "OUT") são adicionados em posições estratégicas nas imagens, utilizando cores padronizadas:
# MAGIC - Verde para "IN"
# MAGIC - Vermelho para "OUT"
# MAGIC
# MAGIC Além disso, o quadro 1 simula a transição inicial exibida no desafio com as legendas "FOR" e "IN/OUT...".
# MAGIC
# MAGIC ### 5. **Montagem Final**
# MAGIC Cada quadro é ajustado individualmente:
# MAGIC - **Quadro 1**: mostra a trajetória da bola e a sombra da área de contato.
# MAGIC - **Quadro 2**: destaca o instante do impacto, com sombra e sobreposição da bola.
# MAGIC - **Quadro 3**: evidencia a marca do toque na visão aérea, com texto indicando o resultado final.
# MAGIC
# MAGIC Essa reconstrução busca aproximar a experiência visual apresentada nas transmissões oficiais, reunindo a análise física e espacial com elementos gráficos de alto impacto.
# MAGIC

# COMMAND ----------

# --- Entrada: carregue imagens base e bola.png ---
quadro1 = cv2.imread('/Assets/QUADRO1.jpg', cv2.IMREAD_UNCHANGED)
quadro2 = cv2.imread('/Assets/QUADRO2.jpg', cv2.IMREAD_UNCHANGED)
quadro3 = cv2.imread('/Assets/QUADRO3.jpg', cv2.IMREAD_UNCHANGED)
bola_img = cv2.imread('/Assets/Bola.png', cv2.IMREAD_UNCHANGED)  # com alpha

# --- Dados da bola ---
i = idx_impacto
y_linha = 2340

# Interpolação baseada em frames 3 e 4 (defina cx3, cx4, etc. antes)
cx = int(cx3 + (cx4 - cx3) * fator)
cy = int(cy3 + (cy4 - cy3) * fator)
lado = int(lado3 + (lado4 - lado3) * fator)
raio = lado // 2

# Posição do impacto
cx_impacto = cx
cy_impacto = y_linha - 1700

# Escala para converter metros em pixels (usada para a elipse do quadro 1)
delta = ((9 * v_y**2 * R_m) / (16 * E_star))**(1/3)
a = np.sqrt(R_m * delta)
a_px = int(a / escala)

# --- Função para sobrepor imagem da bola com alpha ---
def overlay_bola(imagem_fundo, bola_img, cx, cy, lado):
    bola_resized = cv2.resize(bola_img, (lado, lado), interpolation=cv2.INTER_AREA)
    x1, y1 = cx - lado // 2, cy - lado // 2
    x2, y2 = x1 + lado, y1 + lado

    if x1 < 0 or y1 < 0 or x2 > imagem_fundo.shape[1] or y2 > imagem_fundo.shape[0]:
        return imagem_fundo

    bola_rgb = bola_resized[..., :3]
    alpha = bola_resized[..., 3:] / 255.0
    fundo_crop = imagem_fundo[y1:y2, x1:x2]
    blended = (1 - alpha) * fundo_crop + alpha * bola_rgb
    imagem_fundo[y1:y2, x1:x2] = blended.astype(np.uint8)
    return imagem_fundo

# --- Funções para sombras ---
def aplicar_sombra_elipse(imagem, cx, cy, raio, angulo=0):
    eixo_maior = int(raio * 1.2)
    eixo_menor = int(raio * 0.4)
    sombra = np.zeros_like(imagem, dtype=np.uint8)
    cv2.ellipse(sombra, (cx, cy), (eixo_maior, eixo_menor), angulo, 0, 360,
                (50, 50, 50), -1, lineType=cv2.LINE_AA)
    return cv2.subtract(imagem, sombra)

def aplicar_sombra_circular(imagem, cx, cy, raio):
    sombra = np.zeros_like(imagem, dtype=np.uint8)
    sombra_raio = int(raio * 0.95)
    cv2.circle(sombra, (cx, cy), sombra_raio, (50, 50, 50), -1, lineType=cv2.LINE_AA)
    return cv2.subtract(imagem, sombra)

# --- Função para adicionar texto ---
def adicionar_texto(imagem, texto, posicao, cor=(255, 255, 255), fonte=cv2.FONT_HERSHEY_SIMPLEX,
                    tamanho=2, espessura=3):
    cv2.putText(imagem, texto, posicao, fonte, tamanho, cor, espessura, lineType=cv2.LINE_AA)
    return imagem

# Texto e cor para todos os quadros
resultado = "IN" if toca_dentro else "OUT"
cor_resultado = (0, 255, 0) if toca_dentro else (0, 0, 255)

# --- Quadro 1 ---
quadro1_copy = quadro1.copy()
quadro1_com_sombra = aplicar_sombra_elipse(quadro1_copy, cx_impacto, y_linha, a_px)
quadro1_com_bola = overlay_bola(quadro1_com_sombra, bola_img, cx, cy, lado)
quadro1_final = adicionar_texto(quadro1_com_bola, "FOR", (int(2160/2)+100, 3560),)
quadro1_final = adicionar_texto(quadro1_final, "IN/OUT...", (int(2160/2)+40, 3645), tamanho=2.75, espessura=6)
# --- Quadro 2 ---
if i == idx_impacto:
    quadro2_copy = quadro2.copy()
    quadro2_com_sombra = aplicar_sombra_elipse(quadro2_copy, cx_impacto, cy_impacto, raio)
    quadro2_com_bola = overlay_bola(quadro2_com_sombra, bola_img, cx_impacto, cy_impacto - raio, lado)
    quadro2_final = adicionar_texto(quadro2_com_bola, resultado, (int(2160/2)+100, 3610), tamanho=2.75, espessura=8)

# --- Quadro 3 ---
if i == idx_impacto:
    cy_impacto_3 = cy_impacto - 200
    cx_impacto_3 = cx_impacto - 100
    quadro3_copy = quadro3.copy()
    quadro3_com_sombra = aplicar_sombra_circular(quadro3_copy, cx_impacto_3, cy_impacto_3, raio)
    quadro3_final = adicionar_texto(quadro3_com_sombra, resultado, (int(2160/2)+100, 3610), tamanho=2.75, espessura=8)


# COMMAND ----------

# --- Carregamento das imagens base ---
quadro1 = cv2.imread('/Assets/QUADRO1.jpg', cv2.IMREAD_UNCHANGED)
quadro2 = cv2.imread('/Assets/QUADRO2.jpg', cv2.IMREAD_UNCHANGED)
quadro3 = cv2.imread('/Assets/QUADRO3.jpg', cv2.IMREAD_UNCHANGED)
bola_img = cv2.imread('/Assets/Bola.png', cv2.IMREAD_UNCHANGED)  # com alpha

# --- Parâmetros físicos e de escala ---
delta = ((9 * v_y**2 * R_m) / (16 * E_star))**(1/3)
a = np.sqrt(R_m * delta)
a_px = int(a / escala)


# --- Funções auxiliares ---
def overlay_bola(imagem_fundo, bola_img, cx, cy, lado):
    bola_resized = cv2.resize(bola_img, (lado, lado), interpolation=cv2.INTER_AREA)
    x1, y1 = cx - lado // 2, cy - lado // 2
    x2, y2 = x1 + lado, y1 + lado

    if x1 < 0 or y1 < 0 or x2 > imagem_fundo.shape[1] or y2 > imagem_fundo.shape[0]:
        return imagem_fundo

    bola_rgb = bola_resized[..., :3]
    alpha = bola_resized[..., 3:] / 255.0
    fundo_crop = imagem_fundo[y1:y2, x1:x2]
    blended = (1 - alpha) * fundo_crop + alpha * bola_rgb
    imagem_fundo[y1:y2, x1:x2] = blended.astype(np.uint8)
    return imagem_fundo

def aplicar_sombra_elipse(imagem, cx, cy, raio, angulo=0):
    eixo_maior = int(raio)
    eixo_menor = int(raio * 0.35)
    sombra = np.zeros_like(imagem, dtype=np.uint8)
    cv2.ellipse(sombra, (cx, cy), (eixo_maior, eixo_menor), angulo, 0, 360,
                (40, 40, 40), -1, lineType=cv2.LINE_AA)
    return cv2.subtract(imagem, sombra)

def aplicar_sombra_circular(imagem, cx, cy, raio):
    sombra = np.zeros_like(imagem, dtype=np.uint8)
    cv2.circle(sombra, (cx, cy), int(raio), (40, 40, 40), -1, lineType=cv2.LINE_AA)
    return cv2.subtract(imagem, sombra)

def adicionar_texto(imagem, texto, posicao, cor=(255, 255, 255), tamanho=2.3, espessura=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imagem, texto, posicao, font, tamanho, cor, espessura, lineType=cv2.LINE_AA)
    return imagem

# --- Dados principais ---
cor_resultado = (0, 255, 0)

# --- Lista com os centros e lados da bola por frame ---
centros = centros[:3] + [(cx3, cy3), (cx4, cy4), (cx5, cy5)] + centros[6:]
lados = lados[:3] + [lado3, lado4, lado5] + lados[6:]

# --- Loop dos frames: apenas Quadro 1 (trajetória) ---
for i, ((cx, cy), lado) in enumerate(zip(centros, lados)):
    frame = quadro1.copy()

    # Aplica sombra se for após ou no impacto
    if i >= idx_impacto:
        frame = aplicar_sombra_elipse(frame, cx5, y_linha, a_px)

    # Adiciona a bola
    frame = overlay_bola(frame, bola_img, cx, cy, lado)

    # Texto de previsão
    if i < idx_impacto:
        frame = adicionar_texto(frame, "FOR", (int(2160 / 2) + 100, 3560))
        frame = adicionar_texto(frame, "IN/OUT...", (int(2160 / 2) + 40, 3645))
    else:
        frame = adicionar_texto(frame, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

# --- Quadro 2 e 3: apenas no frame de impacto ---
cx_impacto = cx5
cy_impacto = y_linha - 1700

# Quadro 2
quadro2_copy = quadro2.copy()
quadro2_sombra = aplicar_sombra_elipse(quadro2_copy, cx_impacto, cy_impacto, lado5//2)
quadro2_bola = overlay_bola(quadro2_sombra, bola_img, cx_impacto, cy_impacto - lado5 // 2, lado5)
quadro2_bola = adicionar_texto(quadro2_bola, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

# Quadro 3
quadro3_copy = quadro3.copy()
quadro3_sombra = aplicar_sombra_circular(quadro3_copy, cx_impacto - 50, cy_impacto - 200, lado5 // 2)
quadro3_sombra = adicionar_texto(quadro3_sombra, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

# COMMAND ----------

frames_video = []

# --- Função para duplicar um frame N vezes ---
def duplicar_frame(frame, vezes=60):
    return [frame.copy() for _ in range(vezes)]

# --- Loop dos frames para o Quadro 1 (trajetória) ---
for i, ((cx, cy), lado) in enumerate(zip(centros, lados)):
    frame = quadro1.copy()

    if i >= idx_impacto:
        frame = aplicar_sombra_elipse(frame, cx5, y_linha, a_px)

    frame = overlay_bola(frame, bola_img, cx, cy, lado)

    if i < idx_impacto:
        frame = adicionar_texto(frame, "FOR", (int(2160 / 2) + 100, 3560))
        frame = adicionar_texto(frame, "IN/OUT...", (int(2160 / 2) + 40, 3645))
    else:
        frame = adicionar_texto(frame, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

    frames_video.extend(duplicar_frame(frame, vezes=8))  # Duplica 2 vezes cada frame

# --- Quadro 2 e 3: só no frame de impacto, duplicar eles também ---

cx_impacto = cx5
cy_impacto = y_linha - 1700

# Quadro 2
quadro2_copy = quadro2.copy()
quadro2_sombra = aplicar_sombra_elipse(quadro2_copy, cx_impacto, cy_impacto, lado5 // 2)
quadro2_bola = overlay_bola(quadro2_sombra, bola_img, cx_impacto, cy_impacto - lado5 // 2, lado5)
quadro2_bola = adicionar_texto(quadro2_bola, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

frames_video.extend(duplicar_frame(quadro2_bola, vezes=8))

# Quadro 3
quadro3_copy = quadro3.copy()
quadro3_sombra = aplicar_sombra_circular(quadro3_copy, cx_impacto - 50, cy_impacto - 200, lado5 // 2)
quadro3_sombra = adicionar_texto(quadro3_sombra, resultado, (int(2160 / 2) + 100, 3610), cor_resultado, tamanho=2.75, espessura=8)

frames_video.extend(duplicar_frame(quadro3_sombra, vezes=8))

# --- Salvar vídeo com OpenCV ---

# Parâmetros do vídeo (ajuste a largura e altura de acordo com seus quadros)
altura, largura = frames_video[0].shape[:2]
fps = 20  # ou um fps menor para câmera lenta, tipo 15

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/Assets/challenge.mp4', fourcc, fps, (largura, altura))

for f in frames_video:
    out.write(f)

out.release()
