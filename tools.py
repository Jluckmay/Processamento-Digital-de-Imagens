import cv2
import numpy as np
import mahotas # type: ignore
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO


# Função para descrever a imagem com base no descritor selecionado
def describe_image(imagem_rgb, descritor):
    """
    Função para descrever a imagem com base no descritor selecionado.
    Args:
        imagem_rgb (numpy.ndarray): Imagem em formato RGB.
        descritor (str): Descritor selecionado pelo usuário.
    Returns:
        None: Exibe os resultados no Streamlit.
    """
    # Converter imagem para escala de cinza
    imagem_gray = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

    if descritor == "Cor":
        # Calcular histograma de cores
        hist = cv2.calcHist([imagem_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # Normalizar o histograma
        hist = cv2.normalize(hist, hist).flatten()
        # Exibir histograma de cores
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(hist)), hist, color=['red', 'green', 'blue'], alpha=0.6)
        ax.set_title("Histograma de Cores", fontsize=12, fontweight="bold", color="white")
        ax.set_xlabel("Canais de Cor", color="white")
        ax.set_ylabel("Frequência", color="white")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_facecolor("black")  # Fundo preto no histograma
        fig.patch.set_alpha(0)  # Remove fundo branco da figura
        st.pyplot(fig)
        st.download_button(
            "📥 Baixar Histograma de Cores",
            data=convert_for_download(fig),
            file_name="histograma_cores.png",
            mime="image/png"
        )

        # Média e Desvio Padrão
        media = cv2.mean(imagem_rgb)
        desvio_padrao = cv2.meanStdDev(imagem_rgb)[1]
        st.write("Média:", media)
        st.write("Desvio Padrão:", desvio_padrao)

        # Momentos de Cor
        momentos = cv2.moments(imagem_gray)
        st.write("Momentos de Cor:", momentos)

    elif descritor == "Forma":
        
        # Hu Moments
        momentos_hu = cv2.HuMoments(cv2.moments(imagem_gray)).flatten()
        st.write("Momentos de Hu:", momentos_hu)

        # Fourier Descriptors
        contours, _ = cv2.findContours(imagem_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            fourier_descriptors = np.fft.fft(contour[:, 0, 0] + 1j * contour[:, 0, 1])
            st.write("Descritores de Fourier:", fourier_descriptors)
        else:
            st.write("Nenhum contorno encontrado.")

        # Zernike Moments
        zernike_moments = mahotas.features.zernike_moments(imagem_gray, radius=21)
        st.write("Momentos de Zernike:", zernike_moments)

        # Aspectro, Area, Perímetro e Circularidade
        aspecto = cv2.minAreaRect(contours[0])
        area = cv2.contourArea(contours[0])
        perimetro = cv2.arcLength(contours[0], True)
        circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro != 0 else 0
        st.write("Aspecto (Retângulo Mínimo):", aspecto)
        st.write("Área:", area)
        st.write("Perímetro:", perimetro)
        st.write("Circularidade:", circularidade)

        # Contornos RBG
        if contours:
            img_contours = cv2.drawContours(imagem_rgb.copy(), contours, -1, (0, 255, 0), 3)
            show_image(img_contours, "Contornos Detectados")
            st.download_button(
                "📥 Baixar Contornos",
                data=convert_for_download(img_contours),
                file_name="contornos.png",
                mime="image/png"
            )
            st.write(f"Contornos foram detectados na imagem em {contours[0].shape[0]} pontos:")
            for i, point in enumerate(contours[0]):
                
                # Extrair coordenadas do ponto
                ponto = (point[0][0], point[0][1]) 
                # Exibir cada ponto do contorno
                st.write(f"Ponto {i+1}: {ponto}")

    elif descritor == "Textura":

        # Cálculo de textura usando Haralick
        glcm = mahotas.features.haralick(imagem_gray)
        st.write("Textura (Haralick):", glcm)

        # Cálculo de textura usando LBP (Local Binary Patterns)
        lbp = mahotas.features.lbp(imagem_gray, 8, 1)
        st.write("Textura (LBP):", lbp)

    else:
        descritor = "Nenhum descritor selecionado"
        st.write("Nenhum descritor selecionado")

# Função de filtros passa-baixa
def filtro_passa_baixa(img, tipo):
    """
    Aplica um filtro passa-baixa na imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        tipo (str): Tipo de filtro passa-baixa selecionado pelo usuário.
    Returns:
        numpy.ndarray: Imagem resultante após a aplicação do filtro passa-baixa.
    """

    if tipo == "Média":
        return cv2.blur(img, (5, 5))
    elif tipo == "Mediana":
        return cv2.medianBlur(img, 5)
    elif tipo == "Gaussiano":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif tipo == "Máximo":
        return cv2.dilate(img, np.ones((5, 5), np.uint8))
    elif tipo == "Mínimo":
        return cv2.erode(img, np.ones((5, 5), np.uint8))
    return img

# Função de filtros passa-alta
def filtro_passa_alta(img, tipo):
    """
    Aplica um filtro passa-alta na imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        tipo (str): Tipo de filtro passa-alta selecionado pelo usuário.
    Returns:
        numpy.ndarray: Imagem resultante após a aplicação do filtro passa-alta.
    """
    if tipo == "Laplaciano":
        return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
    elif tipo == "Roberts":
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        imgx = cv2.filter2D(img, -1, kernelx)
        imgy = cv2.filter2D(img, -1, kernely)
        return cv2.convertScaleAbs(imgx + imgy)
    elif tipo == "Prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        imgx = cv2.filter2D(img, -1, kernelx)
        imgy = cv2.filter2D(img, -1, kernely)
        return cv2.convertScaleAbs(imgx + imgy)
    elif tipo == "Sobel":
        imgx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        imgy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(imgx + imgy)
    return img

# Função de filtro personalizado
def filtro_personalizado(img, kernel):
    """
    Aplica um filtro personalizado na imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        kernel (list): Kernel do filtro personalizado.
    Returns:
        numpy.ndarray: Imagem resultante após a aplicação do filtro.
    """
    
    kernel = np.array(kernel, dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

# Função para aplicar máscara na imagem original
def aplicar_mascara_na_original(img_original, mascara):
    """
    Aplica uma máscara binária na imagem original.
    Args:
        img_original (numpy.ndarray): Imagem original em formato RGB ou escala de cinza.
        mascara (numpy.ndarray): Máscara binária a ser aplicada.
    Returns:
        numpy.ndarray: Imagem resultante com a máscara aplicada.
    """

    mascara_binaria = (mascara > 0).astype(np.uint8)
    if len(img_original.shape) == 3:
        imagem_mascarada = cv2.bitwise_and(img_original, img_original, mask=mascara_binaria)
    else:
        imagem_mascarada = cv2.bitwise_and(img_original, img_original, mask=mascara_binaria)
    return imagem_mascarada

# Função para exibir imagem sem fundo branco
def show_image(image, title, cmap=None):
    """
    Exibe uma imagem em um gráfico sem fundo branco.
    Args:
        image (numpy.ndarray): Imagem a ser exibida.
        title (str): Título da imagem.
        cmap (str, optional): Mapa de cores para imagens em escala de cinza. Default é None.
    Returns:
        None: Exibe a imagem no Streamlit.
    """

    fig, ax = plt.subplots(figsize=(4, 4))  # Mantém todas do mesmo tamanho
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.axis("off")  
    fig.patch.set_alpha(0)  # Remove fundo branco
    st.pyplot(fig)

# Função para exibir histogramas sem fundo branco
def plot_histogram(image, title, normalized=False):
    """
    Plota o histograma de uma imagem.
    Args:
        image (numpy.ndarray): Imagem em escala de cinza ou RGB.
        title (str): Título do histograma.
        normalized (bool): Se True, normaliza o histograma.
    Returns:
        matplotlib.figure.Figure: Figura do histograma.
    """

    hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])

    if normalized:
        hist = hist / (image.shape[0] * image.shape[1])  # Normalização do histograma

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(range(256), hist, color="purple", alpha=0.6)
    ax.plot(hist, color="purple")
    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Intensidade de Pixel", color="white")
    ax.set_ylabel("Número de Pixels" if not normalized else "Frequência Relativa", color="white")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("black")  # Fundo preto no histograma
    fig.patch.set_alpha(0)  # Remove fundo branco da figura
    st.pyplot(fig)

    return fig

# Para imagens OpenCV (ex: imagens processadas)
def convert_to_bytes(img):
    """
    Converte uma imagem OpenCV para bytes.
    Args:
        img (numpy.ndarray): Imagem OpenCV a ser convertida.
    Returns:
        BytesIO: Objeto em bytes pronto para download.
    """
    # Se for imagem RGB (3 canais), converte para BGR antes de salvar
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode(".png", img)
    return BytesIO(img_encoded.tobytes())

# Para gráficos Matplotlib (ex: histogramas)
def convert_figure_to_bytes(fig):
    """
    Converte um gráfico Matplotlib para bytes.
    Args:
        fig (matplotlib.figure.Figure): Figura Matplotlib a ser convertida.
    Returns:
        BytesIO: Objeto em bytes pronto para download.
    """

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    return buf

# Função para converter objetos para download
def convert_for_download(obj):
    """
    Converte um objeto (imagem ou gráfico) para bytes para download.
    Args:
        obj: Objeto a ser convertido (imagem OpenCV ou gráfico Matplotlib).
    Returns:
        BytesIO: Objeto em bytes pronto para download.
    """

    # 🔹 Se for um objeto Matplotlib (gráfico)
    if isinstance(obj, plt.Figure):
        return convert_figure_to_bytes(obj)
    return convert_to_bytes(obj)

# Função de limiarização iterativa
def limiarizacao_iterativa(img, delta_min=1):
    """
    Limiarização iterativa de uma imagem.
    Args:
        img (numpy.ndarray): Imagem em escala de cinza.
        delta_min (float): Delta mínimo para convergência.
    Returns:
        numpy.ndarray: Imagem binária resultante.
    """

    T = np.mean(img)
    while True:
        G1 = img[img > T]
        G2 = img[img <= T]
        if len(G1) == 0 or len(G2) == 0:
            break
        mu1 = np.mean(G1)
        mu2 = np.mean(G2)
        T_novo = (mu1 + mu2) / 2
        if abs(T - T_novo) < delta_min:
            break
        T = T_novo
    _, binaria = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return binaria

# Índice de separação de Otsu
def otsu_separation_index(img_gray):
    """
    Calcula o índice de separação de Otsu para uma imagem em escala de cinza.
    Args:
        img_gray (numpy.ndarray): Imagem em escala de cinza.
    Returns:
        float: Índice de separação de Otsu.
    """

    # Histograma normalizado
    hist = cv2.calcHist([img_gray], [0], None, [256], [0,256]).ravel()
    hist_norm = hist / hist.sum()
    # Probabilidades acumuladas
    p1 = np.cumsum(hist_norm)
    # Médias acumuladas
    medias = np.cumsum(hist_norm * np.arange(256))
    # Média global
    m_g = medias[-1]
    # Variância entre classes para cada limiar
    variancia_classes = (m_g * p1 - medias)**2 / (p1 * (1 - p1) + 1e-10)
    # Variância global
    variancia_global = np.sum(((np.arange(256) - m_g) ** 2) * hist_norm)
    # k* = índices onde sigma_b2 é máxima
    max_sigma = np.max(variancia_classes)
    k_max = np.where(variancia_classes == max_sigma)[0]
    k_star = int(np.mean(k_max))
    # Índice de separação
    return variancia_classes[k_star]/variancia_global if variancia_global != 0 else 0

# Função de segmentação
def segmentacao(img, tipo, limiar=0):
    """
    Função para segmentar uma imagem com base no tipo de segmentação selecionado.
    Args:
        img (numpy.ndarray): Imagem em escala de cinza.
        tipo (str): Tipo de segmentação selecionado pelo usuário.
        limiar (int): Valor do limiar para segmentação, se aplicável.
    Returns:
        numpy.ndarray: Imagem segmentada.
    """

    if tipo == "Limiarização Simples":
        _, res = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)
        return res
    elif tipo == "Limiarização de Otsu":
        _, res = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res
    elif tipo == "Canny":
        return cv2.Canny(img, 100, 200)
    elif tipo == "Limiarização Local (Adaptativa)":
        res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        return res
    elif tipo == "Limiarização de Otsu Adaptativa (Local)":
        res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        return res
    elif tipo == "Limiarização Iterativa":
       
        delta_min = st.sidebar.number_input("Delta Mínimo", value=0.01, step=0.01)
        return limiarizacao_iterativa(img,delta_min)
    
    
    return img

# Função de morfologia
def morfologia(img, tipo):
    """
    Função para aplicar operações morfológicas em uma imagem binária.
    Args:
        img (numpy.ndarray): Imagem binária.
        tipo (str): Tipo de operação morfológica selecionada pelo usuário.
    Returns:
        numpy.ndarray: Imagem resultante após a operação morfológica.
    """

    kernel = np.ones((5, 5), np.uint8)
    if tipo == "Abertura":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif tipo == "Fechamento":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif tipo == "Erosão":
        return cv2.erode(img, kernel, iterations=1)
    elif tipo == "Dilatação":
        return cv2.dilate(img, kernel, iterations=1)
    elif tipo == "Hit or Miss":
        kernel_hit_or_miss = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_hit_or_miss)
    return img

# Função para selecionar objetos na imagem binária
def selecionar_objeto(img, metodo):
    """
    Função para selecionar objetos em uma imagem binária com base no método selecionado.
    Args:
        img (numpy.ndarray): Imagem binária.
        metodo (str): Método de seleção de objetos selecionado pelo usuário.
    Returns:
        numpy.ndarray: Imagem resultante com o objeto selecionado.
    """

    resultado = img.copy()
    if metodo == "Maior Objeto":
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [maior], -1, 255, -1)
            resultado = mask
    elif metodo == "Objeto Central":
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            h, w = img.shape
            centro = (w // 2, h // 2)
            min_dist = float("inf")
            idx_central = -1
            for i, cnt in enumerate(contornos):
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.sqrt((cx - centro[0]) ** 2 + (cy - centro[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        idx_central = i
            mask = np.zeros_like(img)
            if idx_central != -1:
                cv2.drawContours(mask, [contornos[idx_central]], -1, 255, -1)
                resultado = mask
    elif metodo == "Objetos de Contorno Fechado":
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        for cnt in contornos:
            if cv2.contourArea(cnt) > 0 and cv2.arcLength(cnt, True) > 0:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        resultado = mask
    return resultado

# Função para aplicar filtro negativo
def filtro_negativo(img):
    """
    Aplica um filtro negativo na imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
    Returns:
        numpy.ndarray: Imagem resultante após a aplicação do filtro negativo.
    """
    return 255 - img if len(img.shape) == 3 else 255 - img[:, :, np.newaxis]

# Função para adicionar ruído gaussiano
def ruido_gaussiano(img, media=0, sigma=25):
    """
    Adiciona ruído gaussiano à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        media (float): Média do ruído gaussiano.
        sigma (float): Desvio padrão do ruído gaussiano.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído gaussiano.
    """
    ruido = np.random.normal(media, sigma, img.shape)
    img_ruidosa = img.astype(np.float32) + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

# Função para adicionar ruído sal e pimenta
def ruido_sal_pimenta(img, probabilidade=0.05):
    """
    Adiciona ruído sal e pimenta à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        probabilidade (float): Probabilidade de cada pixel ser afetado pelo ruído.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído sal e pimenta.
    """
    ruido = np.random.rand(*img.shape[:2])
    img_ruidosa = img.copy()
    img_ruidosa[ruido < probabilidade / 2] = 0  # Sal
    img_ruidosa[ruido > 1 - probabilidade / 2] = 255  # Pimenta
    return img_ruidosa

# Função para adicionar ruído de Poisson
def ruido_poisson(img):
    """
    Adiciona ruído de Poisson à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído de Poisson.
    """
    ruido = np.random.poisson(img / 255.0 * 30.0) / 30.0 * 255.0
    img_ruidosa = img.astype(np.float32) + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

# Função para adicionar ruído speckle
def ruido_speckle(img):
    """
    Adiciona ruído speckle à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído speckle.
    """
    ruido = np.random.randn(*img.shape) * 0.1 * img
    img_ruidosa = img.astype(np.float32) + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

# Função para adicionar ruido de quantização
def ruido_quantizacao(img, n_bits=8):
    """
    Adiciona ruído de quantização à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        n_bits (int): Número de bits para a quantização.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído de quantização.
    """
    quantization_levels = 2 ** n_bits
    return (img // quantization_levels) * quantization_levels + quantization_levels // 2

# Função para adicionar ruido uniforme
def ruido_uniforme(img, amplitude=10):
    """
    Adiciona ruído uniforme à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        amplitude (int): Amplitude do ruído uniforme.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído uniforme.
    """
    ruido = np.random.uniform(-amplitude, amplitude, img.shape)
    img_ruidosa = img.astype(np.float32) + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

# Função para adicionar ruido
def adicionar_ruido(img, tipo, **kwargs):
    """
    Adiciona ruído à imagem com base no tipo de ruído selecionado.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        tipo (str): Tipo de ruído selecionado pelo usuário.
        **kwargs: Argumentos adicionais para o tipo de ruído.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído.
    """
    
    if tipo == "Gaussiano":
        return ruido_gaussiano(img, **kwargs)
    elif tipo == "Sal e Pimenta":
        return ruido_sal_pimenta(img, **kwargs)
    elif tipo == "Poisson":
        return ruido_poisson(img)
    elif tipo == "Speckle":
        return ruido_speckle(img)
    elif tipo == "Impulsivo":
        return ruido_impulsivo(img, **kwargs)
    elif tipo == "Quantização":
        return ruido_quantizacao(img, **kwargs)
    elif tipo == "Uniforme":
        return ruido_uniforme(img, **kwargs)
    
    return img