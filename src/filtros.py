import cv2
import numpy as np

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
