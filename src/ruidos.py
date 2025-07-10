import numpy as np

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
def ruido_poisson(img, intensidade=30):
    """
    Adiciona ruído de Poisson à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        intensidade (float): Intensidade do ruído de Poisson.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído de Poisson.
    """
    ruido = np.random.poisson(img / 255.0 * intensidade) / intensidade * 255.0
    img_ruidosa = img.astype(np.float32) + ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255).astype(np.uint8)
    return img_ruidosa

# Função para adicionar ruído speckle
def ruido_speckle(img, intensidade=0.1):
    """
    Adiciona ruído speckle à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        intensidade (float): Intensidade do ruído speckle.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído speckle.
    """
    ruido = np.random.randn(*img.shape) * intensidade * img
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
def ruido_uniforme(img, min_val=0, max_val=255):
    """
    Adiciona ruído uniforme à imagem.
    Args:
        img (numpy.ndarray): Imagem em formato RGB ou escala de cinza.
        min_val (int): Valor mínimo do ruído uniforme.
        max_val (int): Valor máximo do ruído uniforme.
    Returns:
        numpy.ndarray: Imagem resultante após a adição do ruído uniforme.
    """
    ruido = np.random.uniform(min_val, max_val, img.shape)
    ruido = ruido.astype(np.float32)
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
    elif tipo == "Quantização":
        return ruido_quantizacao(img, **kwargs)
    elif tipo == "Uniforme":
        return ruido_uniforme(img, **kwargs)
    
    return img