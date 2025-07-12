# Processamento Digital de Imagens

Este projeto é uma aplicação interativa para processamento digital de imagens, desenvolvida em Python utilizando [Streamlit](https://streamlit.io/). Permite o upload de imagens, aplicação de filtros, operações morfológicas, segmentação, análise de descritores (cor, forma e textura), visualização de histogramas e manipulação de um pipeline de operações. O projeto foi desenvolvido como parte da disciplina de Introdução ao Processamento Digital de Imagens.

## Funcionalidades

- **Upload de Imagem:** Suporte a arquivos PNG, JPG e JPEG.
- **Visualização:** Exibição da imagem original (RGB), em tons de cinza e seus histogramas (normal e normalizado).
- **Descritores de Imagem:** 
  - Cor (histograma, média, desvio padrão e momentos)
  - Forma (momentos de Hu, descritores de Fourier, momentos de Zernike, área, perímetro e circularidade)
  - Textura (Haralick, LBP e contornos)
- **Espectro de Fourier:** Visualização do espectro de magnitude.
- **Operações de Processamento:** 
  - Filtros passa-baixa (Média, Mediana, Gaussiano, Máximo, Mínimo)
  - Filtros passa-alta (Laplaciano, Roberts, Prewitt, Sobel)
  - Filtro personalizado (máscara/kernel customizado)
  - Transformada de Fourier
  - Operações no Domínio de Fourier (Filtros passa-baixa, Filtros passa-alta e Filtro personalizado)
  - Filtro Negativo
  - Segmentação (Limiarização simples, Otsu, Canny, Limiarização adaptativa, Otsu Adaptativo, Limiarização iterativa)
  - Morfologia (Abertura, Fechamento, Erosão, Dilatação, Hit or Miss)
  - Seleção de objetos (Maior objeto, objeto central, contornos fechados)
  - Ruídos (Colocar ruídos na imagem)
- **Pipeline:** Permite adicionar, remover e aplicar múltiplas operações sequencialmente.
- **Download:** Baixe imagens e resultados intermediários/finais.
- **Aplicação de Máscara:** Aplique o resultado do pipeline como máscara sobre a imagem original.

## Como Executar

1. **Pré-requisitos:**
   - Python 3.7+

2. **Instale as dependências:**
   ```
   pip install streamlit opencv-python-headless numpy matplotlib mahotas
   ```

3. **Execute o aplicativo:**
   ```
   streamlit run main.py
   ```

   ![Execução do app](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExczhhYjVlNjNzM3BsaTcyYTFnY2x4empybTVza2dua21pMWJnenBtOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/TvOA2NlXu6AVV93gV3/giphy.gif)

4. **Acesse no navegador:**  
   O Streamlit abrirá automaticamente ou acesse [http://localhost:8501](http://localhost:8501).

## Estrutura dos Arquivos

- `main.py` — Interface principal Streamlit e lógica de controle.
- `src` — Pasta com os arquivos das funções utilitárias para filtros, segmentação, morfologia, descritores, exibição e download.
- `exemplo.png` — Imagem de exemplo para testes.
- `index.html` - Interface web em HTML, CSS e JavaScript.

## Interface Web

Acesse em [jluckmay.github.io/PDI](jluckmay.github.io/PDI).

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
---

Sinta-se à vontade para contribuir ou adaptar este projeto!
