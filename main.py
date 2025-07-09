from tools import *

# Configuração do estilo do Matplotlib para evitar fundos brancos
plt.style.use("dark_background")

# Título da Aplicação
st.title("Processamento de Imagem com Histograma")

# Upload da Imagem
uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Ler imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imagem_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    # Verifique se a imagem de entrada atual já está na sessão
    if "entrada_atual" not in st.session_state:
        st.session_state.entrada_atual = imagem_gray.copy()

    # Atualize a entrada atual
    entrada_atual = st.session_state.entrada_atual

    # --- Exibição das imagens originais ---
    st.title("Imagens Originais")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Exibir imagens processadas
    with col1:
        show_image(imagem_rgb, "Imagem Original (RGB)")
        st.download_button("📥 Baixar Imagem Original", data=convert_for_download(imagem_rgb), file_name="imagem_rgb.png", mime="image/png")
    with col2:
        show_image(imagem_gray, "Imagem em Tons de Cinza", cmap="gray")
        st.download_button("📥 Baixar Imagem Cinza", data=convert_for_download(imagem_gray), file_name="imagem_gray.png", mime="image/png")
    
    # Exibir Histogramas
    with col3:
        hist = plot_histogram(imagem_gray, "Histograma Original (Tons de Cinza)")
        st.download_button("📥 Baixar Histograma Original", data=convert_for_download(hist), file_name="histograma.png", mime="image/png")
    with col4:
        hist_norm = plot_histogram(imagem_gray, "Histograma Normalizado", normalized=True)
        st.download_button("📥 Baixar Histograma Normalizado", data=convert_for_download(hist_norm), file_name="histograma_normalizado.png", mime="image/png")
    
    # Descritores
    st.title("Descritor de Imagem")
    st.sidebar.header("Descritor de Imagem")
    descritor = st.sidebar.selectbox(
        "Escolha um descritor:",
        [
            "Nenhum",
            "Cor",
            "Forma",
            "Textura"
        ]
    )

    # Indicie de separação de Otsu
    st.write("Índice de Separação de Otsu:", otsu_separation_index(imagem_gray))

    # Exibir descritor selecionado
    describe_image(imagem_rgb, descritor)

    # Cálculo do espectro de Fourier
    imagem_fft = np.fft.fft2(imagem_gray)
    # Desloca a frequência zero para o centro
    imagem_fft_shifted = np.fft.fftshift(imagem_fft)
    # Calcula o espectro de magnitude
    magnitude_spectrum = np.log(np.abs(imagem_fft_shifted) + 1)

    # Normaliza o espectro de magnitude para exibição
    show_image(magnitude_spectrum, "Espectro de Fourier", cmap="gray")
    st.download_button(
        "📥 Baixar Espectro de Fourier",
        data=convert_for_download(magnitude_spectrum),
        file_name="espectro_fourier.png",
        mime="image/png"
    )
    
   # Inicialize o pipeline na sessão
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []



    # Seleção do tipo de operação
    st.sidebar.title("Operações de Processamento de Imagem")
    tipo = st.sidebar.selectbox("Selecione uma operação", ["Nenhum", "Filtro Passa-Baixa", "Filtro Passa-Alta", "Segmentação", "Morfologia", "Seleção de Objetos", "Ruídos", "Outros Filtros"])

    if tipo == "Filtro Passa-Baixa":
        # Opções de filtro passa-baixa
        op = st.sidebar.selectbox("Filtro Passa-Baixa", ["Média", "Mediana", "Gaussiano", "Máximo", "Mínimo"])
        
        # Preview
        preview_img = filtro_passa_baixa(entrada_atual, op)
        show_image(preview_img, f"Preview: Filtro Passa-Baixa ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Filtro Passa-Baixa",
            data=convert_for_download(preview_img),
            file_name=f"filtro_passa_baixa_{op}.png",
            mime="image/png"
        )

        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            st.session_state.pipeline.append(("Filtro Passa-Baixa", op))

    elif tipo == "Filtro Passa-Alta":
        # Opções de filtro passa-alta
        op = st.sidebar.selectbox("Filtro Passa-Alta", ["Laplaciano", "Roberts", "Prewitt", "Sobel"])
        
        # Preview  
        preview_img = filtro_passa_alta(entrada_atual, op)
        show_image(preview_img, f"Preview: Filtro Passa-Alta ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Filtro Passa-Alta",
            data=convert_for_download(preview_img),
            file_name=f"filtro_passa_alta_{op}.png",
            mime="image/png"
        )

        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            st.session_state.pipeline.append(("Filtro Passa-Alta", op))

    elif tipo == "Segmentação":
        # Opções de segmentação
        op = st.sidebar.selectbox("Segmentação", ["Limiarização Simples", "Limiarização de Otsu", "Canny","Limiarização Local (Adaptativa)", "Limiarização de Otsu Adaptativa", "Limiarização Iterativa"])
        # Limiar para segmentação simples
        limiar = st.sidebar.slider("Limiar", 0, 255, int(np.mean(entrada_atual))) if (op == "Limiarização Simples" or op == "Limiarização Iterativa") else 0
    
        # Preview
        preview_img = segmentacao(entrada_atual, op, limiar)
        show_image(preview_img, f"Preview: Segmentação ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Segmentação",
            data=convert_for_download(preview_img),
            file_name=f"segmentacao_{op}.png",
            mime="image/png"
        )

        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            if limiar is not None:
                # Adiciona operação com limiar se for segmentação simples
                st.session_state.pipeline.append(("Segmentação", op, limiar))
            elif op != "Limiarização Simples":
                # Adiciona operação sem limiar para outros tipos de segmentação
                st.session_state.pipeline.append(("Segmentação", op, None))

    elif tipo == "Morfologia":
        
        # Seleção de operações morfológicas
        op = st.sidebar.selectbox("Operação Morfológica", ["Abertura", "Fechamento", "Erosão", "Dilatação", "Hit or Miss"])
        
        # Preview
        preview_img = morfologia(entrada_atual, op)
        show_image(preview_img, f"Preview: Morfologia ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Morfologia",
            data=convert_for_download(preview_img),
            file_name=f"morfologia_{op}.png",
            mime="image/png"
        )
        
        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            st.session_state.pipeline.append(("Morfologia", op))

    elif tipo == "Seleção de Objetos":
        
        # Seleção de objetos
        op = st.sidebar.selectbox(
            "Seleção de Objetos",
            ["Maior Objeto", "Objeto Central", "Objetos de Contorno Fechado"]
        )
        
        # Preview
        preview_img = selecionar_objeto(entrada_atual, op) if op != "Nenhum" else entrada_atual
        show_image(preview_img, f"Preview: Seleção de Objetos ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Seleção de Objetos",
            data=convert_for_download(preview_img),
            file_name=f"selecionar_objeto_{op}.png",
            mime="image/png"
        )
        
        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            st.session_state.pipeline.append(("Seleção de Objetos", op))
    
    elif tipo == "Ruídos":
        op = st.sidebar.selectbox(
            "Ruídos",
            ["Gaussiano", "Sal e Pimenta", "Poisson", "Speckle", "Uniforme"]
        )
        
        # Preview
        preview_img = adicionar_ruido(entrada_atual, op)
        show_image(preview_img, f"Preview: Ruído ({op})", cmap="gray")
        st.download_button(
            "📥 Baixar Preview Ruído",
            data=convert_for_download(preview_img),
            file_name=f"ruido_{op}.png",
            mime="image/png"
        )

        # Adicionar operação ao pipeline
        if st.sidebar.button("Adicionar operação"):
            st.session_state.pipeline.append(("Ruídos", op))

    elif tipo == "Outros Filtros":
        op = st.sidebar.selectbox(
            "Outros Filtros",
            ["Filtro Negativo", "Filtro Personalizado"]
        )

        if op == "Filtro Negativo":
            # Preview do filtro negativo
            preview_img = filtro_negativo(entrada_atual)
            show_image(preview_img, "Preview: Filtro Negativo", cmap="gray")
            st.download_button(
                "📥 Baixar Preview Filtro Negativo",
                data=convert_for_download(preview_img),
                file_name="filtro_negativo.png",
                mime="image/png"
            )

            # Adicionar operação ao pipeline
            if st.sidebar.button("Adicionar operação"):
                st.session_state.pipeline.append(("Filtro Negativo", None))

        elif op == "Filtro Personalizado":
            # Filtro personalizado
            st.sidebar.subheader("Filtro Personalizado")
            kernel = st.sidebar.text_area("Insira o kernel", value="", placeholder="[[1, 0, -1], [1, 0, -1], [1, 0, -1]]")
            
            if not kernel:
                st.error("Por favor, insira um kernel válido.")
            else:
                try:
                    kernel = eval(kernel)  # Avalia a string como uma lista de listas
                    if isinstance(kernel, list) and all(isinstance(row, list) for row in kernel):
                        preview_img = filtro_personalizado(entrada_atual, kernel)
                        show_image(preview_img, "Preview: Filtro Personalizado", cmap="gray")
                        st.download_button(
                            "📥 Baixar Preview Filtro Personalizado",
                            data=convert_for_download(preview_img),
                            file_name="filtro_personalizado.png",
                            mime="image/png"
                        )
                    else:
                        st.error("O kernel deve ser uma lista de listas.")
                except Exception as e:
                    st.error(f"Erro ao processar o kernel: {e}")

                # Adicionar operação ao pipeline
                if st.sidebar.button("Adicionar operação"):
                    st.session_state.pipeline.append(("Filtro Personalizado", kernel))




    # Mostrar pipeline e opção de remover
    st.write("Operações aplicadas:")
    for i, op in enumerate(st.session_state.pipeline):
        st.write(f"{i+1}. {op}")
        if st.button(f"Remover operação {i+1}", key=f"remover_{i}"):
            st.session_state.pipeline.pop(i)
            st.rerun()

    # Limpar o pipeline
    if st.button("Limpar Pipeline"):
        st.session_state.pipeline = []
        st.rerun()

    # Aplicar pipeline
    img = imagem_gray.copy()
    for op in st.session_state.pipeline:
        if op[0] == "Filtro Passa-Baixa":
            img = filtro_passa_baixa(img, op[1])
        elif op[0] == "Filtro Passa-Alta":
            img = filtro_passa_alta(img, op[1])
        elif op[0] == "Segmentação":
            if len(op) > 2 and op[2] is not None:
                img = segmentacao(img, op[1], op[2])
            else:
                img = segmentacao(img, op[1])
        elif op[0] == "Morfologia":
            img = morfologia(img, op[1])
        elif op[0] == "Seleção de Objetos":
            img = selecionar_objeto(img, op[1])
        elif op[0] == "Ruídos":
            img = adicionar_ruido(img, op[1])
        elif op[0] == "Filtro Negativo":
            img = filtro_negativo(img)
        elif op[0] == "Filtro Personalizado":
            if op[1] is not None:
                img = filtro_personalizado(img, op[1])
            else:
                st.error("Filtro personalizado não possui kernel definido.")

    # Atualizar a imagem de entrada atual no estado da sessão
    st.session_state.entrada_atual = img

    col5, col6 = st.columns(2)

    with col5:
        # Exibir imagem resultante do pipeline
        show_image(img, "Imagem Resultante do Pipeline", cmap="gray")
        st.download_button(
            "📥 Baixar Imagem Resultante do Pipeline",
            data=convert_for_download(img),
            file_name="resultado_pipeline.png",
            mime="image/png"
        )
    with col6:
        # Exibir histograma da imagem resultante
        hist_result = plot_histogram(img, "Histograma da Imagem Resultante", normalized=True)
        st.download_button(
            "📥 Baixar Histograma da Imagem Resultante",
            data=convert_for_download(hist_result),
            file_name="histograma_resultado.png",
            mime="image/png"
        )

    #Aplicar máscara na imagem original
    if st.button("Aplicar Máscara na Imagem Original"):
        mascara = img
        imagem_mascarada = aplicar_mascara_na_original(imagem_rgb, mascara)
        st.write("Índice de Separação de Otsu:", otsu_separation_index(imagem_mascarada))
        show_image(imagem_mascarada, "Imagem Original com Máscara Aplicada")
        st.download_button(
            "📥 Baixar Imagem com Máscara Aplicada",
            data=convert_for_download(imagem_mascarada),
            file_name="imagem_mascarada.png",
            mime="image/png"
        )