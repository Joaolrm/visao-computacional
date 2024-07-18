import cv2

# Constants

# carregando o vídeo
videos = [
    {
        "video": 'video_rodovia.mp4',
        "posicaoDaLinhaHorizontal": 200,
        "posicaoDaLinhaVertical": 280,
        "ladoDesconsiderado": "esquerdo",
        "contagemDeCarros": 0
    }
]

for video in videos:
    videoRodovia = cv2.VideoCapture(video["video"])

    # Subtrator de fundo
    substratorDeFundo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    while videoRodovia.isOpened():
        ret, frame = videoRodovia.read()
        if not ret:
            break

        # Redimensionar o frame para processamento mais rápido
        frame = cv2.resize(frame, (640, 480))

        # Aplicar subtração de fundo
        fundoComSubstracao = substratorDeFundo.apply(frame)

        # Encontrar contornos
        contornos, _ = cv2.findContours(fundoComSubstracao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cortorno in contornos:
            # Ignorar pequenos contornos
            if cv2.contourArea(cortorno) < 500:
                continue

            # Obter o retângulo delimitador
            x, y, w, h = cv2.boundingRect(cortorno)

            #  Escolhe o lado a ser desconsiderado
            if video["ladoDesconsiderado"] == "esquerdo" and x < video["posicaoDaLinhaVertical"]:
                continue
            elif video["ladoDesconsiderado"] == "direito" and x + w > video["posicaoDaLinhaVertical"]:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Verificar se o carro passou pela linha de contagem
            if y >  video["posicaoDaLinhaHorizontal"] and y <  video["posicaoDaLinhaHorizontal"] + 5:
                 video["contagemDeCarros"] += 1

        # Desenhar a linha de contagem
        cv2.line(frame, (0, video["posicaoDaLinhaHorizontal"]), (640, video["posicaoDaLinhaHorizontal"]), (255, 0, 0), 2)

        # Desenhar a linha vertical
        cv2.line(frame, (video["posicaoDaLinhaVertical"], 0), (video["posicaoDaLinhaVertical"], 480), (0, 0, 255), 2)

        # Mostrar a contagem de carros
        cv2.putText(frame, f'Carros: {video["contagemDeCarros"]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar o frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    videoRodovia.release()
    cv2.destroyAllWindows()
