import cv2
import mediapipe as mp
import math

# Inicializa MediaPipe para manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Parámetros del objeto virtual
box_size = 60
box_pos = [300, 200]  # posición inicial del cuadrado
dragging = False  # bandera para saber si está siendo arrastrado

# Función para calcular la distancia entre dos puntos
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Inicia la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltea la imagen como un espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convierte a RGB para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lm_list.append((px, py))

            # Dibuja la mano en pantalla
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas del índice (8) y del pulgar (4)
            x1, y1 = lm_list[8]
            x2, y2 = lm_list[4]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Dibuja un círculo entre los dedos
            cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

            # Detecta gesto de "agarre"
            dist = distance(x1, y1, x2, y2)
            if dist < 40:
                # Si el centro está dentro del cuadrado, activar arrastre
                if (box_pos[0] < cx < box_pos[0] + box_size) and (box_pos[1] < cy < box_pos[1] + box_size):
                    dragging = True
            else:
                dragging = False

            # Si está siendo arrastrado, actualizar posición
            if dragging:
                box_pos[0] = cx - box_size // 2
                box_pos[1] = cy - box_size // 2

    # Dibuja el objeto virtual (cuadrado)
    cv2.rectangle(frame, tuple(box_pos),
                  (box_pos[0]+box_size, box_pos[1]+box_size),
                  (255, 0, 255), -1)

    # Mensaje en pantalla
    cv2.putText(frame, "Junta los dedos para arrastrar el objeto", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    # Muestra el frame
    cv2.imshow("Arrastrar y Soltar con la Mano", frame)

    # Salir con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
