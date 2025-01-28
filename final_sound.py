import cv2
import numpy as np
import pygame

# pygame 초기화 및 소리 파일 로드
pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("sound.mp3")

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)  # 혹은 웹캠 인덱스에 맞게 변경 (일반적으로 0 또는 1)

# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("yolo/yolov2-tiny.weights", "yolo/yolov2-tiny.cfg")

# YOLO NETWORK 재구성
classes = []
with open("yolo/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

# 얼굴 탐지기 로드
face_detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# 거리 측정 변수
Known_distance = 7.28  # Inches
Known_width = 5.7  # Inches

# 색상 및 폰트
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (0, 69, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

# Focal Length 계산 함수
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# 거리 계산 함수
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame / 0.3937 

# 얼굴 탐지 함수
def face_data(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    face_width = 0
    for (x, y, w, h) in faces:
        face_width = w
    return face_width, faces

# 참조 이미지에서 Focal Length 계산 
ref_image = cv2.imread("haarcascades/Ref_image.png")
ref_image_face_width, _ = face_data(ref_image)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)

# 시계 방향으로 90도 회전
def rotate_image_clockwise(image):
    transposed_image = cv2.transpose(image)  # 전치
    return cv2.flip(transposed_image, 1)  # 수평 반전

last_detection = False  # 마지막 탐지 상태 변수

# OpenCV 창을 전체화면으로 설정
cv2.namedWindow("YOLOv2", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # 웹캠 프레임
    ret, frame = VideoSignal.read()
    if not ret:
        break  # 비디오 신호를 받지 못하면 종료
    
    # 프레임 회전
    frame = rotate_image_clockwise(frame)
    
    h, w, c = frame.shape
    
    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    
    detection_occurred = False
    
    for i in range(len(boxes)):
        if i in indexes:
            detection_occurred = True
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            
            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    face_width_in_frame, Faces = face_data(frame)
    
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame
            )
            Distance = round(Distance, 2)
            cv2.putText(
                frame,
                f"Distance: {Distance} cm",
                (face_x, face_y - 50),
                fonts,
                0.5,
                (ORANGE),
                2,
            )
    
    # 소리 제어
    if detection_occurred and not last_detection:
        sound.play()
    elif not detection_occurred and last_detection:
        sound.stop()
    
    last_detection = detection_occurred
    
    # 결과 영상 출력
    cv2.imshow("YOLOv2", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

# 해제
VideoSignal.release()
cv2.destroyAllWindows()
pygame.quit()
