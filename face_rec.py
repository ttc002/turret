import cv2
import face_recognition

# Загрузка изображения целевого лица
target_image = face_recognition.load_image_file("target.jpg")
target_encoding = face_recognition.face_encodings(target_image)[0]

# Инициализация камеры
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Уменьшаем размер кадра для ускорения
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Поиск лиц и их кодировок
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Проверка на совпадение с целевым лицом
    match_found = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([target_encoding], face_encoding)
        if matches[0]:
            match_found = True
            break

    # Отображение статуса на экране
    text = "MATCH FOUND" if match_found else "No match"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if match_found else (0, 0, 255), 2)

    # Показ кадра
    cv2.imshow('Face Recognition', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

