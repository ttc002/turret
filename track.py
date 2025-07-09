from collections import defaultdict
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import time
import torch


class Person:
    def __init__(self, track_id, current_time, confidence):
        self.track_id = track_id
        self.confidence = confidence
        self.start_time = current_time
        self.track_history = []
        self.is_safe = None  # None - не проверен, True - безопасный, False - неопознан
        self.face_checked = False
        self.warning_shown = False
        self.last_seen_time = current_time
        
    def update_position(self, x, y, current_time, confidence):
        """Обновляет позицию и время последнего обнаружения"""
        self.track_history.append((float(x), float(y)))
        if len(self.track_history) > 30:  # ограничение истории до 30 кадров
            self.track_history.pop(0)
        self.last_seen_time = current_time
        self.confidence = confidence
        
    def get_tracking_duration(self, current_time):
        """Возвращает время отслеживания в секундах"""
        return current_time - self.start_time
    
    def check_face_safety(self, frame, x, y, w, h, target_encoding, height, width):
        """Проверяет безопасность лица"""
        if self.face_checked or target_encoding is None:
            return
            
        # Извлекаем область лица из бокса
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Обрезаем область лица с небольшим отступом
        face_region = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        
        if face_region.size > 0:
            # Уменьшаем размер для ускорения
            small_face = cv2.resize(face_region, (0, 0), fx=0.5, fy=0.5)
            rgb_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
            
            # Поиск лиц в области
            face_locations = face_recognition.face_locations(rgb_face)
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
            
            # Проверка на совпадение с целевым лицом
            self.is_safe = False
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.6)
                if matches[0]:
                    self.is_safe = True
                    print(f"✅ Человек ID {self.track_id} опознан как БЕЗОПАСНЫЙ")
                    break
            
            self.face_checked = True
            if not self.is_safe:
                print(f"⚠️  Человек ID {self.track_id} НЕ ОПОЗНАН")
    
    def check_long_tracking_warning(self, current_time):
        """Проверяет необходимость предупреждения о длительном отслеживании"""
        if self.get_tracking_duration(current_time) > 10.0 and not self.warning_shown:
            safety_status = "БЕЗОПАСНЫЙ" if self.is_safe else "НЕ ОПОЗНАН"
            print(f"⚠️  Человек ID {self.track_id} ({safety_status}, уверенность: {self.confidence:.2f}) находится в кадре уже {self.get_tracking_duration(current_time):.1f} секунд!")
            self.warning_shown = True
    
    def get_track_color(self):
        """Возвращает цвет трека в зависимости от безопасности"""
        if self.is_safe is True:
            return (0, 255, 0)  # Зеленый для безопасных
        elif self.is_safe is False:
            return (0, 0, 255)  # Красный для неопознанных
        else:
            return (0, 255, 255)  # Желтый для непроверенных
    
    def get_status_text(self):
        """Возвращает текст статуса"""
        if self.is_safe is True:
            return "SAFE"
        elif self.is_safe is False:
            return "UNKNOWN"
        else:
            return "CHECKING"
    
    def draw_track(self, frame):
        """Рисует трек на кадре"""
        if len(self.track_history) > 1:
            points = np.hstack(self.track_history).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=self.get_track_color(), thickness=10)
    
    def draw_status_text(self, frame, x, y, w, h):
        """Рисует текст статуса на кадре"""
        status_text = self.get_status_text()
        text_color = self.get_track_color()
        cv2.putText(frame, f"ID:{self.track_id} {status_text}", 
                   (int(x-w/2), int(y-h/2-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


class PersonTracker:
    def __init__(self, target_encoding):
        self.people = {}  # Словарь для хранения объектов Person
        self.target_encoding = target_encoding
        
    def update_person(self, track_id, x, y, w, h, confidence, current_time, frame, height, width):
        """Обновляет информацию о человеке или создает нового"""
        if track_id not in self.people:
            self.people[track_id] = Person(track_id, current_time, confidence)
        
        person = self.people[track_id]
        person.update_position(x, y, current_time, confidence)
        person.check_face_safety(frame, x, y, w, h, self.target_encoding, height, width)
        person.check_long_tracking_warning(current_time)
        
        return person
    
    def cleanup_old_tracks(self, current_time, threshold=5.0):
        """Удаляет старые треки, которые не обновлялись более threshold секунд"""
        tracks_to_remove = []
        for track_id, person in self.people.items():
            if current_time - person.last_seen_time > threshold:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.people[track_id]
    
    def get_person(self, track_id):
        """Возвращает объект Person по ID"""
        return self.people.get(track_id)
    
    def get_all_people(self):
        """Возвращает всех людей"""
        return self.people.values()
    
    def get_safe_people_count(self):
        """Возвращает количество безопасных людей"""
        return sum(1 for person in self.people.values() if person.is_safe is True)
    
    def get_unknown_people_count(self):
        """Возвращает количество неопознанных людей"""
        return sum(1 for person in self.people.values() if person.is_safe is False)


# Проверка доступности CUDA
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'
print(f"🔧 Используется устройство: {device}")
if cuda_available:
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 CUDA память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Принудительная настройка CUDA для PyTorch
    torch.cuda.set_device(0)
    print(f"🔧 Активное CUDA устройство: {torch.cuda.current_device()}")

# Загрузка модели YOLOv11 с принудительным использованием CUDA
model = YOLO("yolo11n.pt")
if cuda_available:
    # Принудительно перемещаем модель на GPU
    model.model = model.model.to(device)
    print("✅ Модель YOLO принудительно загружена на GPU")
    
    # Тестовый запуск для проверки GPU
    print("🧪 Тестирование GPU...")
    test_tensor = torch.randn(1, 3, 640, 640).to(device)
    print(f"🧪 Тестовый тензор создан на: {test_tensor.device}")
    del test_tensor
    torch.cuda.empty_cache()
else:
    print("⚠️  Модель YOLO загружена на CPU")

# Загрузка изображения целевого лица для распознавания
try:
    target_image = face_recognition.load_image_file("target.jpg")
    target_encoding = face_recognition.face_encodings(target_image)[0]
    print("✅ Целевое лицо загружено успешно")
except Exception as e:
    print(f"❌ Ошибка загрузки целевого лица: {e}")
    target_encoding = None

# Открытие видео файла
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Проверка успешного открытия видео
if not cap.isOpened():
    print(f"Ошибка открытия {video_path}")
    exit()


# Получение FPS видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настройка VideoWriter для сохранения выходного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

# Создание трекера людей
tracker = PersonTracker(target_encoding)

# Дополнительная проверка GPU после загрузки модели
if cuda_available:
    print("🔍 Проверка GPU после загрузки модели:")
    print(f"   - Модель на устройстве: {next(model.model.parameters()).device}")
    print(f"   - Доступна ли CUDA: {torch.cuda.is_available()}")
    print(f"   - Количество GPU: {torch.cuda.device_count()}")
    print(f"   - Активное устройство: {torch.cuda.current_device()}")
    
    # Принудительная очистка кэша
    torch.cuda.empty_cache()
    print("🧹 Кэш GPU очищен перед началом обработки")

# Счетчики для мониторинга производительности
frame_count = 0
start_time = time.time()

# Цикл для обработки каждого кадра видео
while cap.isOpened():
    # Считывание кадра из видео
    success, frame = cap.read()

    if not success:
        print("Конец видео")
        break

    frame_count += 1
    
    # Мониторинг GPU каждые 30 кадров
    if cuda_available and frame_count % 30 == 0:
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**2  # в MB
        gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**2  # в MB
        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time
        
        # Дополнительная информация о GPU
        gpu_util = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'N/A'
        print(f"📊 Кадр {frame_count} | FPS: {fps_current:.1f} | GPU память: {gpu_memory_used:.1f}MB (кэш: {gpu_memory_cached:.1f}MB) | Утилизация: {gpu_util}%")

    # Применение YOLO для отслеживания объектов на кадре с принудительным использованием GPU
    if cuda_available:
        # Явное указание device и настройка для GPU
        results = model.track(frame, persist=True, device=0, half=True)  # half=True для оптимизации
    else:
        results = model.track(frame, persist=True, device='cpu')
    
    # Получение текущего времени для отслеживания длительности
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # время в секундах
    
    # Проверка на наличие объектов
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Получение всех данных об объектах (с проверкой устройства)
        if cuda_available:
            # Принудительное перемещение данных с GPU на CPU
            boxes = results[0].boxes.xywh.detach().cpu()
            track_ids = results[0].boxes.id.int().detach().cpu().tolist()
            confidences = results[0].boxes.conf.detach().cpu().tolist()
            classes = results[0].boxes.cls.int().detach().cpu().tolist()
        else:
            # Обычная обработка для CPU
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
        
        # Фильтрация: оставляем только людей (класс 0) с уверенностью > 0.4
        filtered_data = []
        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            if cls == 0 and conf > 0.4:  # класс 0 - это "person" в COCO dataset
                filtered_data.append((box, track_id, conf))
        
        # Если есть отфильтрованные объекты
        if filtered_data:
            # Визуализация результатов на кадре
            annotated_frame = results[0].plot()

            # Обработка каждого человека
            for box, track_id, conf in filtered_data:
                x, y, w, h = box  # координаты центра и размеры бокса
                
                # Обновление или создание объекта Person
                person = tracker.update_person(track_id, x, y, w, h, conf, current_time, frame, height, width)
                
                # Рисование трека и статуса
                person.draw_track(annotated_frame)
                person.draw_status_text(annotated_frame, x, y, w, h)
            
            # Очистка старых треков
            tracker.cleanup_old_tracks(current_time)
            
            # Добавление общей статистики на кадр
            safe_count = tracker.get_safe_people_count()
            unknown_count = tracker.get_unknown_people_count()
            total_count = len(tracker.people)
            
            cv2.putText(annotated_frame, f"Total: {total_count} | Safe: {safe_count} | Unknown: {unknown_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Отображение кадра
            cv2.imshow("YOLOv11 Tracking", annotated_frame)
            out.write(annotated_frame)  # запись кадра в выходное видео
        else:
            # Если отфильтрованных объектов нет, просто отображаем кадр
            cv2.imshow("YOLOv11 Tracking", frame)
            out.write(frame)  # запись кадра в выходное видео
    else:
        # Если объекты не обнаружены, просто отображаем кадр
        cv2.imshow("YOLOv11 Tracking", frame)
        out.write(frame)  # запись кадра в выходное видео

    # Прерывание цикла при нажатии клавиши 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Освобождение видеозахвата и закрытие всех окон OpenCV
cap.release()
out.release()  # закрытие выходного видеофайла
cv2.destroyAllWindows()

# Очистка GPU памяти
if cuda_available:
    torch.cuda.empty_cache()
    print("🧹 GPU память очищена")

# Итоговая статистика
total_time = time.time() - start_time
avg_fps = frame_count / total_time
print(f"📈 Обработано кадров: {frame_count}")
print(f"⏱️  Общее время: {total_time:.1f} сек")
print(f"📊 Средний FPS: {avg_fps:.1f}")
print("✅ Обработка завершена")
