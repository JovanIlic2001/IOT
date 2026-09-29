import cv2
import numpy as np
import urllib.request
import time
import threading
import os
import shutil
import sys

if len(sys.argv) > 1:
    stream_url = sys.argv[1]
else:
    stream_url = 'http://192.168.0.4:8080/video'

alarm_threshold = 3
confidence_threshold = 0.5

prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
person_id = classes.index("person")

net = cv2.dnn.readNetFromCaffe(prototxt, model)


class LatestFrameReader:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        stream = urllib.request.urlopen(self.url)
        buffer = b''
        while self.running:
            buffer += stream.read(65536)
            while True:
                a = buffer.find(b'\xff\xd8')
                b = buffer.find(b'\xff\xd9')
                if a == -1 or b == -1:
                    break
                jpg = buffer[a:b + 2]
                buffer = buffer[b + 2:]

                next_a = buffer.find(b'\xff\xd8')
                next_b = buffer.find(b'\xff\xd9')
                if next_a != -1 and next_b != -1:
                    continue

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


print(f"Povezujem se na: {stream_url}")
reader = LatestFrameReader(stream_url)

if os.path.exists('session_frames'):
    shutil.rmtree('session_frames')
os.makedirs('session_frames', exist_ok=True)
frame_number = 0

print("Cekam prvi frejm...")
while reader.get_frame() is None:
    time.sleep(0.1)
print("Krecem sa obradom.")

while True:
    start_time = time.time()

    frame = reader.get_frame()
    if frame is None:
        continue

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > confidence_threshold and class_id == person_id:
            person_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            startX, startY, endX, endY = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"Osoba: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elapsed = time.time() - start_time
    fps = 1 / elapsed if elapsed > 0 else 0

    cv2.putText(frame, f"Broj osoba: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    alarm = person_count > alarm_threshold
    if alarm:
        cv2.putText(frame, "ALARM! PREVISE OSOBA", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print(f"Osoba: {person_count} | FPS: {fps:.1f} | Vreme obrade: {elapsed*1000:.0f}ms"
          f"{' | [ALARM]' if alarm else ''}")

    frame_number += 1
    filename = f"session_frames/frame_{frame_number:04d}_osoba{person_count}.jpg"
    cv2.imwrite(filename, frame)