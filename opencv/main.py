import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(
    'data/haarcascade_frontalface_default.xml')

ALPHA = 1.0

def set_alpha(x):
    global ALPHA
    ALPHA = x / 255

cap = cv2.VideoCapture('data/english.mp4')

cv2.namedWindow('frame')
cv2.createTrackbar('ALPHA', 'frame', 255, 255, set_alpha)

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        edge = cv2.Canny(gray[y:y+h, x:x+w], 0, 255)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        blend = cv2.addWeighted(edge, ALPHA, frame[y:y+h, x:x+w, :], (1 - ALPHA), 0.0)
        frame[y:y+h, x:x+w, :] = blend

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
