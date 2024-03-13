import mediapipe as mp
import cv2

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture("v1.mp4")

while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    #print("Height, Width", height, width)

    # Facial Landmarks
    result = face_mesh.process(image)
    #resultRGB = face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0,468):
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * width)
            y = int(pt.y * height)
            #print("x, y", x, y)
            cv2.circle(image, (x ,y), 2, (100, 100, 0), -1)

    cv2.imshow("Image", image)
    #cv2.imshow("Image", rgb_image)
    cv2.waitKey(0)