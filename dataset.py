import cv2 
import numpy as np

face_classifier = cv2.CascadeClassifier("haar_face.xml") 

def face_extractor(img):    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray,1.3,5)

    if len(face) == 0:
      return Noner̥

    for(x,y,w,h) in face:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        count = count + 1
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y+h, x:x+w]
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = r"C:\Users\hiii\OneDrive\Desktop\face detection\dataset\\" + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection completed.")