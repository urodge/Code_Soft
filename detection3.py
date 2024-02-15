import cv2
import numpy as np
import os
import pyttsx3
import datetime

# Set up the text-to-speech engine
engine = pyttsx3.init()

# Define the path to your main dataset folder
main_dataset_path = r"C:\Users\hiii\OneDrive\Desktop\main_dataset"

# Initialize Training_Data and Labels
Training_Data, Labels = [], []

# Create a mapping from person's name to an integer label
label_mapping = {}

# Assign integer labels to each person
current_label = 0
gender = None
if "Mister." in Labels:
    gender = "Male"

elif "Miss." in Labels:
    gender = "female" 
# Loop through subdirectories in the main dataset folder
for person_folder in os.listdir(main_dataset_path):
    if os.path.isdir(os.path.join(main_dataset_path, person_folder)):
        # List the files in the person's subdirectory
        person_files = [f for f in os.listdir(os.path.join(main_dataset_path, person_folder)) if os.path.isfile(os.path.join(main_dataset_path, person_folder, f))]

        # Assign a label to the person and update the label_mapping
        label_mapping[person_folder] = current_label
        current_label += 1

        # Load images and labels for the person
        for i, file in enumerate(person_files):
            image_path = os.path.join(main_dataset_path, person_folder, file)
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if images is not None:
                Training_Data.append(np.asarray(images, dtype=np.uint8))
                Labels.append(label_mapping[person_folder])  # Use the integer label as the label

# Convert label_mapping to a reverse mapping for recognition
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Create and train the LBPH model
model = cv2.face_LBPHFaceRecognizer.create()
model.train(Training_Data, np.array(Labels))

print("Training complete")

# ... Rest of the code remains the same

face_classifier = cv2.CascadeClassifier('haar_face.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, None
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

def get_greeting():
    current_time = datetime.datetime.now().time()
    if current_time.hour < 12:
        return "Good morning"
    elif 12 <= current_time.hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        if face is not None:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1] / 300)))
                label = result[0]  # Get the label as an integer

                if label in reverse_label_mapping:
                    recognized_name = reverse_label_mapping[label]
                else:
                    recognized_name = "Unknown"

                if confidence > 82:
                    greeting = get_greeting()
                    cv2.putText(image, f"Recognized: {recognized_name}", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    engine.say(f"{greeting}, {recognized_name},{gender}")
                    engine.runAndWait()
                else:
                    cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    engine.say(f"{greeting}")
                    engine.runAndWait()
            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                engine.say(f"{greeting}")
                engine.runAndWait()

        cv2.imshow('Face Detection', image)
    except Exception as e:
        print(f"Error: {str(e)}")
        pass

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()