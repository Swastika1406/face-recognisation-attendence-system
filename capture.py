import cv2
import os

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = input("Enter Name: ")

if name == "":
    print("Name required!")
    exit()

# Create dataset folder
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load trained model (for duplicate check)
recognizer = None
names = {}

if os.path.exists("trainer/trainer.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                id, n = line.strip().split(',')
                names[int(id)] = n

# Auto ID
face_id = len(os.listdir("dataset")) // 50 + 1

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # 🔒 Duplicate check
        if recognizer is not None:
            id_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 70:
                existing_name = names.get(id_pred, "Unknown")
                print(f"Face already registered as {existing_name}")
                cam.release()
                cv2.destroyAllWindows()
                exit()

        count += 1

        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg",
                    gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Capturing Faces', img)

    if cv2.waitKey(1) == 13 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()

# Save name
with open("names.txt", "a") as f:
    f.write(f"{face_id},{name}\n")

print(f"{name} registered successfully!")