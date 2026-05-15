import cv2
import numpy as np
import tkinter as tk
from datetime import datetime
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists("trainer/trainer.yml"):
    print("Train model first!")
    exit()

recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load names
names = {}
if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f:
            id, name = line.strip().split(',')
            names[int(id)] = name

# Attendance function
def mark_attendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Time\n")

    with open('attendance.csv','r+') as f:
        data = f.readlines()
        nameList = [line.split(',')[0] for line in data]

        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.write(f"{name},{time}\n")

def start_recognition():

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:

        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:

            id, conf = recognizer.predict(gray[y:y+h,x:x+w])

            if conf < 100:
                name = names.get(id, "Unknown")
                mark_attendance(name)
                color = (0,255,0)
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow('Attendance System', img)

        if cv2.waitKey(1)==13:
            break

    cam.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x300")

label = tk.Label(root,text="Face Attendance System",
                 font=("Arial",14))
label.pack(pady=20)

start_btn = tk.Button(root,
                      text="Start Attendance",
                      command=start_recognition,
                      width=20,
                      height=2)
start_btn.pack(pady=20)

exit_btn = tk.Button(root,
                     text="Exit",
                     command=root.quit,
                     width=20,
                     height=2)
exit_btn.pack(pady=20)

root.mainloop()