# 👁️‍🗨️ Smart Face Recognition Attendance System  

A **Python-based Face Recognition Attendance System** that automatically encodes known faces and marks attendance in real time using a webcam.  

Attendance records are stored in a single **CSV file** with timestamps. The system uses **OpenCV** and **face_recognition** library for detection and recognition.  

---

## 📦 Requirements  
- Python 3.8+  
- OpenCV  
- dlib  
- face_recognition  
- pandas  
- numpy  

---

## ✨ Features  
✅ Automatically encodes faces from `known_faces/` folder  
✅ Real-time face detection & recognition using webcam  
✅ Records attendance with **Name & Timestamp**  
✅ Prevents duplicate marking for the same person in a day  
✅ Attendance saved in a single `attendance.csv` file  
✅ Shows **“Already marked today”** message if person is recognized again  
✅ Simple, lightweight, and easy to use  

---

## 🚀 How to Run?  
1. Clone or download this project  
2. Place images of known faces inside the `known_faces/` folder  
   - Example: `Alice.jpg`, `Bob.png`  
     or you can register a new person through webcan while running the program.
     While regiatering follow the instructions 
        - press 'c' to capture
        - press 'q' to quit

3. Run the program:  
   ```bash
   python main.py

4. This system will
    -Encode new faces automatically
    -Open webcam for real-time recognition
    -Mark attendance in attendance.csv
    -Press q to stop the session