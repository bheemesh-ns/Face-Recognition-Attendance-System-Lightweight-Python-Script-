import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Folders & Files
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Ensure known_faces folder exists
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load known faces dynamically from known_faces/
# ------------------------------------------------------------
def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    return known_encodings, known_names


# ------------------------------------------------------------
# Mark attendance
# ------------------------------------------------------------
def record_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])

    if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
        df = pd.read_csv(ATTENDANCE_FILE)

        # ✅ Prevent marking multiple times on the same day
        if not ((df["Name"] == name) & (df["Date"] == date)).any():
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            print(f"✅ {name} marked present at {time}")
        else:
            print(f"⚠ {name} already marked for today.")
    else:
        # Create new file with headers
        new_entry.to_csv(ATTENDANCE_FILE, index=False)
        print(f"✅ {name} marked present at {time}")


# ------------------------------------------------------------
# Run live attendance system
# ------------------------------------------------------------
def start_attendance():
    print("\n🔄 Loading known faces...")
    known_encodings, known_names = load_known_faces()
    print(f"✅ Loaded {len(known_names)} known faces.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(
                    face_recognition.face_distance(known_encodings, face_encoding)
                )
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    record_attendance(name)

            # Draw box & label
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Attendance", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Register new user (capture face & save to folder)
# ------------------------------------------------------------
def register_user(name):
    cap = cv2.VideoCapture(0)
    print(f"\n📸 Capturing face for {name}. Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Register User", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Capture image
            path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(path, frame)
            print(f"✅ Image saved for {name} at {path}")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Main menu
# ------------------------------------------------------------
def main():
    while True:
        print("\n==== FACE ATTENDANCE SYSTEM ====")
        print("1. Register New User")
        print("2. Start Attendance")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter name: ").strip()
            if name:
                register_user(name)
            else:
                print("⚠ Name cannot be empty.")
        elif choice == "2":
            start_attendance()
        elif choice == "3":
            print("👋 Exiting...")
            break
        else:
            print("⚠ Invalid choice, try again.")


if __name__ == "__main__":
    main()
