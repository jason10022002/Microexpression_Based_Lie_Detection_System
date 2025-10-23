import cv2
import dlib
import numpy as np
import joblib
import csv
import datetime

# 模型與預測器載入
model = joblib.load("lie_detector_model.pkl")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 距離計算
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 特徵擷取
def extract_features(landmarks):
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    face_width = dist(points[0], points[16])
    face_height = dist(points[8], points[27])

    mlen = dist(points[48], points[54])
    ewid_l = dist(points[37], points[41])
    ewid_r = dist(points[43], points[47])
    ewid = (ewid_l + ewid_r) / 2

    au4 = 1 if dist(points[21], points[39]) / face_width < 0.16 else 0
    au20 = 1 if mlen / face_width > 0.43 else 0
    lip_height = dist(points[62], points[66])
    au23 = 1 if lip_height / face_height < 0.012 else 0

    return mlen, ewid, abs(mlen - mean_mlen), abs(ewid - mean_ewid), au4, au20, au23

# 模型訓練時的平均值
mean_mlen = 100
mean_ewid = 10

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化 CSV 紀錄
csv_file = open("realtime_predictions.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "mlen", "ewid", "AU4", "AU20", "AU23", "Probability", "Verdict"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        mlen, ewid, abs_mlen, abs_ewid, au4, au20, au23 = extract_features(landmarks)
        features = np.array([[mlen, ewid, abs_mlen, abs_ewid, au4, au20, au23]])

        proba = model.predict_proba(features)[0][1]  # 說謊的機率

        if proba >= 0.85:
            verdict = "Might be lying"
            color = (0, 0, 255)
        elif proba >= 0.6:
            verdict = "Suspicious"
            color = (0, 165, 255)
        else:
            verdict = "Honest"
            color = (0, 255, 0)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, mlen, ewid, au4, au20, au23, round(proba, 4), verdict])
        csv_file.flush()

        cv2.putText(frame, f"{verdict} ({proba:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Lie Detection - Real Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()