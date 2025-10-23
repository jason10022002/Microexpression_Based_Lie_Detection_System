import cv2
import dlib
import os
import csv
import numpy as np
import pandas as pd

# 路徑設定
predictor_path = "shape_predictor_68_face_landmarks.dat"
video_root = r"C:\Users\user\CASME2_RAW\CASME2-RAW"
excel_path = r"C:\Users\user\OneDrive\文件\專題_微表情辨識\CASME2-coding-20140508.xlsx"

# 模型初始化
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# 認定為說謊的情緒類別
lie_emotions = ["repression", "disgust", "fear"]

# 距離計算

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 擷取特徵

def extract_features_from_frame(gray):
    faces = detector(gray)
    if not faces:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
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

    return {
        "mlen": mlen,
        "ewid": ewid,
        "AU4": au4,
        "AU20": au20,
        "AU23": au23
    }

# 讀取 Excel 標註
print("讀取標註中...")
df = pd.read_excel(excel_path)
df = df.dropna(subset=["Filename", "ApexFrame", "Estimated Emotion"])

features = []
mlen_list, ewid_list = [], []

print("開始處理影片...")
for i, row in df.iterrows():
    filename = row["Filename"].strip() + ".avi"  
    try:
        apex = int(row["ApexFrame"])
    except ValueError:
        print(f" 無效的 ApexFrame：{row['ApexFrame']}，略過此筆資料 ({filename})")
        continue
   

    emotion = row["Estimated Emotion"].strip().lower()
    subject = str(row["Subject"]).strip()
    label = 1 if emotion in lie_emotions else 0

    subject_dir = f"sub{int(subject):02d}"
    video_path = os.path.join(video_root, subject_dir, filename)
    if not os.path.exists(video_path):
        print(f"找不到影片: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if apex >= total_frames:
        print(f" Apex 幀 {apex} 超出影片總長度 {total_frames}: {filename}")
        cap.release()
        continue
    frame_id = 0
    success = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == apex:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feat = extract_features_from_frame(gray)
            if feat:
                feat["filename"] = filename
                feat["frame"] = apex
                feat["label"] = label
                mlen_list.append(feat["mlen"])
                ewid_list.append(feat["ewid"])
                features.append(feat)
                success = True
            break
        frame_id += 1
    cap.release()
    if not success:
        print(f"擷取失敗: {filename} 第 {apex} 幀")

# 加入 abs 差值
mean_mlen = np.mean(mlen_list)
mean_ewid = np.mean(ewid_list)
for f in features:
    f["abs_mlen"] = abs(f["mlen"] - mean_mlen)
    f["abs_ewid"] = abs(f["ewid"] - mean_ewid)

# 輸出 CSV
with open("casme_features.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "frame", "mlen", "ewid", "abs_mlen", "abs_ewid",
        "AU4", "AU20", "AU23", "label"
    ])
    writer.writeheader()
    writer.writerows(features)

print("特徵擷取完成，已寫入 casme_features.csv")
