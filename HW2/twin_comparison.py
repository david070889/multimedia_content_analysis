import cv2
import numpy as np

def calculate_difference(frame1, frame2):
    # 計算兩幀之間的差異
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return np.sum(diff)

def twin_comparison_approach(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    ret, curr_frame = cap.read()
    
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        
        # 比較前兩幀和後兩幀
        diff1 = calculate_difference(prev_frame, curr_frame)
        diff2 = calculate_difference(curr_frame, next_frame)
        
        # 判斷是否為場景變化
        if diff1 > threshold and diff2 > threshold:
            print("Shot change detected.")
        
        prev_frame = curr_frame
        curr_frame = next_frame

    cap.release()

# 設置場景變化檢測的閾值
threshold = 100000  # 需要根據實際視頻調整這個值

# 替換成你的視頻文件路徑
video_path = 'climate_out.mp4'
twin_comparison_approach(video_path)