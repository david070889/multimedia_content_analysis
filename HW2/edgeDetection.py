import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def ECR(frame, prev_frame):
    safe_div = lambda x,y: 0 if y == 0 else x / y #確保不會出現除法錯誤

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray_image, 0, 200)
    dilated = cv2.dilate(edge, np.ones((1, 1)))
    inverted = (255 - dilated)

    gray_image2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    edge2 = cv2.Canny(gray_image2, 0, 200)
    dilated2 = cv2.dilate(edge2, np.ones((1, 1)))
    inverted2 = (255 - dilated2)

    log_and1 = (edge2 & inverted)
    log_and2 = (edge & inverted2)
    
    pixels_sum_new = np.sum(edge)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)

    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))

def process_images_from_dict(folder, hist_diff_arr):
    prev_img = None
    img = None
    for image_name in os.listdir(folder):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            img_path = os.path.join(folder, image_name)
            img = cv2.imread(img_path)
            if prev_img is not None:
                hist_diff = ECR(img, prev_img)
                hist_diff_arr.append(hist_diff)

        prev_img = img

def calculate_precision_recall(actual, predicted):
    TP = 0
    FP = 0
    FN = 0
    
    for predicted_val in predicted:
        for actual_val in actual:
            if isinstance(actual_val, list):  # 如果 a 中的元素是列表（區間）
                if predicted_val in actual_val:  # 檢查 b_val 是否在區間內
                    TP += 1
                    break
            elif predicted_val == actual_val:  # 如果 a 中的元素是數值，直接比較
                TP += 1
                break

    FP = len(predicted) - TP
    FN = len(actual) - TP
            
            
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision, recall

# Replace 'path_to_directory' with the path to the folder containing your images
ecr_arr = []
process_images_from_dict('climate_out', ecr_arr)
predicted = []
for i in range(len(ecr_arr)):
    if ecr_arr[i] >= 0.8:
        predicted.append(i + 2)
print(predicted)
a = [range(455, 479)]
b = [range(542, 579)]
c = [range(608, 645)]
d = [range(675, 697)]
e = [range(774, 799)]
f = [range(886, 887)]

actual = [93, 157, 232, 314, 355, a, b, c, d, e, f, 1021, 1237, 1401, 1555]
precision, recall = calculate_precision_recall(actual, predicted)
print(precision)
print(recall)
# 使用列表的索引作為 x 軸，列表中的數字作為 y 軸
x_axis = list(range(len(ecr_arr)))  # 生成索引列表
y_axis = ecr_arr  # 數據列表
z = [0.7] * len(ecr_arr)
# 繪製圖像
plt.figure()  # 設置圖像大小
plt.plot(x_axis, y_axis, linestyle='-', color='b')  # 繪製折線圖，帶有標記
plt.plot(x_axis, z, linestyle='--', color='orange')
plt.title('Edge Detection Ratio')  # 圖像標題
plt.xlabel('Number of frame')  # x 軸標籤
plt.ylabel('ratio')  # y 軸標籤
plt.grid(True)  # 顯示網格
plt.show()