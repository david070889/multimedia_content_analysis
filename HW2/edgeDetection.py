import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def ECR(frame, prev_frame, going, goout):
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

def process_images_from_directory(directory, ecr_arr):
    images = sorted([img for img in os.listdir(directory) if img.endswith(".jpg") or img.endswith(".png")])
    prev_frame = None
    for image_name in images:
        frame = cv2.imread(os.path.join(directory, image_name))
        if prev_frame is not None:
            ecr_value = ECR(frame, prev_frame, frame.shape[1], frame.shape[0])
            ecr_arr.append(ecr_value)
            ##print(f"ECR between {prev_image_name} and {image_name}: {ecr_value}")
            ##combined_image = np.hstack((prev_frame, frame))
            ##cv2.imshow("Comparison", combined_image)
            ##cv2.waitKey(1000) # Waits for 1 second before moving on to the next image pair
        prev_frame = frame
        prev_image_name = image_name

    cv2.destroyAllWindows()

# Replace 'path_to_directory' with the path to the folder containing your images
ecr_arr = []
process_images_from_directory('climate_out', ecr_arr)
threshold = []
for i in range(len(ecr_arr)):
    if ecr_arr[i] >= 0.7:
        threshold.append(i + 2)
print(threshold)
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