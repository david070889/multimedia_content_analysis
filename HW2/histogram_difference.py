import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def gray_hist_compare(image1, image2):
    grayImg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    data1 = np.array(grayImg1)
    data2 = np.array(grayImg2)

    hist1, bin_edge = np.histogram(data1, bins=256, range=(0, 255))
    hist2, bin_edge = np.histogram(data2, bins=256, range=(0, 255))

    degree = 0
    for i in range(len(hist1)): # 跑每一條 histogram
        if hist1[i] != hist2[i]: # 不相同
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:degree = degree + 1 # 相同

    degree = degree / len(hist1)
    return degree

def process_images_from_dict(directory, hist_diff_arr):
    images = sorted([img for img in os.listdir(directory) if img.endswith(".jpg") or img.endswith(".png")])
    prev_frame = None
    for image_name in images:
        frame = cv2.imread(os.path.join(directory, image_name))
        if prev_frame is not None:
            hist_diff = gray_hist_compare(frame, prev_frame)
            hist_diff_arr.append(hist_diff)

        prev_frame = frame
        prev_image_name = image_name
    cv2.destroyAllWindows()

hist_diff_arr = []
process_images_from_dict('climate_out', hist_diff_arr)
threshold = []
x_axis = list(range(len(hist_diff_arr)))  # 生成索引列表
y_axis = hist_diff_arr  # 數據列表
plt.figure() 
plt.plot(x_axis, y_axis, linestyle='-', color='b')
plt.grid(True)  # 顯示網格
plt.show()
# plt.figure(figsize=(10, 6))
# plt.bar(range(256), hist1, color='gray')
# plt.title('Grayscale Histogram')
# plt.tight_layout()
# plt.show()