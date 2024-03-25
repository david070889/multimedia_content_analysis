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

def process_images_from_dict(folder, hist_diff_arr):
    prev_img = None
    img = None
    for image_name in os.listdir(folder):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            img_path = os.path.join(folder, image_name)
            img = cv2.imread(img_path)
            if prev_img is not None:
                hist_diff = gray_hist_compare(img, prev_img)
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

hist_diff_arr = []
predicted = []
folder = 'ngc_out'
process_images_from_dict(folder, hist_diff_arr)
for i in range(len(hist_diff_arr)):
    if hist_diff_arr[i] <= 0.4:
        predicted.append(i + 2)
print(predicted)

# news
# actual = [73,235, 301, 370, 452, 861, 1281]

# climate
# pic_name = 'climate'
# a = [range(455, 479)]
# b = [range(542, 579)]
# c = [range(608, 645)]
# d = [range(675, 697)]
# e = [range(774, 799)]
# f = [range(886, 887)]
# actual = [93, 157, 232, 314, 355, a, b, c, d, e, f, 1021, 1237, 1401, 1555]

#ngc
pic_name = 'ngc'
a = [range(127, 164)]
b = [range(196, 253)]
c = [range(384, 444)]
d = [range(516, 535)]
e = [range(540, 573)]
f = [range(573, 622)]
g = [range(622, 664)]
h = [range(728, 749)]
i = [range(760, 816)]
j = [range(816, 838)]
k = [range(840, 851)]
l = [range(1003, 1009)]
m = [range(1048, 1059)]
actual = [a, b, 285, 340, 383, c, 456, d, e, f, g, 683, 703, 722, h, i, j, k, 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, l, 1038, m]
precision, recall = calculate_precision_recall(actual, predicted)
print(f'Performance of {pic_name} images with gray_histogram_difference')
print(f'precision :{precision}')
print(f'recall :{recall}')
x_axis = list(range(len(hist_diff_arr)))  # 生成索引列表
y_axis = hist_diff_arr  # 數據列表
z = [0.8] * len(hist_diff_arr)
plt.figure() 
plt.plot(x_axis, y_axis, linestyle='-', color='b', label = 'similarity')
plt.plot(x_axis, z, linestyle='--', color='orange', label = 'threshold')
plt.title('Grayscale Histogram difference')
plt.grid(True)  # 顯示網格
plt.legend(loc='lower right', fontsize='small', frameon=True)
plt.show()
