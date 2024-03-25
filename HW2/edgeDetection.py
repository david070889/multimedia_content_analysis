import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def ECR(frame, prev_frame):
    safe_div = lambda x,y: 0 if y == 0 else x / y #確保不會出現除法錯誤

    gray_image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge1 = cv2.Canny(gray_image1, 100, 200)
    inverted = (255 - edge1)

    gray_image2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    edge2 = cv2.Canny(gray_image2, 100, 200)
    inverted2 = (255 - edge2)

    log_and1 = (edge1 & inverted2)
    log_and2 = (edge2 & inverted)
    
    pixels_sum_new = np.sum(gray_image1)
    pixels_sum_old = np.sum(gray_image2)

    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)

    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))

def process_images_from_dict(folder, ecr_arr):
    prev_img = None
    img = None
    for image_name in os.listdir(folder):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            img_path = os.path.join(folder, image_name)
            img = cv2.imread(img_path)
            if prev_img is not None:
                ecr_diff = ECR(img, prev_img)
                ecr_arr.append(ecr_diff)

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
    
if __name__ == '__main__':
    
    #初始化
    ecr_arr = [] 
    predicted = []
 
    # news
    actual_news = [73,235, 301, 370, 452, 861, 1281]

    # climate
    # 這邊的interval 不知道要怎麼處理 用笨方法
    a_climate = [range(455, 479)]
    b_climate = [range(542, 579)]
    c_climate = [range(608, 645)]
    d_climate = [range(675, 698)]
    e_climate = [range(774, 800)]
    f_climate = [range(886, 888)]
    actual_climate= [93, 157, 232, 314, 355, a_climate, b_climate, c_climate, d_climate, e_climate, f_climate, 1021, 1237, 1401, 1555]

    #ngc
    a_ngc = [range(127, 165)]
    b_ngc = [range(196, 254)]
    c_ngc = [range(384, 445)]
    d_ngc = [range(516, 536)]
    e_ngc = [range(540, 574)]
    f_ngc = [range(573, 623)]
    g_ngc = [range(622, 665)]
    h_ngc = [range(728, 749)]
    i_ngc = [range(760, 817)]
    j_ngc = [range(816, 839)]
    k_ngc = [range(840, 852)]
    l_ngc = [range(1003, 1010)]
    m_ngc = [range(1048, 1060)]
    actual_ngc = [a_ngc, b_ngc, 285, 340, 383, c_ngc, 456, d_ngc, e_ngc, f_ngc, g_ngc, 683, 703, 722, h_ngc, i_ngc, j_ngc, k_ngc, 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, l_ngc, 1038, m_ngc]

    ## 4個需要改變的parameter
    folder = 'ngc_out' # 三個分別為 'news_out' 'climate_out' 'ngc_out'
    pic_name = 'ngc' # 三個分別為 'news' 'climate' 'ngc'
    actual = actual_ngc # 三個分別為 actual_news, actual_climate, actual_ngc
    threshold = 1 #這裡可調整threshold選取

    starttime = time.time()
    process_images_from_dict(folder, ecr_arr)
    for i in range(len(ecr_arr)):
        if ecr_arr[i] >= threshold: 
            predicted.append(i + 1) ## 加1的原因是 index i = 0 時，其實為第0張圖片改變為第1張圖片，所以加入offset  
            # news, ngc 要 + 1 
            #climate因為從0001開始 所以 + 2
    print(predicted)
    endtime = time.time()

    precision, recall = calculate_precision_recall(actual, predicted) # 1st參數 決定比較的data set # 三個分別為 actual_news, actual_climate, actual_ngc
    print(f'Performance of edge change ratio method with {pic_name} images')
    print(f'precision :{precision}')
    print(f'recall :{recall}')
    print(f'execution time : {endtime - starttime} sec')
    x_axis = list(range(len(ecr_arr)))  # 生成索引列表
    y_axis = ecr_arr  # 數據列表
    z = [threshold] * len(ecr_arr) #圖片中的threshold

    plt.figure() 
    plt.plot(x_axis, y_axis, linestyle='-', color='b', label = 'difference')
    plt.plot(x_axis, z, linestyle='--', color='orange', label = 'threshold')
    plt.title('edge change ratio difference of ' + pic_name)
    plt.grid(True)  # 顯示網格
    plt.legend(loc='upper right', fontsize='small', frameon=True)
    plt.show()