import cv2
import os

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        # 檢查檔案是否為圖片格式
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                print(filename)


folder_path = 'ngc_out'
images = load_images_from_folder(folder_path)