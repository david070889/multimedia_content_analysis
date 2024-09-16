import cv2

# 讀取圖片
image = cv2.imread("dataloader_c23/test/real/007_35_YT.jpg")

# 調整圖片大小
resized_image = cv2.resize(image, (224, 224))

# 保存或顯示圖片
cv2.imwrite("resized_image.jpg", resized_image)