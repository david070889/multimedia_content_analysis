import cv2
import numpy as np


img1 = cv2.imread('test01.jpg')
img2 = cv2.imread('test02.jpg')
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
edge1 = cv2.Canny(gray_img1, 100, 200)
edge2 = cv2.Canny(gray_img2, 100, 200)
inverted2 = (255 - edge2)
log_and1 = (edge1 & inverted2)
print(log_and1)
cv2.imshow("result", log_and1)
cv2.imshow("edge1", edge1)
cv2.imshow("inverted2", inverted2)

cv2.waitKey(0)

# 關閉所有視窗
cv2.destroyAllWindows()