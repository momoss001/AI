import cv2

import numpy as np

image = np.zeros((3,3),dtype=np.float32)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


image = cv2.imshow("C:/Users/Administrator/Downloads/验证码.jpg")
image = cv2.imread("Rplot01.png")


cv2.imwrite('‪C:/Users/Administrator/Documents/Rplot01.jpg',image)