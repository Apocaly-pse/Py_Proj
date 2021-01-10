import numpy
import cv2


def laplace2(img):
    g = numpy.array(((0, 1, 0), (1, -4, 1), (0, 1, 0)))
    # g = numpy.array(((1, 1, 1), (1, -8, 1), (1, 1, 1)))
    re = numpy.zeros_like(img)  # 生成与img相同shape的全0数组
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            re[i, j] = (img[i - 1: i + 2, j - 1: j + 2] * g).sum()  # +img[i,j]
    cv2.imwrite('./' + '1.jpg', re)
    return re


img = cv2.imread('cugb_title.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplace2(img)
