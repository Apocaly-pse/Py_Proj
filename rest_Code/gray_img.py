from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('cugb_title.png')
img = np.array(img)
if img.ndim == 3:
    img = img[:, :, 0]

plt.imshow(img, cmap=plt.cm.gray_r)
plt.show()
