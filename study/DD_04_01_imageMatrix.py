import sys, os
sys.path.append('C:\\Users\\HOME\\git\\ml\\study')
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

np.set_printoptions(linewidth=200, threshold=1000)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # unsigned integer : 0~255로 맞춰주는 역할
    pil_img.show()

# 60,000 X 1 X 28 X 28 -> 60,000 X 784
(x_train, t_train), (x_text, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label) # 5 (첫번째 숫자 라벨)
print(img.shape) # (784, )
img = img.reshape(28,28) # 형상을 원래 이미지의 크기로 변형
print(img.shape) # (28, 28)
# img_show(img)
# img_show(255-img) # 색이 반전

# PIL 대신 pyplot 으로 이미지 보여주기
import matplotlib.pyplot as plt
(x_train, t_train), (x_text, t_test) = load_mnist(flatten=False, normalize=False)
plt.figure()
plt.imshow(x_train[0][0])
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i][0], cmap=plt.cm.binary)
    plt.xlabel(t_train[i])
plt.show()