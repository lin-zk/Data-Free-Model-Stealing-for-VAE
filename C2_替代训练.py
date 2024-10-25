import scipy.signal as signal
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, Reshape, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

n_images = 6
num_train = 60000
num_test = 10000
epochs = 500

# train the VAE on MNIST digits
x_train = np.load("数据集\\替代数据集\\dataset_train_DVAEout_全MINST.npy")
x_test = np.load("数据集\\替代数据集\\dataset_test_DVAEout_全MINST.npy")
# 对训练集和测试集的图像进行直方图均衡化
equalized_train = np.zeros_like(x_train)
equalized_test = np.zeros_like(x_test)

for i in range(len(x_train)):
    equalized_train[i, :, :, 0] = cv2.equalizeHist((x_train[i, :, :, 0] * 255).astype(np.uint8))

for i in range(len(x_test)):
    equalized_test[i, :, :, 0] = cv2.equalizeHist((x_test[i, :, :, 0] * 255).astype(np.uint8))

x_train = equalized_train.astype('float32') / 255.
x_test = equalized_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# 给数据添加噪声
noise_factor = 0.5
noise_train = x_train + noise_factor * np.random.randn(*x_train.shape)
noise_test = x_test + noise_factor * np.random.randn(*x_test.shape)

noise_train = np.clip(noise_train, 0., 1.)
noise_test = np.clip(noise_test, 0., 1.)

# 输入维度
x_noise = Input(shape=(28, 28, 1))
# 基于卷积和池化的编码器
conv_1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(x_noise)
conv_2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(conv_1)
pool_1 = MaxPooling2D((2, 2))(conv_2)
conv_3 = Conv2D(32, (3, 3), padding='valid', activation='relu')(pool_1)
encoded = MaxPooling2D((2, 2))(conv_3)

fc1 = Flatten()(encoded)
fc1 = Reshape([1, 1, 800])(fc1)

# 基于卷积核上采样的解码器
conv_0T = Conv2DTranspose(128, (1, 1), padding='valid', activation='relu')(fc1)  # 1*1
conv_1T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(conv_0T)  # 3*3
conv_2T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(conv_1T)  # 5*5
conv_3T = Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv_2T)  # 10*10
conv_4T = Conv2DTranspose(48, (3, 3), padding='valid', activation='relu')(conv_3T)  # 12*12
conv_5T = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv_4T)  # 24*24
conv_6T = Conv2DTranspose(16, (3, 3), padding='valid', activation='relu')(conv_5T)  # 26*26
x_out = Conv2DTranspose(1, (3, 3), padding='valid', activation='sigmoid')(conv_6T)  # 28*28
# 搭建模型并编译
autoencoder = Model(x_noise, x_out)
autoencoder.summary()

from keras.utils.vis_utils import plot_model

plot_model(autoencoder, to_file='模型\\替代模型\\DAE_模型架构.png', show_shapes=True)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 对噪声数据进行自编码训练
autoencoder.fit(noise_train, x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=512,
                validation_data=(noise_test, x_test))

# 保存模型
autoencoder.save("模型\\替代模型\\DAE_替代模型_更变模型.h5")

digit_size = 28
figure = np.zeros((digit_size * 4, digit_size * n_images))

# 评估模型

showidx = np.random.randint(0, num_test, n_images)
x_out = autoencoder.predict(noise_test[showidx])

# Display
for i, idx in enumerate(showidx):
    figure[0: 28, i * 28: (i + 1) * 28] = np.reshape(x_test[idx], [28, 28])
    figure[28: 28 * 2, i * 28: (i + 1) * 28] = np.reshape(noise_test[idx], [28, 28])
    figure[28 * 2: 28 * 3, i * 28: (i + 1) * 28] = np.reshape(x_out[i], [28, 28])
    figure[28 * 3: 28 * 4, i * 28: (i + 1) * 28] = signal.medfilt2d(np.reshape(noise_test[idx], [28, 28]), [3, 3])
plt.figure(figsize=(28 * 4, 28 * n_images))
plt.axis('off')
plt.imshow(figure, cmap='Greys_r')
plt.savefig('模型\\替代模型\\DAE_图像结果_更变模型.png')
plt.show()
