import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2
from keras.layers import Input, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Reshape
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

n_images = 6
batch_size = 256
original_dim = 784
latent_dim = 8
epochs = 500
epsilon_std = 1.0
noise_factor = 0.5
num_train = 50000  # 训练集数据量
num_test = 10000  # 测试集数据量

# train the VAE on MNIST digits
x_train = np.load("数据集\\替代数据集\\dataset_train_DVAEout_cifar10.npy")
x_test = np.load("数据集\\替代数据集\\dataset_test_DVAEout_cifar10.npy")

# 对训练集和测试集的图像进行直方图均衡化
equalized_train = np.zeros_like(x_train)
equalized_test = np.zeros_like(x_test)

for i in range(len(x_train)):
    equalized_train[i, :, :, 0] = cv2.equalizeHist((x_train[i, :, :, 0] * 255).astype(np.uint8))

for i in range(len(x_test)):
    equalized_test[i, :, :, 0] = cv2.equalizeHist((x_test[i, :, :, 0] * 255).astype(np.uint8))

# 将像素值重新缩放到0到1之间
x_train = equalized_train.astype('float32') / 255.
x_test = equalized_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# 为训练集和测试集加噪
noise_train = x_train + noise_factor * np.random.randn(*x_train.shape)
noise_test = x_test + noise_factor * np.random.randn(*x_test.shape)

# Clip the images to be between 0 and 1
noise_train = np.clip(noise_train, 0., 1.)
noise_test = np.clip(noise_test, 0., 1.)

# encoder part
x_noise = Input(shape=(28, 28, 1))
conv_1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(x_noise)
conv_2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(conv_1)
pool_1 = MaxPooling2D((2, 2))(conv_2)
conv_3 = Conv2D(32, (3, 3), padding='valid', activation='relu')(pool_1)
pool_2 = MaxPooling2D((2, 2))(conv_3)
h = Flatten()(pool_2)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


# reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder part
# we instantiate these layers separately so as to reuse them later
z = Reshape([1, 1, latent_dim])(z)
conv_0T = Conv2DTranspose(128, (1, 1), padding='valid', activation='relu')(z)  # 1*1
conv_1T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(conv_0T)  # 3*3
conv_2T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(conv_1T)  # 5*5
conv_3T = Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv_2T)  # 10*10
conv_4T = Conv2DTranspose(48, (3, 3), padding='valid', activation='relu')(conv_3T)  # 12*12
conv_5T = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv_4T)  # 24*24
conv_6T = Conv2DTranspose(16, (3, 3), padding='valid', activation='relu')(conv_5T)  # 26*26
x_out = Conv2DTranspose(1, (3, 3), padding='valid', activation='sigmoid')(conv_6T)  # 28*28

# instantiate VAE model
vae = Model(x_noise, x_out)
vae.summary()

from keras.utils.vis_utils import plot_model

plot_model(vae, to_file='模型\\替代模型\\DVAE_模型架构.png', show_shapes=True)


# Compute VAE loss
def VAE_loss(x_origin, x_out):
    x_origin = K.flatten(x_origin)
    x_out = K.flatten(x_out)
    xent_loss = original_dim * metrics.binary_crossentropy(x_origin, x_out)  # x_origin是受害者模型的输出，x_out就是模型自己的输出了
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


vae.compile(optimizer='adam', loss=VAE_loss)

vae.fit(noise_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(noise_test, x_test))

vae.save("模型\\替代模型\\DVAE_替代模型_cifar10.h5")

digit_size = 28
figure = np.zeros((digit_size * 4, digit_size * n_images))

# 评估模型

showidx = np.random.randint(0, num_test, n_images)
x_out = vae.predict(noise_test[showidx])

# Display
for i, idx in enumerate(showidx):
    figure[0: 28, i * 28: (i + 1) * 28] = np.reshape(x_test[idx], [28, 28])
    figure[28: 28 * 2, i * 28: (i + 1) * 28] = np.reshape(noise_test[idx], [28, 28])
    figure[28 * 2: 28 * 3, i * 28: (i + 1) * 28] = np.reshape(x_out[i], [28, 28])
    figure[28 * 3: 28 * 4, i * 28: (i + 1) * 28] = signal.medfilt2d(np.reshape(noise_test[idx], [28, 28]), [3, 3])
plt.figure(figsize=(28 * 4, 28 * n_images))
plt.axis('off')
plt.imshow(figure, cmap='Greys_r')
plt.savefig('模型\\替代模型\\DVAE_图像结果_cifar10.png')
plt.show()
