import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Reshape
import keras.backend as K

latent_dim = 8
epsilon_std = 1.0
noise_factor = 0.5
num_train = 60000  # 训练集数据量
num_test = 10000  # 测试集数据量


def create_mosaic(images, num_cols):
    num_images = images.shape[0]
    num_rows = (num_images + num_cols - 1) // num_cols
    mosaic_shape = (num_rows * images.shape[1], num_cols * images.shape[2])
    mosaic = np.zeros(mosaic_shape, dtype=images.dtype)
    for i in range(num_images):
        row_index = i // num_cols
        col_index = i % num_cols
        image = np.squeeze(images[i])
        mosaic[row_index * images.shape[1]: (row_index + 1) * images.shape[1],
        col_index * images.shape[2]: (col_index + 1) * images.shape[2]] = image

    print(num_images, num_cols, num_rows)

    return mosaic


# train the VAE on MNIST digits
x_train = np.random.randint(0, 256, size=(num_train, 28, 28))
x_test = np.random.randint(0, 256, size=(num_test, 28, 28))

np.save('数据集\\原数据集\\IID_train.npy', x_train)
np.save('数据集\\原数据集\\IID_test.npy', x_test)

# 将训练集和测试集的图像合并成一张大图片
train_and_test_images = np.concatenate((x_train, x_test), axis=0)

# 将所有图像合并成一张大图片
mosaic = create_mosaic(train_and_test_images, num_cols=200)  # 200列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')
plt.imsave('数据集\\替代数据集\\无噪图像_IID.png', mosaic, cmap='gray')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
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
model = Model(x_noise, x_out)

model.load_weights('模型\\受害者模型\\DVAE_受害者模型_全MINST.h5')

dataset_train = model.predict(noise_train)
dataset_test = model.predict(noise_test)

np.save('数据集\\替代数据集\\dataset_train_DVAEout_IID.npy', dataset_train)
np.save('数据集\\替代数据集\\dataset_test_DVAEout_IID.npy', dataset_test)

# 将训练集和测试集的图像合并成一张大图片
train_and_test_images = np.concatenate((dataset_train, dataset_test), axis=0)

# 将所有图像合并成一张大图片
mosaic = create_mosaic(train_and_test_images, num_cols=200)  # 200列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')

# 保存合并后的大图片
plt.imsave('数据集\\替代数据集\\替代数据集图像_IID.png', mosaic, cmap='gray')

os.system('start ' + '数据集\\替代数据集\\无噪图像_IID.png')
os.system('start ' + '数据集\\替代数据集\\替代数据集图像_IID.png')
