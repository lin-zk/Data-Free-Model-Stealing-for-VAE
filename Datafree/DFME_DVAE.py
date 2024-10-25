import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from keras.losses import binary_crossentropy, MSE
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model, clone_model
from keras.layers import Input, Dense, Reshape, BatchNormalization, Conv2D, UpSampling2D, Activation
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Reshape
from tensorflow.python.framework.ops import disable_eager_execution
from skimage.metrics import mean_squared_error as mse
from skimage import img_as_float
from keras.optimizers import SGD

disable_eager_execution()

num_train = 100000
epochs = 100
batch_size = 256
original_dim = 784
latent_dim = 8
epochs_S = 20
epochs_G = 1
epsilon_std = 1.0
noise_factor = 0.5

# 生成器部分

nz = 100
ngf = 64
nc = 1
img_size = 28
activation = None
init_size = img_size // 4

(_, __), (x_test, ___) = mnist.load_data()

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# 为训练集和测试集加噪
x_noise_test = x_test + noise_factor * np.random.randn(*x_test.shape)

# Clip the images to be between 0 and 1
x_noise_test = np.clip(x_noise_test, 0., 1.)

# 构建G，做为生成器
G_input = Input(shape=(nz,))

# Linear layer to map input noise to appropriate shape for convolution
G_Dense_1 = Dense(ngf * 2 * init_size ** 2)(G_input)
G_Reshape_1 = Reshape((init_size, init_size, ngf * 2))(G_Dense_1)

# Convolutional blocks
G_BN_1 = BatchNormalization()(G_Reshape_1)
G_Conv2D_1 = Conv2D(ngf * 2, 3, strides=1, padding='same')(G_BN_1)
G_BN_2 = BatchNormalization()(G_Conv2D_1)
G_Activation_1 = Activation('relu')(G_BN_2)

G_UpSampling2D_1 = UpSampling2D()(G_Activation_1)
G_Conv2D_2 = Conv2D(ngf * 2, 3, strides=1, padding='same')(G_UpSampling2D_1)
G_BN_3 = BatchNormalization()(G_Conv2D_2)
G_Activation_2 = Activation('relu')(G_BN_3)

G_UpSampling2D_2 = UpSampling2D()(G_Activation_2)

G_Conv2D_3 = Conv2D(ngf, 3, strides=1, padding='same')(G_UpSampling2D_2)
G_BN_4 = BatchNormalization()(G_Conv2D_3)
G_Activation_3 = Activation('relu')(G_BN_4)
G_Conv2D_4 = Conv2D(nc, 3, strides=1, padding='same')(G_Activation_3)
G_BN_5 = BatchNormalization()(G_Conv2D_4)
G_out = Activation('sigmoid')(G_BN_5)

# 构建S和V，做为判别器
V_input = Input(shape=(28, 28, 1))
V_conv_1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(V_input)
V_conv_2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(V_conv_1)
V_pool_1 = MaxPooling2D((2, 2))(V_conv_2)
V_conv_3 = Conv2D(32, (3, 3), padding='valid', activation='relu')(V_pool_1)
V_pool_2 = MaxPooling2D((2, 2))(V_conv_3)
V_h = Flatten()(V_pool_2)
V_z_mean = Dense(latent_dim)(V_h)
V_z_log_var = Dense(latent_dim)(V_h)


# reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
V_z = Lambda(sampling, output_shape=(latent_dim,))([V_z_mean, V_z_log_var])

# decoder part
# we instantiate these layers separately so as to reuse them later
V_z = Reshape([1, 1, latent_dim])(V_z)
V_conv_0T = Conv2DTranspose(128, (1, 1), padding='valid', activation='relu')(V_z)  # 1*1
V_conv_1T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(V_conv_0T)  # 3*3
V_conv_2T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(V_conv_1T)  # 5*5
V_conv_3T = Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(V_conv_2T)  # 10*10
V_conv_4T = Conv2DTranspose(48, (3, 3), padding='valid', activation='relu')(V_conv_3T)  # 12*12
V_conv_5T = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(V_conv_4T)  # 24*24
V_conv_6T = Conv2DTranspose(16, (3, 3), padding='valid', activation='relu')(V_conv_5T)  # 26*26
V_out = Conv2DTranspose(1, (3, 3), padding='valid', activation='sigmoid')(V_conv_6T)  # 28*28

# 构建S和V，做为判别器
S_input = Input(shape=(28, 28, 1))
S_conv_1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(S_input)
S_conv_2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(S_conv_1)
S_pool_1 = MaxPooling2D((2, 2))(S_conv_2)
S_conv_3 = Conv2D(32, (3, 3), padding='valid', activation='relu')(S_pool_1)
S_pool_2 = MaxPooling2D((2, 2))(S_conv_3)
S_h = Flatten()(S_pool_2)
S_z_mean = Dense(latent_dim)(S_h)
S_z_log_var = Dense(latent_dim)(S_h)


# reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
S_z = Lambda(sampling, output_shape=(latent_dim,))([S_z_mean, S_z_log_var])

# decoder part
# we instantiate these layers separately so as to reuse them later
S_z = Reshape([1, 1, latent_dim])(S_z)
S_conv_0T = Conv2DTranspose(128, (1, 1), padding='valid', activation='relu')(S_z)  # 1*1
S_conv_1T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(S_conv_0T)  # 3*3
S_conv_2T = Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(S_conv_1T)  # 5*5
S_conv_3T = Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', activation='relu')(S_conv_2T)  # 10*10
S_conv_4T = Conv2DTranspose(48, (3, 3), padding='valid', activation='relu')(S_conv_3T)  # 12*12
S_conv_5T = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(S_conv_4T)  # 24*24
S_conv_6T = Conv2DTranspose(16, (3, 3), padding='valid', activation='relu')(S_conv_5T)  # 26*26
S_out = Conv2DTranspose(1, (3, 3), padding='valid', activation='sigmoid')(S_conv_6T)  # 28*28

# 生成器的损失函数
def generator_loss(x_V, x_S):
    mse_V_S = MSE(x_V, x_S)
    return 1 - mse_V_S

G = Model(G_input, G_out)
S = Model(S_input, S_out)
G_to_S_input = G.inputs
S.trainable = False
G_to_S_out = S(G.output)
G_to_S = Model(G_to_S_input, G_to_S_out)

G.compile(optimizer='adam', loss=generator_loss)
G_to_S.compile(optimizer='adam', loss=generator_loss)
S.trainable = True
S.compile(optimizer='adam', loss = 'MSE')

V = Model(V_input, V_out)  # VAE模型有自定义函数，h5文件保存不了，先构建模型再导入参数
V.load_weights('DVAE_受害者模型_全MINST.h5')
V_denoised_images_test = V.predict(x_noise_test)
V_denoised_images_test = img_as_float(V_denoised_images_test)


loss = 1
model_loss = -100
logs_path = '日志\\训练日志_'+ str(num_train) +'.txt'

for epoch in range(epochs):
    '''用噪声生成图片'''
    noise = np.random.normal(0, 1, (num_train, nz))
    generated_images = G.predict(noise)

    '''受害者预测生成去噪图片'''
    equalized_generated = np.zeros_like(generated_images)

    for i in range(len(generated_images)):
        equalized_generated[i, :, :, 0] = cv2.equalizeHist((generated_images[i, :, :, 0] * 255).astype(np.uint8))

    generated_images = equalized_generated.astype('float32') / 255.
    generated_images = np.reshape(generated_images, (len(generated_images), 28, 28, 1))
    denoisde_images = V.predict(generated_images)
    '''训练S'''
    equalized_train = np.zeros_like(denoisde_images)

    for i in range(len(denoisde_images)):
        equalized_train[i, :, :, 0] = cv2.equalizeHist((denoisde_images[i, :, :, 0] * 255).astype(np.uint8))

    x_train = equalized_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    equalized_noise = np.zeros_like(generated_images)

    # # 给数据添加噪声
    noise_train = x_train + noise_factor * np.random.randn(*x_train.shape)
    noise_train = np.clip(noise_train, 0., 1.)

    S.fit(generated_images, x_train,
          shuffle=True,
          epochs=epochs_S,
          batch_size=batch_size)
    '''训练G'''
    noise = np.random.normal(0, 1, (num_train, nz))
    S.trainable = False
    G_to_S.fit(noise, x_train,
               shuffle=True,
               epochs=epochs_G,
               batch_size=batch_size)
    S.trainable = True
    '''评估'''
    S_denoised_images_test = S.predict(x_noise_test)
    S_denoised_images_test = img_as_float(S_denoised_images_test)
    loss = snr_value = 10 * np.log10(np.mean(x_test ** 2) / np.mean((S_denoised_images_test - x_test) ** 2))
    print("MSE between Victims and Surrogate:", loss)
    '''保存模型'''
    if loss > model_loss:
        G.save('模型\\最优\\生成器模型_best_'+ str(num_train) +'.h5')
        S.save('模型\\最优\\替代模型_best_'+ str(num_train) +'.h5')
        model_loss = loss
        epoch_best = epoch
    '''写日志'''
    with open(logs_path, 'a') as file:
        file.write('Now Epoch is ' + str(epoch) + '\n')
        file.write("SNR between Victims and Surrogate:" + str(loss) + '\n')
        file.write("Now saved model's SNR:" + str(model_loss) + '\n')
        file.write("And Epoch is:" + str(epoch_best) + '\n')
with open('SNR数据.txt', 'a') as file:
    file.write('item_once = ' + str(num_train) + '\n')
    file.write('item_total = ' + str(num_train * epochs) + '\n')
    file.write("SNR_final" + str(loss) + '\n')
    file.write("SNR_best is" + str(model_loss) + 'when epoch is' + str(epoch_best) + '\n')

G.save('模型\\生成器模型_' + str(num_train) + '.h5')
S.save('模型\\替代模型_' + str(num_train) + '.h5')