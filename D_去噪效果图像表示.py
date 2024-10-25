import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Reshape
import keras.backend as K

latent_dim = 8
epsilon_std = 1.0
noise_factor = 0.5
n_digit = 3
n_images = n_digit * 10
digit_size = 28
num_test = 10000


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
(x_train, _), (x_test, y_test) = mnist.load_data()

# 将训练集和测试集的图像合并成一张大图片
test_images = np.concatenate((x_test,), axis=0)

# 将所有图像合并成一张大图片
mosaic = create_mosaic(test_images, num_cols=100)  # 100列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')
plt.imsave('实验结果\\无噪图像_测试集.png', mosaic, cmap='gray')

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# 为训练集和测试集加噪
noise_test = x_test + noise_factor * np.random.randn(*x_test.shape)

# Clip the images to be between 0 and 1
noise_test = np.clip(noise_test, 0., 1.)

test_images = np.concatenate((noise_test,), axis=0)

mosaic = create_mosaic(test_images, num_cols=100)  # 100列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')
plt.imsave('实验结果\\加噪图像_测试集.png', mosaic, cmap='gray')

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

# 评估模型
x_out = []

showidx = np.empty(0, dtype=int)
count_list = [0] * 10
while True:
    random_number = np.random.randint(0, num_test)
    while random_number in showidx:
        random_number = np.random.randint(0, num_test)
    if count_list[y_test[random_number]] != n_digit:
        count_list[y_test[random_number]] += 1
        showidx = np.append(showidx, random_number)
    if count_list == [n_digit] * 10:
        break

vae_Victims = ['半MINST', '全MINST']

for vae_model in vae_Victims:
    model.load_weights('模型\\受害者模型\\DVAE_受害者模型_' + vae_model + '.h5')

    result_victims = model.predict(noise_test)

    # 将训练集和测试集的图像合并成一张大图片
    test_images = np.concatenate((result_victims,), axis=0)

    # 将所有图像合并成一张大图片
    mosaic = create_mosaic(test_images, num_cols=100)  # 100列

    # 显示合并后的大图片
    plt.figure()
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')

    # 保存合并后的大图片
    plt.imsave('实验结果\\去噪图片_测试集_受害者推理_' + vae_model + '.png', mosaic, cmap='gray')

    x_out.append(result_victims[showidx])

vae_Surrogate = ['半MINST', '均匀噪声', '加强噪声', 'EMINST', 'FMINST', 'cifar10', 'IID', 'IID_输入是原噪声', 'DFEM']

for vae_model in vae_Surrogate:
    model.load_weights('模型\\替代模型\\DVAE_替代模型_' + vae_model + '.h5')

    result_victims = model.predict(noise_test)

    # 将训练集和测试集的图像合并成一张大图片
    test_images = np.concatenate((result_victims,), axis=0)

    # 将所有图像合并成一张大图片
    mosaic = create_mosaic(test_images, num_cols=100)  # 100列

    # 显示合并后的大图片
    plt.figure()
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')

    if vae_model == 'DFEM':
        plt.imsave('实验结果\\去噪图片_测试集_替代推理_DVAE_' + vae_model + '.png', mosaic, cmap='gray')
    else:
        # 保存合并后的大图片
        plt.imsave('实验结果\\去噪图片_测试集_替代推理_' + vae_model + '.png', mosaic, cmap='gray')

    x_out.append(result_victims[showidx])

model = load_model('模型\\替代模型\\DAE_替代模型_更变模型.h5')

result_victims = model.predict(noise_test)

# 将训练集和测试集的图像合并成一张大图片
test_images = np.concatenate((result_victims,), axis=0)

# 将所有图像合并成一张大图片
mosaic = create_mosaic(test_images, num_cols=100)  # 100列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')

# 保存合并后的大图片
plt.imsave('实验结果\\去噪图片_测试集_替代推理_更变模型.png', mosaic, cmap='gray')

x_out.append(result_victims[showidx])

model = load_model('模型\\替代模型\\DAE_替代模型_DFEM.h5')

result_victims = model.predict(noise_test)

# 将训练集和测试集的图像合并成一张大图片
test_images = np.concatenate((result_victims,), axis=0)

# 将所有图像合并成一张大图片
mosaic = create_mosaic(test_images, num_cols=100)  # 100列

# 显示合并后的大图片
plt.figure()
plt.imshow(mosaic, cmap='gray')
plt.axis('off')

# 保存合并后的大图片
plt.imsave('实验结果\\去噪图片_测试集_替代推理_DAE_DFEM.png', mosaic, cmap='gray')

x_out.append(result_victims[showidx])

figure = np.zeros((digit_size * 15, digit_size * n_images))

# Display
for i, idx in enumerate(showidx):
    figure[0: 28, i * 28: (i + 1) * 28] = np.reshape(x_test[idx], [28, 28])
    figure[28: 28 * 2, i * 28: (i + 1) * 28] = np.reshape(noise_test[idx], [28, 28])
    figure[28 * 2: 28 * 3, i * 28: (i + 1) * 28] = np.reshape(x_out[0][i], [28, 28])
    figure[28 * 3: 28 * 4, i * 28: (i + 1) * 28] = np.reshape(x_out[2][i], [28, 28])
    figure[28 * 4: 28 * 5, i * 28: (i + 1) * 28] = np.reshape(x_out[3][i], [28, 28])
    figure[28 * 5: 28 * 6, i * 28: (i + 1) * 28] = np.reshape(x_out[4][i], [28, 28])
    figure[28 * 6: 28 * 7, i * 28: (i + 1) * 28] = np.reshape(x_out[1][i], [28, 28])
    figure[28 * 7: 28 * 8, i * 28: (i + 1) * 28] = np.reshape(x_out[11][i], [28, 28])
    figure[28 * 8: 28 * 9, i * 28: (i + 1) * 28] = np.reshape(x_out[5][i], [28, 28])
    figure[28 * 9: 28 * 10, i * 28: (i + 1) * 28] = np.reshape(x_out[6][i], [28, 28])
    figure[28 * 10: 28 * 11, i * 28: (i + 1) * 28] = np.reshape(x_out[7][i], [28, 28])
    figure[28 * 11: 28 * 12, i * 28: (i + 1) * 28] = np.reshape(x_out[8][i], [28, 28])
    figure[28 * 12: 28 * 13, i * 28: (i + 1) * 28] = np.reshape(x_out[9][i], [28, 28])
    figure[28 * 13: 28 * 14, i * 28: (i + 1) * 28] = np.reshape(x_out[10][i], [28, 28])
    figure[28 * 14: 28 * 15, i * 28: (i + 1) * 28] = np.reshape(x_out[12][i], [28, 28])

plt.figure(figsize=(28 * 4, 28 * n_images))
plt.axis('off')
plt.imsave('实验结果\\结果示例.png', figure, cmap='gray')
