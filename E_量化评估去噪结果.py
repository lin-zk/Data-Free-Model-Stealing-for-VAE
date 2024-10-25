from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import pandas as pd
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage import img_as_float

result = []
snr = []

Surrogate = ['半MINST', '均匀噪声', '加强噪声']

# 加载原始图像和去噪后的图像
original_img = cv2.imdecode(np.fromfile('实验结果\\无噪图像_测试集.png', dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
original_img = img_as_float(original_img)

denoised_img_Victims = cv2.imdecode(np.fromfile('实验结果\\去噪图片_测试集_受害者推理_半MINST.png', dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
denoised_img_Victims = img_as_float(denoised_img_Victims)

noised_img = cv2.imdecode(np.fromfile('实验结果\\加噪图像_测试集.png', dtype=np.uint8),
                          cv2.IMREAD_GRAYSCALE)
noised_img = img_as_float(noised_img)

snr_pre = 0
snr_vic_1 = 0
for row in range(100):
    for col in range(100):
        # 计算当前小图片的起始坐标
        start_x = col * 28
        start_y = row * 28

        # 截取当前小图片
        tile_image_original = original_img[start_y:start_y + 28, start_x:start_x + 28]
        tile_image_noised = noised_img[start_y:start_y + 28, start_x:start_x + 28]
        tile_image_Victims = denoised_img_Victims[start_y:start_y + 28, start_x:start_x + 28]

        snr_pre += 10 * np.log10(
            np.mean(tile_image_original ** 2) / np.mean((tile_image_noised - tile_image_original) ** 2))
        snr_vic_1 += 10 * np.log10(np.mean(tile_image_original ** 2) / np.mean((tile_image_Victims - tile_image_original) ** 2))

snr_pre = snr_pre / 10000
snr_vic_1 = snr_vic_1 / 10000
snr.append(snr_pre)
snr.append(snr_vic_1)

for S in Surrogate:
    denoised_img_Surrogate = cv2.imdecode(
        np.fromfile('实验结果\\去噪图片_测试集_替代推理_' + S + '.png', dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    denoised_img_Surrogate = img_as_float(denoised_img_Surrogate)
    # 计算 MSE
    mse_value = mse(denoised_img_Victims, denoised_img_Surrogate)
    # 计算 SSIM和信噪比（SNR）
    ssim_value = 0
    snr_value = 0
    for row in range(100):
        for col in range(100):
            # 计算当前小图片的起始坐标
            start_x = col * 28
            start_y = row * 28

            # 截取当前小图片
            tile_image_original = original_img[start_y:start_y + 28, start_x:start_x + 28]
            tile_image_Victims = denoised_img_Victims[start_y:start_y + 28, start_x:start_x + 28]
            tile_image_Surrogate = denoised_img_Surrogate[start_y:start_y + 28, start_x:start_x + 28]

            ssim_value += ssim(tile_image_Victims, tile_image_Surrogate)
            snr_value += 10 * np.log10(
                np.mean(tile_image_original ** 2) / np.mean((tile_image_Surrogate - tile_image_original) ** 2))
    ssim_value = ssim_value / 10000
    snr_value = snr_value / 10000

    snr.append(snr_value)

    result.append([mse_value,  ssim_value])

Surrogate = ['更变模型', 'EMINST', 'FMINST', 'cifar10', 'IID', 'IID_输入是原噪声', 'DVAE_DFEM', 'DAE_DFEM']

# 加载原始图像和去噪后的图像
denoised_img_Victims = cv2.imdecode(np.fromfile('实验结果\\去噪图片_测试集_受害者推理_全MINST.png', dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
denoised_img_Victims = img_as_float(denoised_img_Victims)

snr_vic_2 = 0
for row in range(100):
    for col in range(100):
        # 计算当前小图片的起始坐标
        start_x = col * 28
        start_y = row * 28

        # 截取当前小图片
        tile_image_original = original_img[start_y:start_y + 28, start_x:start_x + 28]
        tile_image_Victims = denoised_img_Victims[start_y:start_y + 28, start_x:start_x + 28]

        snr_vic_2 += 10 * np.log10(np.mean(tile_image_original ** 2) / np.mean((tile_image_Victims - tile_image_original) ** 2))

snr_vic_2 = snr_vic_2 / 10000
snr.append(snr_vic_2)

for S in Surrogate:
    denoised_img_Surrogate = cv2.imdecode(
        np.fromfile('实验结果\\去噪图片_测试集_替代推理_' + S + '.png', dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    denoised_img_Surrogate = img_as_float(denoised_img_Surrogate)
    # 计算 MSE
    mse_value = mse(denoised_img_Victims, denoised_img_Surrogate)
    # 计算 SSIM和信噪比（SNR）
    ssim_value = 0
    snr_value = 0
    for row in range(100):
        for col in range(100):
            # 计算当前小图片的起始坐标
            start_x = col * 28
            start_y = row * 28

            # 截取当前小图片
            tile_image_original = original_img[start_y:start_y + 28, start_x:start_x + 28]
            tile_image_Victims = denoised_img_Victims[start_y:start_y + 28, start_x:start_x + 28]
            tile_image_Surrogate = denoised_img_Surrogate[start_y:start_y + 28, start_x:start_x + 28]

            ssim_value += ssim(tile_image_Victims, tile_image_Surrogate)
            snr_value += 10 * np.log10(
                np.mean(tile_image_original ** 2) / np.mean((tile_image_Surrogate - tile_image_original) ** 2))
    ssim_value = ssim_value / 10000
    snr_value = snr_value / 10000

    snr.append(snr_value)

    result.append([mse_value, ssim_value])

index = ['半MINST', '均匀噪声', '加强噪声', '更变模型', 'EMINST',
         'FMINST', 'cifar10', 'IID', 'IID_DFEM', 'DVAE_DFEM', 'DAE_DFEM']
columns = ['MSE ↓', 'SSIM ↑' ]

df_1 = pd.DataFrame(result, index=index, columns=columns)

index = ['加噪', '半MINST训练的受害者', '半MINST', '均匀噪声', '加强噪声', '全MINST训练的受害者', '更变模型', 'EMINST',
         'FMINST', 'cifar10', 'IID', 'IID_DFEM', 'DVAE_DFEM', 'DAE_DFEM']
columns = ['SNR ↑']

df_2 = pd.DataFrame(np.transpose(snr), index=index, columns=columns).transpose()


with pd.ExcelWriter('实验结果\\量化评估_new.xlsx') as writer:
    df_1.to_excel(writer, sheet_name='Sheet1')  # 将 df1 保存到 Sheet1
    df_2.to_excel(writer, sheet_name='Sheet2')  # 将 df2 保存到 Sheet2