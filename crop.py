import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
if __name__ == '__main__':
    path = r'./dataset/0910/enhancement/test/AD&CN&MCI'
    save_path = r'./dataset/0911/enhancement/test/AD&CN&MCI'
    # img = Image.open(path).convert('RGB')
    # tf = transforms.Compose([ transforms.RandomRotation(degrees=90, expand=True)])
    # img = tf(img)
    # img = np.array(img)
    # print(img.shape)
    # plt.imshow(img, cmap='Greys_r')
    # plt.show()

    i = 0
    for file in os.listdir(path):
        i += 1
        file_path = os.path.join(path, file)
        new_path = os.path.join(save_path, file)
        # if i != 50:
        #     continue
        print(file_path)
        img = Image.open(file_path).convert('RGB')


        # tf = transforms.Compose([transforms.RandomAffine(0, scale=(0.8, 0.8))])
        # img = tf(img)
        img = np.array(img)
        # img = img[5:138,15:230]
        print(img.shape)

        # plt.imshow(img, cmap='Greys_r')
        # plt.show()
        # break
        dowm = img.shape[0]
        up = img.shape[1]
        max1 = max(dowm, up)
        dowm = (max1 - dowm) // 2
        up = (max1 - up) // 2
        dowm_zuo, dowm_you = dowm, dowm
        up_zuo, up_you = up, up
        if (max1 - img.shape[0]) % 2 != 0:
            dowm_zuo = dowm_zuo + 1
        if (max1 - img.shape[1]) % 2 != 0:
            up_zuo = up_zuo + 1
        matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),  # 向上填充1个维度，向下填充两个维度
                                            (up_zuo, up_you),  # 向左填充2个维度，向右填充一个维度
                                            (0, 0))  # 通道数不填充
                            , mode="constant",  # 填充模式
                            constant_values=(0, 0))  # 第一个维度（就是向上和向左）填充6，第二个维度（向下和向右）填充5
        print(matrix_pad.shape)
        # plt.imshow(matrix_pad, cmap='Greys_r')
        # plt.show()
        # break
    #     # # print(img.shape)
    #     # index = np.where(img > 50)
    #     # # print(index)
    #     # x = index[0]
    #     # y = index[1]
    #     # max_x = max(x)
    #     # min_x = min(x)
    #     # max_y = max(y)
    #     # min_y = min(y)
    #     # max_x = max_x + 10
    #     # min_x = min_x - 10
    #     # max_y = max_y + 10
    #     # min_y = min_y - 10
    #     # if max_x > img.shape[0]:
    #     #     max_x = img.shape[0]
    #     # if min_x < 0:
    #     #     min_x = 0
    #     # if max_y > img.shape[1]:
    #     #     max_y = img.shape[1]
    #     # if min_y < 0:
    #     #     min_y = 0
    #     # print(min_x, max_x, min_y, max_y)
        img = Image.fromarray(matrix_pad)
    #     # plt.imshow(img, cmap='Greys_r')
    #     # plt.show()
    #     # p = r'dataset/fusai/enhancement/enhancement/train/AD'
    #     # new_path = os.path.join(p, file)
        img.save(new_path)
    #     break
