# coding=utf-8

import os
from PIL import Image
import numpy as np
import csv
from keras import utils

'''
       要先自己將同一類別的images放到同一個directory
       EX: 衣服類別組(team3)，就要把50種類別的衣服放進50個資料夾中
       資料夾名稱可以取 01_Checked、02_Floral、...等以利讀取及理解
'''

def LoadCSV():

    path_list_category_img = '../DeepFashion_package/Anno/list_category_img.csv'
    path_list_category_cloth = '../DeepFashion_package/Anno/list_category_cloth.csv'
    path_list_eval_partition = '../DeepFashion_package/Eval/list_eval_partition.csv'

    img_category = []
    cloth_category = []
    set_category = []

    # 訓練圖檔的種類標籤
    with open(path_list_category_img) as csvfile:

        rows = csv.reader(csvfile)

        for row in rows:
            img_category.append(row)


    # 衣服種類名稱
    with open(path_list_category_cloth) as csvfile:

        rows = csv.reader(csvfile)

        for row in rows:
            cloth_category.append(row)

    # 區分訓練與測試資料集
    with open(path_list_eval_partition) as csvfile:

        rows = csv.reader(csvfile)

        for row in rows:
            set_category.append(row)

    return [img_category[2:],cloth_category[2:],set_category[2:]]


def DataPreprocessing():

    # 列出要分類的類別數量(category組:50，attribute組則依attribute的個數自訂)
    nb_classes = 50

    # 可以定義每個class所代表的名稱，For Displaying
    class_name = {
        0: 'Checked',
        1: 'Floral',
        2: 'Graphic',
    }

    # dimensions of our images
    img_width, img_height = 150, 150

    # train & test data path
    # train_data_dir = './Dataset/Train'
    train_data_dir = '../DeepFashion_package/'

    [img_category,cloth_category,set_category] = LoadCSV()

    filename_list = [row[0] for row in img_category]

    # Declare list for training and testing data and its' label
    train_data = []
    test_data = []
    train_label = []
    test_label = []


    # Simple example for loading training data.
    # loading testing data by yourself
    # num_dir=0
    # for root, dirs, files in os.walk(train_data_dir):

    #     for f in files:

    #         [filename, extension] = os.path.splitext(f)

    #         if extension != '.jpg':
    #             continue

    #         print(os.path.join(root,f))
    #         img = Image.open(os.path.join(root,f))
    #         img = img.resize((img_width, img_height))
    #         img = np.array(img)
    #         train_data.append(img)
    #         train_label.append(num_dir)
    #     num_dir += 1

    #     print(num_dir)

    for i in range(len(filename_list)):

        # print(str(i)+':'+train_data_dir+filename_list[i])
        filepath = train_data_dir+filename_list[i]

        try:
            img = Image.open(os.path.abspath(filepath))
        except IOError:
            print('Error in '+filepath)
            continue
        img = img.resize((img_width, img_height))
        img = np.array(img)

        print(set_category[i][1])

        if set_category[i][1]=='train':
            train_data.append(img)
            train_label.append(int(img_category[i][1]))
            print('train:\t'+img_category[i][1])
        else:
            test_data.append(img)
            test_label.append(int(img_category[i][1]))
            print('test:\t'+img_category[i][1])





    # 將data type轉換成 numpy array
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    # 為了使用categorical_cross_entropy loss，我們需要先將label轉換成one-hot encoding
    train_label = utils.to_categorical(range(50), num_classes=50)
    test_label = utils.to_categorical(range(50), num_classes=50)

    # data shape(影像數量, 長, 寬, 深度) RGB深度=3、灰階=1
    print('training data shape : ', train_data.shape)
    print('training label shape :', train_label.shape)

    # 儲存處理好的data
    np.save('./data/train_data.npy',train_data)
    np.save('./data/train_label.npy', train_label)
    np.save('./data/test_data.npy',test_data)
    np.save('./data/test_label.npy', test_label)


if __name__ == '__main__':

    DataPreprocessing()
