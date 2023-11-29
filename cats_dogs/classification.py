import os
import random
import shutil
from shutil import copyfile


# ---------------分類圖片---------------
print(len(os.listdir('tmp/PetImages/Cat')))
print(len(os.listdir('tmp/PetImages/Dog')))

try:
    os.mkdir('tmp/cats-v-dogs')
    os.mkdir('tmp/cats-v-dogs/training')
    os.mkdir('tmp/cats-v-dogs/testing')
    os.mkdir('tmp/cats-v-dogs/training/cats')
    os.mkdir('tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # SOURCE 原來的存放路徑, TRAINING , TESTING, SPLIT_SIZE
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    # print(len(file))

    # 訓練樣本與測試樣本劃分
    training_length = int(len(files) * SPLIT_SIZE) 
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files)) # 函数会随机选择files列表中的元素，但返回的列表长度将与原始列表相同
    training_set = shuffled_set[0:training_length] # 訓練樣本個數
    testing_set = shuffled_set[-testing_length:] # 測試樣本個數

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "tmp/cats-v-dogs/testing/dogs/"

def create_dir(file_dir):
    if os.path.exists(file_dir):
        print("true")
        shutil.rmtree(file_dir) #删除已存在的目錄
        os.makedirs(file_dir) # 再建立
    else:
        os.makedirs(file_dir)

create_dir(TRAINING_CATS_DIR)
create_dir(TESTING_CATS_DIR)
create_dir(TRAINING_DOGS_DIR)
create_dir(TESTING_DOGS_DIR)

split_size = 0.9
split_data(CAT_SOURCE_DIR,TRAINING_CATS_DIR,TESTING_CATS_DIR,split_size)
split_data(DOG_SOURCE_DIR,TRAINING_DOGS_DIR,TESTING_DOGS_DIR,split_size)


print(len(os.listdir('tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('tmp/cats-v-dogs/testing/dogs/')))