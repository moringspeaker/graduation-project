import pandas as pd
import os
import cv2
import sys
sys.path.append('..')
from img_gist_feature.utils_gist import GistUtils
import numpy as np
MODE='gray'
def read_photo():
    """
    Read photo data from csv file
    """
    filelis1=[]
    filelis2=[]
    path1='D:/2021川大学习/毕业论文/xgboost_train/bad_sites'
    path2='D:/2021川大学习/毕业论文/xgboost_train/good_sites'
    for home,dirs, files in os.walk(path1):
        for file in files:
            if file.endswith(".png"):
                path = os.path.join(home, file)
                filelis1.append(path)
    for home,dirs, files in os.walk(path2):
        for file in files:
            if file.endswith(".png"):
                path = os.path.join(home, file)
                filelis2.append(path)
    return filelis1,filelis2

def calculate_gist_and_2csv(filelis1,filelis2):
    global MODE
    df_bad = pd.DataFrame()
    df_good = pd.DataFrame()
    time=0
    for i in filelis1:#恶意网站列表
        img = cv2.imread(i)
        gist_helper = GistUtils()
        img_gist = gist_helper.get_gist_vec(img, mode=MODE)
        # gist_blue=img_gist[0][0]
        # gist_green=img_gist[1][0]
        # gist_red=img_gist[2][0]
        gist_vec=img_gist[0]
        if time==0:
            for vec in range(len(gist_vec)):  # insert 1536 columns into dataframe
                df_bad.insert(vec, 'feature' + str(vec), '')
            time=time+1
        df_bad.loc[len(df_bad.index)] = gist_vec.tolist()
    df_bad.insert(0, 'label', 1)
    time=0
    for i in filelis2:#良性网站列表
        img = cv2.imread(i)
        gist_helper = GistUtils()
        img_gist = gist_helper.get_gist_vec(img, mode=MODE)
        # gist_blue=img_gist[0][0]
        # gist_green=img_gist[1][0]
        # gist_red=img_gist[2][0]
        # gist_vec=np.concatenate((gist_blue,gist_green,gist_red),axis=0)
        gist_vec = img_gist[0]
        if time==0:
            for vec in range(len(gist_vec)):  # insert 1536 columns into dataframe
                df_good.insert(vec, 'feature' + str(vec), '')
            time=time+1
        df_good.loc[len(df_good.index)] = gist_vec.tolist()
    df_good.insert(0, 'label', 0)
    df_train = pd.concat([df_bad, df_good])
    df_test = pd.concat([df_bad.loc[347:], df_good.loc[390:]])
    df_train.to_csv('train_final_mix.csv', index=False)
    # df_test.to_csv('test_final_gray.csv', index=False)

if __name__=='__main__':
    filelis1,filelis2=read_photo()
    calculate_gist_and_2csv(filelis1,filelis2)