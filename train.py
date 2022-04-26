import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn.metrics import accuracy_score


plt.style.use("ggplot")


def decorate_csv(filename1, filename2):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df1.insert(0, 'lable', 0, allow_duplicates=True)
    df2.insert(0, 'lable', 1, allow_duplicates=True)
    df_train = pd.concat([df2.loc[:219], df1.loc[:219]])
    df_test = pd.concat([df2.loc[219:], df1.loc[219:243]])
    return df_train, df_test

class Booster():

    def read_csv(self):  # load and preprocess data
        """
        Reads in the CSV file and returns a Pandas DataFrame.
        """
        df_train = pd.read_csv('train_final_gray.csv')
        # df_test = pd.read_csv('test_final.csv')
        y_train = df_train.label.values  # 设定标签集
        # print(df_train.columns)
        df_train.drop(['label'], axis=1, inplace=True)  # 删除不需要的列
        x_train = df_train.values  # 设定训练集
        # y_test = df_test.label.values  # 设定标签集
        # df_test.drop(['label'], axis=1, inplace=True)
        # x_test = df_test.values  # 设定测试集
        # dtest=xgb.DMatrix(x_test)
        # print(x_train.shape,y_train.shape,x_test.shape)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtrain.save_binary('dtrain.buffer')
        return dtrain

    def train(self,dtrain,param,num):
        """
        Sets the parameters for the XGBoost model.
        """
        default_param = {'booster':'gbtree','objective': 'binary:logistic','lambda': 2,'eta': 0.3, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 3,'silten': 0, 'gamma': 0.1,'eval_metric':'auc'}
        numroud=100
        p=xgb.cv(param,dtrain,num,nfold=5,metrics={'auc'})
        print(p['test-auc-mean'])#取出迭代最后一次的auc值
        print ( p["test-auc-mean"].max())
        return (p["test-auc-mean"].max(),p["test-auc-mean"].idxmax())


    def save_result(self,bst):
        bst.save_model('0001.model.')
        # dump model
        bst.dump_model('dump.raw.txt')
        pickle.dump(bst, open("pima.pickle.dat", "wb"))

        # dump model with feature map

    def predict(self,bst,y_test,dtest):
        yy_pred = bst.predict(dtest)  #
        print(yy_pred)
        y_pred = [1 if i > 0.5 else 0 for i in yy_pred]
        print(accuracy_score(y_test, y_pred))
        return(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    Boost=Booster()
    dtrain= Boost.read_csv()
    # eta=0.1
    eta=0.3
    max_depth=6
    gamma=0.1

    param = {'booster': 'gbtree',
             'objective': 'binary:logistic',  # 二分类的问题
             'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
             'max_depth': 6,
             'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
             'subsample': 0.7,  # 随机采样训练样本
             'colsample_bytree':0.7,  # 生成树时进行的列采样
             'min_child_weight':3,
             'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
             'eta': 0.3,  # 如同学习率
             'seed': 1000,
             'nthread': 4,  # cpu 线程数
             'eval_metric': 'auc'}
    x=[]
    y=[]
    loc=[]
    result=[]
    gmma=[]
    Gamma=[0,0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    param['gamma']=0.3
    param['eta']='0.11'
    for x_i in range(60,80):
        for y_i in np.arange(0.5,0.7,0.01):
            param['subsample'] = y_i
            res = Boost.train(dtrain, param, x_i)
            result.append(res[0])
            x.append(x_i)
            y.append(y_i)
            print(res)
            # Boost.save_result(bst)
            # Boost.predict(bst,y_test,dtest)
            # rate = Boost.predict(bst, y_test, dtest)
            # result.append(rate)

    fig = plt.figure()
    #创建3d绘图区域
    ax = plt.axes(projection='3d')
    X=np.array(x)
    Y=np.array(y)
    Z=np.array(result)
    # plt.xlim(0.005, 0.15)
    ax.scatter3D(X,Y,Z, cmap='rainbow')
    ax.set_title('3D scattered plot')
    plt.show()
    location=result.index(max(result))
    print(loc)
    print("X=%s;Y=%s;rate=%s"%(x[location],y[location],max(result)))