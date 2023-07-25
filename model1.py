# -*- coding: utf-8 -*-
import time, datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
from datetime import datetime
import time
from sklearn.model_selection import GridSearchCV


def create_feature_map(features):
    outfile = open(r'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def xgb_model(train_set):
    actions = pd.read_csv(train_set)
    print(actions.groupby('label')[['user_id', 'cate', 'shop_id']].count())
    # read train_set
    # 单纯的删掉模型前一遍训练认为无用的特征（根据特征重要性中不存在的特征）
    # lst_useless = ['brand']
    #
    # actions.drop(lst_useless, inplace=True, axis=1)

    users = actions[['user_id', 'cate', 'shop_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['cate']
    del actions['shop_id']
    del actions['label']
    # 尝试通过设置scale_pos_weight来调整政府比例不均的问题，但是经过采样的正负比为1:10，训练结果反而不如设置为1
    #     ratio = float(np.sum(labels==0)) / np.sum(labels==1)
    #     print ratio

    # write to feature map
    features = list(actions.columns[:])
    print('total features: ', len(features))
    create_feature_map(features)
    # 训练时即传入特征名
    #     features = list(actions.columns.values)

    user_index = users
    training_data = actions
    label = labels
    label = label.astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(training_data.values, label.values, test_size=0.2,
                                                          random_state=0)
    # 尝试通过提前设置传入训练的正负例的权重来改善正负比例不均的问题
    weights = np.zeros(len(y_train))
    weights[y_train==0] = 1
    weights[y_train==1] = 50

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    #     dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    #     dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=features)
    #     dtrain = xgb.DMatrix(training_data.values, label.values)
    param = {'n_estimators': 4000, 'max_depth': 3, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0,
             'colsample_bytree': 0.8, 'scale_pos_weight': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic',
             'eval_metric': 'auc'}
    #     param = {'n_estimators': 4000, 'max_depth': 6, 'seed': 7, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0,
    #              'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'eta': 0.09, 'silent': 1, 'objective': 'binary:logistic',
    #              'eval_metric':'auc'}

    num_round = 300
    #     param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    evallist = [(dtrain, 'train'), (dvalid, 'eval')]
    #     evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    #     evallist = [(dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
    bst.save_model('bst.model')
    return bst, features


bst_xgb, features = xgb_model('data/jdata_user_cate_shop_train_without_cate_shop.csv')
print(bst_xgb.attributes())
