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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
from datetime import datetime
import time
from sklearn.model_selection import GridSearchCV

test = pd.read_csv('data/jdata_user_cate_shop_predict.csv')

users1 = test.ix[:, 'user_id'].to_frame()
cates = test.ix[:, 'cate'].to_frame()
shop = test.ix[:, 'shop_id'].to_frame()
users = test['user_id'].copy()
test.drop('user_id', axis=1, inplace=True)
test.drop('cate', axis=1, inplace=True)
test.drop('shop_id', axis=1, inplace=True)
# test.drop(['cate_shop_view','cate_shop_buy','cate_shop_gz','cate_shop_comment','cate_shop_cart'], axis=1, inplace = True)
tar = xgb.Booster(model_file='./bst.model')
dtest = xgb.DMatrix(test)
preds = tar.predict(dtest)

preds = np.array(preds)
pred = pd.DataFrame(preds, columns=['pred'])

finalresult1 = pd.concat([users1, cates, shop, pred], axis=1)
# finalresult1['pred'] = finalresult1['pred'].astype(int)
finalresult1 = finalresult1.ix[finalresult1['pred'] > 0.995]
finalresult1.drop('pred', inplace=True, axis=1)
finalresult1['user_id'] = finalresult1['user_id'].astype(int)
finalresult1['cate'] = finalresult1['cate'].astype(int)
finalresult1['shop_id'] = finalresult1['shop_id'].astype(int)
finalresult1 = finalresult1.drop_duplicates()
print(len(finalresult1))
finalresult1.to_csv('./finalresult1.csv', index=None, encoding='utf-8')
