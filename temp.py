# -*- coding: utf-8 -*-
import time, datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from datetime import datetime, timedelta

jaction = "jdata/jdata_action.csv"
juser = "new_data/jdata_user_new.csv"


def get_basic_user_feat():
    user = pd.read_csv(juser, encoding='utf-8')
    user['age'] = user['age'].astype(int)
    user['sex'] = user['sex'].astype(int)
    user['user_lv_cd'] = user['user_lv_cd'].astype(int)
    user['city_level'] = user['city_level'].astype(int)
    le = preprocessing.LabelEncoder()
    age_df = le.fit_transform(user['age'])

    age_df = pd.get_dummies(age_df, prefix='age')
    sex_df = pd.get_dummies(user['sex'], prefix='sex')
    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    city_level_df = pd.get_dummies(user['city_level'], prefix='city_level')

    user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df, city_level_df], axis=1)
    return user


def get_action_in_time(all_actions, start_date, end_date):
    start_date = pd.Timestamp(datetime.strptime(start_date, '%Y-%m-%d').date())
    end_date = pd.Timestamp(datetime.strptime(end_date, '%Y-%m-%d').date())
    actions = all_actions[(all_actions.action_time >= start_date) & (all_actions.action_time < end_date)].copy()
    return actions


if __name__ == "__main__":
    # df_action = pd.read_csv('data/jdata_action.csv')
    # df_product = pd.read_csv('data/jdata_product.csv')
    # df_action = pd.merge(df_action, df_product, on='sku_id')
    # print(len(df_action))
    # df_action.to_csv("data/jdata_action_new.csv", index=None, encoding='utf-8')
    all_action = pd.read_csv('data/jdata_action_new.csv')
    print(len(all_action))
    all_action['action_time'] = pd.to_datetime(all_action['action_time'])

    # 测试集的label 测试窗口往后7天
    testaction = get_action_in_time(all_action, '2018-03-26', '2018-04-02')
    testaction = testaction[testaction['type'] == 2]
    testaction = testaction[['user_id', 'cate', 'shop_id']]
    print(len(testaction))
    testaction = testaction.drop_duplicates()
    print(len(testaction))
    testaction.to_csv('data/jdata_test_0326_0401.csv', encoding='utf-8', index=None)

    testaction = get_action_in_time(all_action, '2018-04-02', '2018-04-09')
    testaction = testaction[testaction['type'] == 2]
    testaction = testaction[['user_id', 'cate', 'shop_id']]
    print(len(testaction))
    testaction = testaction.drop_duplicates()
    print(len(testaction))
    testaction.to_csv('data/jdata_test_0402_0408.csv', encoding='utf-8', index=None)

    testaction = get_action_in_time(all_action, '2018-04-09', '2018-04-16')
    testaction = testaction[testaction['type'] == 2]
    testaction = testaction[['user_id', 'cate', 'shop_id']]
    print(len(testaction))
    testaction = testaction.drop_duplicates()
    print(len(testaction))
    testaction.to_csv('data/jdata_test_0409_0415.csv', encoding='utf-8', index=None)

    # all_action = pd.read_csv('data/jdata_action_new.csv')
    # all_action['action_time'] = pd.to_datetime(all_action['action_time'])
    #
    # # 测试集的label 测试窗口往后7天
    # testaction = get_action_in_time(all_action, '2018-02-01', '2018-04-09')
    # testaction = testaction[['cate', 'shop_id', 'user_id','type']]
    # print(testaction)
    # testaction.to_csv('data/jdata_action_train.csv', encoding='utf-8', index=None)
