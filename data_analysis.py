import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ACTION_201602_FILE = "data/JData_Action_201602.csv"
# ACTION_201603_FILE = "data/JData_Action_201603.csv"
# ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/jdata_comment.csv"
PRODUCT_FILE = "data/jdata_product.csv"
USER_FILE = "data/jdata_user.csv"
ACTION_FILE = "data/jdata_action_new.csv"

df_action = pd.read_csv("data/jdata_action_my.csv", encoding='utf-8')
df_action.to_csv("data/jdata_action_my.csv", index=None, encoding='utf-8')
print("1")
# df_action = pd.read_csv("data/jdata_0305_0401.csv", encoding='utf-8')
# df_action.to_csv("data/jdata_0305_0401.csv", index=None, encoding='utf-8')
# print("2")
# df_action = pd.read_csv("data/jdata_0312_0408.csv", encoding='utf-8')
# df_action.to_csv("data/jdata_0312_0408.csv", index=None, encoding='utf-8')
# print("3")
# df_action = pd.read_csv("data/jdata_0319_0415.csv", encoding='utf-8')
# df_action.to_csv("data/jdata_0319_0415.csv", index=None, encoding='utf-8')
# print("4")
# df_action = pd.read_csv("data/jdata_test_0326_0401.csv", encoding='utf-8')
# df_action.to_csv("data/jdata_test_0326_0401.csv", index=None, encoding='utf-8')
# print("5")
# print(len(df_action))
# del df_action
# df_action_new = pd.read_csv(ACTION_FILE, encoding='utf-8')
# print(len(df_action_new))
# del df_action_new
# df_action_my = pd.read_csv("data/jdata_action_my.csv", encoding='utf-8')
# print(len(df_action_my))

# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action['type'].hist()
# plt.show()
# df_action = df_action.groupby('type')
# print(df_action['type'].value_counts())
# USER_TABLE_FILE = "data/user_table.csv"
# ITEM_TABLE_FILE = "data/item_table.csv"


# df_user = pd.read_csv(USER_FILE, encoding='utf-8')
# df_user['user_reg_tm'] = pd.to_datetime(df_user['user_reg_tm'])
# df_user_lv = df_user.groupby('user_lv_cd')
# df_user_lv['user_lv_cd'].hist()
# plt.title('user_lv_cd')
# plt.show()
#
# df_user_age = df_user.groupby('age')
# df_user_age['age'].hist()
# plt.title('age')
# plt.show()
#
# df_user_sex = df_user.groupby('sex')
# df_user_sex['sex'].hist()
# plt.title('sex')
# plt.show()
#
# df_user_city_level = df_user.groupby('city_level')
# df_user_city_level['city_level'].hist()
# plt.title('city_level')
# plt.show()
#
# df_user_province = df_user.groupby('province')
# df_user_province['province'].hist()
# plt.title('province')
# plt.show()
