import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

ACTION_FILE = "data/jdata_action.csv"
ACTION_201802_FILE = "data/jdata_action2.csv"
ACTION_201803_FILE = "data/jdata_action3.csv"
ACTION_201804_FILE = "data/jdata_action4.csv"
COMMENT_FILE = "data/jdata_comment.csv"
PRODUCT_FILE = "data/jdata_product.csv"
USER_FILE = "data/jdata_user.csv"

# df_user = pd.read_csv("data/jdata_action.csv", encoding='utf-8')
# print(len(df_user))
# # df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
# # df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# # print(len(df_product))
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# type = pd.DataFrame({'type': [2, 5]})
# print(type)
# df_action['type'] = df_action['type'].astype(int)
# df_action = pd.merge(df_action, type, on='type')
# df_action = df_action[df_action['action_time'] >= '2018-04-08']
# df = df_action.groupby(['user_id', 'sku_id', 'type']).count()
# print(df)
# df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# print(df_product[df_product['brand']>1]['brand'].value_counts())
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action = df_action[['user_id', 'type']].groupby('type')
df_shop = pd.read_csv('data/jdata_shop.csv')
del df_shop['shop_reg_tm']
df_shop.to_csv('data/jdata_shop.csv', encoding='utf-8', index=None)
# df_fake = pd.merge(df_product, df_shop, on='shop_id')
# df_fake = df_fake[df_fake['cate_y'].isnull()]
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
# df_shop = df_shop[df_shop['fans_num'].notnull()]
# df_shop = df_shop[df_shop['fans_num'] != 0]
# df_shop = df_shop[df_shop['shop_score'] > 9]
# df_product = df_product.groupby(['cate', 'shop_id'], as_index=False).count()
# df_product = pd.merge(df_product, df_shop, on='shop_id')
# print(df_product)
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action = df_action[df_action['type'] == 2]
# df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# df_action = pd.merge(df_action, df_product, on='sku_id')
# df_action = df_action.groupby(['cate', 'shop_id'], as_index=False).count()
# df_action = df_action[df_action['brand'] > 50]
# print(df_action)
# df_action = df_action.groupby(['cate_x'])
# df_comment = pd.read_csv(COMMENT_FILE, encoding='utf-8')
# df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
# print(len(df_shop))
# df_shop = df_shop.dropna(subset=['shop_reg_tm'])
# print(len(df_shop))
# fake_shop = df_shop[df_shop['vender_id'] == 3666]['shop_id']
# del df_comment['dt']
# df_comment = df_comment.drop_duplicates()
# df_comment = df_comment.groupby('sku_id').count()
# print(len(df_comment))
#
# df_product = df_product
# print(len(df_product))
# print(len(df_action[df_action['type'] == 5]))
# df_act = df_action[df_action['user_id'] == 606851]
# print(df_act)
# df_act = df_action[df_action['user_id'] == 312832]
# print(df_act)
# df_act = df_action[df_action['user_id'] == 203896]
# print(df_act)
# df_act = df_action[df_action['user_id'] == 1491349]
# print(df_act)
# df_act = df_action[df_action['user_id'] == 919002]
# print(df_act)
# # df_comment = pd.read_csv(COMMENT_FILE, encoding='utf-8')
# # print(len(df_comment))
# df_usr = df_user[['user_id', 'province']]
# user = pd.DataFrame({'user_id': df_usr['user_id'].isnull()})
# df = pd.merge(user, df_action, on='user_id')
# print(df)
# df_comment['dt'].hist()
# plt.show()
# df_user1 = df_user[df_user['province'].isnull()]
# print(len(df_user1))
# df_user['user_reg_tm'] = pd.to_datetime(df_user['user_reg_tm'])
# print(df_user['user_id'].value_counts())

# df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
# print(df_shop['shop_id'].value_counts())

# df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# print(df_product['cate'].value_counts())
# print(len(df_shop))
# df_product = pd.merge(df_product, df_shop, on='shop_id')
# pd.set_option('display.max_columns', None)
# print(df_product)
# df_user = pd.read_csv(USER_FILE, encoding='utf-8')
# user_list = pd.DataFrame({'user_id': df_user['user_id'], 'key': 1})
# df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
# cate_list = pd.DataFrame({'cate': df_product['cate'].drop_duplicates(), 'key':1})
# user_cate = pd.merge(user_list, cate_list, on='key')
# del user_cate['key']
# user_cate['user_id'].astype(int)
# user_cate['cate'].astype(int)
# user_cate.sort_values(by=['user_id', 'cate'], ascending=True, inplace=True)
# print(user_cate)
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action = df_action[['user_id', 'type']]
# df_action['count'] = 0
# df_action = df_action.groupby(['user_id', 'type'], as_index=False).agg({'count':'count'})
# print(df_action)

# print(df_action)
# df_action = df_action.groupby(['user_id', 'type'], as_index=False)
# df_action.count()
# # df_action = df_action.merge(df_user, on="user_id")
# #
# # df_action

# df = pd.DataFrame({'key1':list('aabba'),
#                   'key2': ['one','two','one','two','one'],
#                   'data1': np.random.randn(5),
#                   'data2': np.random.randn(5)})
# grouped = df.groupby(['key1', 'key2'], as_index=False).agg({'data1':'count'})
# print(grouped)

# df_action = df_action[df_action['type'] == 2]
# df_action = df_action.drop_duplicates()
# len1=len(df_action)
# df_action = df_action[['user_id', 'sku_id', 'type']]
# print(df_action.groupby(['user_id', 'sku_id']).count())
# df_action = df_action.drop_duplicates()
# len2=len(df_action)
# print(len2/len1)


# print(len(df_action))
# df_action = pd.merge(df_action, df_product, on='sku_id')
# df_action = pd.merge(df_action, df_shop, on='shop_id')
# print(len(df_action))
# pd.set_option('display.max_columns', None)
# print(df_action)

# df_action_equal = df_action[df_action['cate_x'] == df_action['cate_y']]
# print(len(df_action_equal))
#
# df_action_not_equal = df_action[df_action['cate_x'] != df_action['cate_y']]
# print(len(df_action_not_equal))

# def empty_detail(f_path, f_name):
#     df_file = pd.read_csv(f_path)
#     print ('empty info in detail of {0}:'.format(f_name))
#     print (pd.isnull(df_file).any())
#
#
# empty_detail(USER_FILE, 'User')

# df_user_lv = df_user.groupby('user_lv_cd')
# # print(np.unique(df_user['user_lv_cd']))
# df_user_lv['user_lv_cd'].hist()
# plt.title('user_lv_cd')
# plt.show()

# df_user_age = df_user.groupby('age')
# # print(np.unique(df_user['age']))
# df_user_age['age'].hist()
# plt.title('age')
# plt.show()
#
# df_user_age = df_user.groupby('sex')
# # print(np.unique(df_user['sex']))
# df_user_age['sex'].hist()
# plt.title('sex')
# plt.show()
#
# df_user_age = df_user.groupby('city_level')
# # print(np.unique(df_user['city_level']))
# df_user_age['city_level'].hist()
# plt.title('city_level')
# plt.show()
#
# df_user_age = df_user.groupby('province')
# # print(np.unique(df_user['province']))
# df_user_age['province'].hist()
# plt.title('province')
# plt.show()

# print(df_user.count())
# print(df_user['age'].value_counts())
# print(df_user['sex'].value_counts())
# print(df_user['user_reg_tm'].value_counts())
# print(df_user['user_lv_cd'].value_counts())
# print(df_user['city_level'].value_counts())
# print(df_user['province'].value_counts())
# print(df_user['city'].value_counts())
# print(df_user['county'].value_counts())


#######################################################################################################


# import pandas as pd
# df_month = pd.read_csv('data\jdata_action2.csv')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('data\jdata_action2.csv',index=None)
#
# df_month = pd.read_csv('data\jdata_action3.csv')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('data\jdata_action3.csv',index=None)
#
# df_month = pd.read_csv('data\jdata_action4.csv')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('data\jdata_action4.csv',index=None)


# # 提取购买(type=2)的行为数据
# def get_from_action_data(fname, chunk_size=100000):
#     reader = pd.read_csv(fname, header=0, iterator=True)
#     chunks = []
#     loop = True
#     while loop:
#         try:
#             chunk = reader.get_chunk(chunk_size)[
#                 ["user_id", "sku_id", "action_time", "module_id", "type"]]
#             chunks.append(chunk)
#         except StopIteration:
#             loop = False
#             print("Iteration is stopped")
#     df_ac = pd.concat(chunks, ignore_index=True)
#     # type=2为购买
#     df_ac = df_ac[df_ac['type'] == 2]
#     return df_ac[["user_id", "sku_id", "action_time", "module_id", "type"]]
# #
#
ACTION_2_FILE = 'data\jdata_action2.csv'
ACTION_3_FILE = 'data\jdata_action3.csv'
ACTION_4_FILE = 'data\jdata_action4.csv'
df_ac = []
# df_ac.append(get_from_action_data(fname=ACTION_2_FILE))
# df_ac.append(get_from_action_data(fname=ACTION_3_FILE))
# df_ac.append(get_from_action_data(fname=ACTION_4_FILE))
# df_ac = pd.concat(df_ac, ignore_index=True)


# # 将time字段转换为datetime类型
# df_ac['action_time'] = pd.to_datetime(df_ac['action_time'])
# # 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
# df_ac['action_time'] = df_ac['action_time'].apply(lambda x: x.weekday() + 1)
# # 周一到周日每天购买用户个数
# df_user = df_ac.groupby('action_time')['user_id'].nunique()
# df_user = df_user.to_frame().reset_index()
# df_user.columns = ['weekday', 'user_num']
#
# # 周一到周日每天购买商品个数
# df_item = df_ac.groupby('action_time')['sku_id'].nunique()
# df_item = df_item.to_frame().reset_index()
# df_item.columns = ['weekday', 'item_num']
#
# # 周一到周日每天购买记录个数
# df_ui = df_ac.groupby('action_time', as_index=False).size()
# df_ui = df_ui.to_frame().reset_index()
# df_ui.columns = ['weekday', 'user_item_num']
#
# # 条形宽度
# bar_width = 0.2
# # 透明度
# opacity = 0.4
# plt.bar(df_user['weekday'], df_user['user_num'], bar_width,
#         alpha=opacity, color='c', label='user')
# plt.bar(df_item['weekday']+bar_width, df_item['item_num'],
#         bar_width, alpha=opacity, color='g', label='item')
# plt.bar(df_ui['weekday']+bar_width*2, df_ui['user_item_num'],
#         bar_width, alpha=opacity, color='m', label='user_item')
# plt.xlabel('weekday')
# plt.ylabel('number')
# plt.title('A Week Purchase Table')
# plt.xticks(df_user['weekday'] + bar_width * 3 / 2., (1,2,3,4,5,6,7))
# plt.tight_layout()
# plt.legend(prop={'size':10})
# plt.show()

#######################################################################################################

###2018年2月

# df_ac = get_from_action_data(fname=ACTION_2_FILE)
# # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
# df_ac['action_time'] = pd.to_datetime(df_ac['action_time']).apply(lambda x: x.day)
#
#
# df_user = df_ac.groupby('action_time')['user_id'].nunique()
# df_user = df_user.to_frame().reset_index()
# df_user.columns = ['day', 'user_num']
#
# df_item = df_ac.groupby('action_time')['sku_id'].nunique()
# df_item = df_item.to_frame().reset_index()
# df_item.columns = ['day', 'item_num']
#
# df_ui = df_ac.groupby('action_time', as_index=False).size()
# df_ui = df_ui.to_frame().reset_index()
# df_ui.columns = ['day', 'user_item_num']
#
# # 条形宽度
# bar_width = 0.2
# # 透明度
# opacity = 0.4
# # 天数
# day_range = range(1,len(df_user['day']) + 1, 1)
# # 设置图片大小
# plt.figure(figsize=(14,10))
# plt.bar(df_user['day'], df_user['user_num'], bar_width,
#         alpha=opacity, color='c', label='user')
# plt.bar(df_item['day']+bar_width, df_item['item_num'],
#         bar_width, alpha=opacity, color='g', label='item')
# plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'],
#         bar_width, alpha=opacity, color='m', label='user_item')
# plt.xlabel('day')
# plt.ylabel('number')
# plt.title('February Purchase Table')
# plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# # plt.ylim(0, 80)
# plt.tight_layout()
# plt.legend(prop={'size':9})
# plt.show()


#######################################################################
# #2018年3月
# df_ac = get_from_action_data(fname=ACTION_3_FILE)
# # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
# df_ac['action_time'] = pd.to_datetime(df_ac['action_time']).apply(lambda x: x.day)
#
#
# df_user = df_ac.groupby('action_time')['user_id'].nunique()
# df_user = df_user.to_frame().reset_index()
# df_user.columns = ['day', 'user_num']
#
# df_item = df_ac.groupby('action_time')['sku_id'].nunique()
# df_item = df_item.to_frame().reset_index()
# df_item.columns = ['day', 'item_num']
#
# df_ui = df_ac.groupby('action_time', as_index=False).size()
# df_ui = df_ui.to_frame().reset_index()
# df_ui.columns = ['day', 'user_item_num']
#
# # 条形宽度
# bar_width = 0.2
# # 透明度
# opacity = 0.4
# # 天数
# day_range = range(1,len(df_user['day']) + 1, 1)
# # 设置图片大小
# plt.figure(figsize=(14,10))
# plt.bar(df_user['day'], df_user['user_num'], bar_width,
#         alpha=opacity, color='c', label='user')
# plt.bar(df_item['day']+bar_width, df_item['item_num'],
#         bar_width, alpha=opacity, color='g', label='item')
# plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'],
#         bar_width, alpha=opacity, color='m', label='user_item')
# plt.xlabel('day')
# plt.ylabel('number')
# plt.title('March Purchase Table')
# plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# # plt.ylim(0, 80)
# plt.tight_layout()
# plt.legend(prop={'size':9})
# plt.show()


##################################################################################
# 2018年4月
# df_ac = get_from_action_data(fname=ACTION_4_FILE)
# # 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
# df_ac['action_time'] = pd.to_datetime(df_ac['action_time']).apply(lambda x: x.day)
#
#
# df_user = df_ac.groupby('action_time')['user_id'].nunique()
# df_user = df_user.to_frame().reset_index()
# df_user.columns = ['day', 'user_num']
#
# df_item = df_ac.groupby('action_time')['sku_id'].nunique()
# df_item = df_item.to_frame().reset_index()
# df_item.columns = ['day', 'item_num']
#
# df_ui = df_ac.groupby('action_time', as_index=False).size()
# df_ui = df_ui.to_frame().reset_index()
# df_ui.columns = ['day', 'user_item_num']
#
# # 条形宽度
# bar_width = 0.2
# # 透明度
# opacity = 0.4
# # 天数
# day_range = range(1,len(df_user['day']) + 1, 1)
# # 设置图片大小
# plt.figure(figsize=(14,10))
# plt.bar(df_user['day'], df_user['user_num'], bar_width,
#         alpha=opacity, color='c', label='user')
# plt.bar(df_item['day']+bar_width, df_item['item_num'],
#         bar_width, alpha=opacity, color='g', label='item')
# plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'],
#         bar_width, alpha=opacity, color='m', label='user_item')
# plt.xlabel('day')
# plt.ylabel('number')
# plt.title('April Purchase Table')
# plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# # plt.ylim(0, 80)
# plt.tight_layout()
# plt.legend(prop={'size':9})
# plt.show()


########################################################################

# 从行为记录中提取商品类别数据
# def get_from_action_data(fname, chunk_size=100000):
#     reader = pd.read_csv(fname, header=0, iterator=True)
#     chunks = []
#     loop = True
#     while loop:
#         try:
#             chunk = reader.get_chunk(chunk_size)[
#                 ["cate", "brand", "type", "time"]]
#             chunks.append(chunk)
#         except StopIteration:
#             loop = False
#             print("Iteration is stopped")
#     df_ac = pd.concat(chunks, ignore_index=True)
#     # type=2,为购买
#     df_ac = df_ac[df_ac['type'] == 4]
#     return df_ac[["cate", "brand", "type", "time"]]
