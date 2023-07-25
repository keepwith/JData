import pandas as pd
import numpy as np
import gc


def make_shop():
    SHOP_FILE = "data/jdata_shop.csv"
    COMMENT_FILE = "data/jdata_comment.csv"
    PRODUCT_FILE = "data/jdata_product.csv"
    df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
    df_shop = df_shop.dropna(subset=['shop_reg_tm'])
    df_shop = df_shop[df_shop['shop_score'] > 8]
    del df_shop['cate']
    print(len(df_shop))
    df_comment = pd.read_csv(COMMENT_FILE, encoding='utf-8')
    df_product = pd.read_csv(PRODUCT_FILE, encoding='utf-8')
    df_comment = df_comment.groupby('sku_id', as_index=False).sum()
    df_comment = pd.merge(df_comment, df_product)
    df_comment = df_comment.groupby('shop_id', as_index=False).sum()
    del df_comment['sku_id']
    del df_comment['brand']
    del df_comment['cate']
    df_shop = pd.merge(df_shop, df_comment, how='left', on='shop_id')
    df_shop.fillna(0, inplace=True)
    pd.options.display.max_columns = None
    df_shop.to_csv('data/jdata_shop.csv', encoding='utf-8', index=None)


# def make_cate_shop_list():
# SHOP_FILE = "data/jdata_shop.csv"
# ACTION_FILE = "data/jdata_action_new.csv"
# df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action = df_action[df_action['type'] == 2]
#
# df_action1 = df_action[['cate', 'shop_id', 'type']]
# df_action1 = df_action1.groupby('cate', as_index=False)['type'].count()
# df_action1 = df_action1.drop_duplicates()
# df_action1.rename(columns={'type': 'total'}, inplace=True)
#
# df_action = df_action[['cate', 'shop_id', 'type']]
# df_action = df_action.groupby(['cate', 'shop_id'], as_index=False)['type'].count()
# df_action = df_action.drop_duplicates()
# df_action = pd.merge(df_action, df_action1, on='cate')
# df_action['type'] = df_action['type']/df_action['total']
# df_action = df_action.sort_values(by=['cate', 'type'], ascending=True)
# df_action = df_action[df_action['type'] >= 0.005]
# df_action = df_action[['cate', 'shop_id']]
# df_action.drop_duplicates(inplace=True)
# print(df_action)
# df_action.to_csv('data/jdata_cate_shop.csv', encoding='utf-8', index=None)


# SHOP_FILE = "data/jdata_shop.csv"
# df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
# del df_shop['vender_id']
# df_shop.to_csv('data/jdata_shop.csv', encoding='utf-8', index=None)

# df_user_cate = pd.read_csv("data/jdata_user_cate_shop.csv", encoding='utf-8')
# df_user_cate = df_user_cate[['user_id', 'cate']]
# df_user_cate = df_user_cate.drop_duplicates()
# print(df_user_cate)
# df_user_cate.to_csv('data/jdata_user_cate.csv', encoding='utf-8', index=None)


# def make_cate_shop():
USER_CATE = "data/result2.csv"
SHOP_FILE = "data/jdata_shop.csv"
ACTION_FILE = "data/jdata_action_new.csv"
CATE_SHOP = "data/jdata_cate_shop.csv"

#  类别-商店交互
user_cate = pd.read_csv(USER_CATE, encoding='utf-8')
print(len(user_cate))
user = user_cate['user_id'].drop_duplicates()
cate_shop = pd.read_csv(CATE_SHOP, encoding='utf-8')
print(len(cate_shop))
cate_shop = pd.merge(user_cate, cate_shop, on='cate')
print(len(cate_shop))
print(cate_shop)
del user_cate
# df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
# df_action = df_action[['cate', 'shop_id', 'type']]
# df_action['count'] = 0
# df_action = df_action.groupby(['cate', 'shop_id', 'type'])['count'].count()
# df_action = df_action.unstack()
# df_action = df_action.reset_index()
# df_action.rename(columns={1: 'cate_shop_view', 2: 'cate_shop_buy', 3: 'cate_shop_gz', 4: 'cate_shop_comment',
#                           5: 'cate_shop_cart'}, inplace=True)
# cate_shop_action = pd.merge(cate_shop, df_action, on=['cate', 'shop_id'])
# cate_shop_action = cate_shop_action.drop_duplicates()
# del df_action
# print(cate_shop_action)
# cate_shop_action.to_csv('data/jdata_cate_shop_action.csv', encoding='utf-8', index=None)
# del cate_shop_action

#  用户-商店交互
df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
df_action = df_action[['user_id', 'shop_id', 'type']]
df_action = df_action.groupby(['user_id', 'shop_id'], as_index=False)['type'].count()
df_action.rename(columns={'type': 'user_shop_actions'}, inplace=True)
user_shop_action = pd.merge(user, df_action, how='left', on='user_id')
del df_action
print(user_shop_action)

df_action = pd.read_csv(ACTION_FILE, encoding='utf-8')
df_action = df_action[['user_id', 'shop_id', 'type']]
df_action = df_action[df_action['type'] == 5]
df_action = df_action.groupby(['user_id', 'shop_id'], as_index=False)['type'].count()
df_action.rename(columns={'type': 'user_shop_carts'}, inplace=True)
print(df_action)
user_shop_action = pd.merge(user_shop_action, df_action, how='left', on=['user_id', 'shop_id'])
del df_action
print(user_shop_action)
user_shop_action.to_csv('data/jdata_user_shop_action.csv', encoding='utf-8', index=None)
del user_shop_action
# df_cate_shop_action = pd.read_csv('data/jdata_cate_shop_action.csv', encoding='utf-8')
# cate_shop = pd.merge(cate_shop, df_cate_shop_action, how='left', on=['user_id', 'cate', 'shop_id'])
# del df_cate_shop_action

df_user_shop_action = pd.read_csv('data/jdata_user_shop_action.csv', encoding='utf-8')
cate_shop = pd.merge(cate_shop, df_user_shop_action, how='left', on=['user_id', 'shop_id'])
del df_user_shop_action

df_shop = pd.read_csv(SHOP_FILE, encoding='utf-8')
cate_shop = pd.merge(cate_shop, df_shop, how='left', on=['shop_id'])
del df_shop
print(cate_shop)

# user_cate_shop = pd.read_csv("data/jdata_user_cate_shop.csv", encoding='utf-8')
# print(user_cate_shop)
# cate_shop = pd.merge(cate_shop, user_cate_shop, how='left', on=['user_id', 'cate', 'shop_id'])
cate_shop.fillna(0, inplace=True)
print(cate_shop)
cate_shop.to_csv('data/jdata_user_cate_shop_predict.csv', encoding='utf-8', index=None)
