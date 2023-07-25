import pandas as pd
jdata_action = pd.read_csv('data/jdata_0226_0325.csv')
jdata_product = pd.read_csv('data/jdata_product.csv')
jdata_user = pd.read_csv('data/jdata_user.csv')
jdata_shop = pd.read_csv('data/jdata_shop.csv')

# 1 筛选jdata action信息
jdata_action = jdata_action[jdata_action['type'] != 5]

# 合并数据集
train_action = pd.merge(jdata_action, jdata_product, on=['sku_id'], how='left', copy=False)
train_action['cate'] = jdata_action['cate'].fillna(-1)
print(len(jdata_action))
train_action = jdata_action[jdata_action['cate'] != -1]
print(len(jdata_action))

label = pd.read_csv("data/jdata_test_0326_0401.csv", encoding='utf-8')
label['label'] = 1
train_action = pd.merge(train_action, label, on=['user_id', 'cate', 'shop_id'], how='left', copy=False)
train_action['label'] = train_action['label'].fillna(0)

train_action = pd.merge(train_action, jdata_shop, on=['shop_id'], how='left', copy=False)
train_action.rename(columns={'cate_x': 'cate', 'cate_y': 'shop_main_cate'}, inplace=True)
train_action = train_action[train_action['fans_num'] != 0]

train_action = pd.merge(train_action, jdata_user, on=['user_id'], how='left', copy=False)
data_time = ['action_time', 'market_time', 'shop_reg_tm', 'user_reg_tm', 'shop_score']
train_int_feature = [x for x in train_action.columns if x not in data_time]
for i in train_int_feature:
    train_action[i] = train_action[i].fillna(-1).astype(int)

train_action.to_csv('cache/0226_0325train_action.csv', index=False)



