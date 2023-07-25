import pandas as pd
import datetime
import warnings

jdata_action = pd.read_csv('data/jdata_0319_0415.csv')
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

train_action = pd.merge(train_action, jdata_shop, on=['shop_id'], how='left', copy=False)
train_action.rename(columns={'cate_x': 'cate', 'cate_y': 'shop_main_cate'}, inplace=True)
train_action = train_action[train_action['fans_num'] != 0]

train_action = pd.merge(train_action, jdata_user, on=['user_id'], how='left', copy=False)
data_time = ['action_time', 'market_time', 'shop_reg_tm', 'user_reg_tm', 'shop_score']
train_int_feature = [x for x in train_action.columns if x not in data_time]
for i in train_int_feature:
    train_action[i] = train_action[i].fillna(-1).astype(int)

train_action.to_csv('cache/0319_0415test_action.csv', index=False)

warnings.filterwarnings('ignore')
test = pd.read_csv('cache/0319_0415test_action.csv',
                   parse_dates=['action_time', 'market_time', 'shop_reg_tm', 'user_reg_tm'])


def get_count_fearue(data, type_list=[1, 2, 3, 4]):
    end_data = data['action_time'].max()
    for i in [1, 3, 5, 7]:
        begin_data = end_data - datetime.timedelta(days=i)
        tmp = data[(data['action_time'] > begin_data) & (data['action_time'] <= end_data)]
        tmp = tmp[tmp['type'].isin(type_list)]
        print(begin_data, end_data)
        # 被交互的次数特征族
        # 统计发生了多少次的特征 count
        using_feature = ['user_id', 'cate', 'shop_id']
        tmp = tmp[['user_id', 'cate', 'shop_id', 'sku_id', 'module_id', 'brand']]
        for count_feature in [['user_id'], ['cate'], ['shop_id'],
                              ['user_id', 'cate'], ['user_id', 'shop_id'], ['cate', 'shop_id'],
                              ['user_id', 'cate', 'shop_id']]:
            using_feature.append(
                "pre_{}_day_{}_{}_count".format(i, '_'.join(list(count_feature)), '_'.join(list(map(str, type_list)))))
            tmp["pre_{}_day_{}_{}_count".format(i, '_'.join(list(count_feature)),
                                                '_'.join(list(map(str, type_list))))] = tmp.groupby(count_feature)[
                'sku_id'].transform('count')
        count_tmp = tmp[using_feature].drop_duplicates()

        # 交叉统计
        for k in ['user_id', 'shop_id', 'cate']:
            for f in ['sku_id', 'module_id', 'brand']:
                f1 = tmp.groupby([k], as_index=False)[f].agg(
                    {'{}_{}_unique_1{}__{}'.format(k, f, i, '_'.join(list(map(str, type_list)))): 'count'})
                # f2 = tmp.groupby([f],as_index=False)[k].agg({'{}_{}_unique_2{}'.format(f,k,i):'count'})
                count_tmp = pd.merge(count_tmp, f1, on=[k], how='left', copy=False)
                # count_tmp = pd.merge(count_tmp,f2,on=[f],how='left',copy=False)
                del f1
        del tmp
        if i == 1:
            feature = count_tmp
        else:
            feature = pd.merge(feature, count_tmp, on=['user_id', 'cate', 'shop_id'], how='outer', copy=False)
        del count_tmp
    return feature.fillna(-1)


test_count_feaure = get_count_fearue(test, [1, 2, 3, 4])
for i in [[1], [2], [3], [4]]:
    print('type', i)
    test_tmp = get_count_fearue(test, i)
    test_count_feaure = pd.merge(test_count_feaure, test_tmp, on=['user_id', 'cate', 'shop_id'], how='left')
    del test_tmp

test_df = test_count_feaure.copy()
del test_count_feaure
user_feature = ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level', 'province', 'city', 'county']
shop_feature = ['vender_id', 'shop_id', 'fans_num', 'vip_num', 'shop_main_cate', 'shop_score']
# 用户基础特征
test_user_f = test[user_feature].drop_duplicates()
test_df = pd.merge(test_df, test_user_f, on=['user_id'], how='left', copy=False)
del test_user_f
# 店铺基础特征
test_shop_f = test[shop_feature].drop_duplicates()
test_df = pd.merge(test_df, test_shop_f, on=['shop_id'], how='left', copy=False)
del test_shop_f
del test

import lightgbm as lgb
predictions = test_df[['user_id', 'cate', 'shop_id']]
features = [col for col in test_df.columns if col not in ['label', 'user_id']]
X_test = test_df[features].values
bst = lgb.Booster(model_file='cache/model.txt')
ypred = bst.predict(X_test, num_iteration=bst.best_iteration)
predictions['pred'] = ypred
predictions.to_csv('result/submit.csv', index=False)

