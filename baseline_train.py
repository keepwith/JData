import pandas as pd
import datetime
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('cache/0226_0325train_action.csv',
                    parse_dates=['action_time', 'market_time', 'shop_reg_tm', 'user_reg_tm'])


def get_count_fearue(data, type_list=[1, 2, 3, 4]):
    end_data = data['action_time'].max()
    for i in [1, 3, 7, 14]:
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


train_count_feaure = get_count_fearue(train, [1, 2, 3, 4])

for i in [[1], [2], [3], [4]]:
    print('type', i)
    train_tmp = get_count_fearue(train, i)
    train_count_feaure = pd.merge(train_count_feaure, train_tmp, on=['user_id', 'cate', 'shop_id'], how='left')
    del train_tmp
# 构造标签信息
train_df = pd.merge(train_count_feaure, train[['user_id', 'cate', 'shop_id', 'label']].drop_duplicates(),
                    on=['user_id', 'cate', 'shop_id'], how='left', copy=False)
del train_count_feaure
user_feature = ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level', 'province', 'city', 'county']
shop_feature = ['vender_id', 'shop_id', 'fans_num', 'vip_num', 'shop_main_cate', 'shop_score']
# 用户基础特征
train_user_f = train[user_feature].drop_duplicates()
train_df = pd.merge(train_df, train_user_f, on=['user_id'], how='left', copy=False)
del train_user_f
# 店铺基础特征
train_shop_f = train[shop_feature].drop_duplicates()
train_df = pd.merge(train_df, train_shop_f, on=['shop_id'], how='left', copy=False)
del train_shop_f
# 查看样本比例
del train
print(train_df.groupby('label').size())

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 2 ** 5 - 1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 42,
    # 'nthread': 4,
    'verbose': 0,
    'use_two_round_loading': True
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = train_df[['user_id', 'cate', 'shop_id', 'label']]
oof['predict'] = 0
features = [col for col in train_df.columns if col not in ['label', 'user_id']]
print(features)
feature_importance_df = pd.DataFrame()
for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print('clf_{}'.format(fold))
    X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['label']
    X_valid, y_valid = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['label']
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(params,
                        trn_data,
                        100000,
                        valid_sets=[trn_data, val_data],
                        early_stopping_rounds=100,
                        verbose_eval=20,
                        evals_result=evals_result)
    p_valid = lgb_clf.predict(X_valid)
    oof['predict'][val_idx] = p_valid
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
oof.to_csv('cache/oof.csv', index=False)
feature_importance_df.to_csv('cache/feature.csv', index=False)

lgb_clf.save_model('cache/model.txt')



