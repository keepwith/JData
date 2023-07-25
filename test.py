import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

nrows = None


def get_data(file_path, nrows=10000, sep=',', ):
    return pd.read_csv(file_path, sep=',', nrows=nrows)


jdata_action = get_data('data/jdata_action.csv', nrows=nrows)
jdata_comment = get_data('data/jdata_comment.csv', nrows=nrows)
jdata_product = get_data('data/jdata_product.csv', nrows=nrows)
jdata_shop = get_data('data/jdata_shop.csv', nrows=nrows)
jdata_user = get_data('data/jdata_user.csv', nrows=nrows)


def Recall(df_true_pred, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold = 0.5
            Recall = TP / (TP + FP)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob'] >= threshold]
    if flag == 'user_cate':
        temp_ = temp_.drop_duplicates(['user_id', 'cate'])
        recall = np.sum(temp_['label']) * 1.0 / np.sum(df_true_pred['label'])
    elif flag == 'user_cate_shop':
        recall = np.sum(temp_['label']) * 1.0 / np.sum(df_true_pred['label'])
    else:
        recall = -1
    return recall


def Precision(df_true_pred, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold
            Precision = TP / (TP + TN)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob'] >= threshold]
    if flag == 'user_cate':
        temp_ = temp_.drop_duplicates(['user_id', 'cate'])
        precision = np.sum(temp_['label']) * 1.0 / np.size(df_true_pred)
    elif flag == 'user_cate_shop':
        precision = np.sum(temp_['label']) * 1.0 / np.size(df_true_pred)
    else:
        precision = -1
    return precision


def get_metrice(df_true_pred, threshold):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
        Threshold = 0.5
    """

    R1_1 = Recall(df_true_pred, threshold, flag='user_cate')
    P1_1 = Precision(df_true_pred, threshold, flag='user_cate')
    F1_1 = 3 * R1_1 * P1_1 / (2 * R1_1 + P1_1)

    R1_2 = Recall(df_true_pred, threshold, flag='user_cate_shop')
    P1_2 = Precision(df_true_pred, threshold, flag='user_cate_shop')
    F1_2 = 5 * R1_2 * P1_2 / (2 * R1_2 + 3 * P1_2)

    score = 0.4 * F1_1 + 0.6 * F1_2
    return score


def get_train_set():
    jdata_data = jdata_action.merge(jdata_product, on=['sku_id'])
    train_buy = jdata_data[(jdata_data['action_time'] >= '2018-04-09') \
                           & (jdata_data['action_time'] <= '2018-04-15') \
                           & (jdata_data['type'] == 2)][['user_id', 'cate', 'shop_id']].drop_duplicates()
    train_buy['label'] = 1
    train_set = jdata_data[(jdata_data['action_time'] >= '2018-03-26') \
                           & (jdata_data['action_time'] <= '2018-04-08')][
        ['user_id', 'cate', 'shop_id']].drop_duplicates()
    train_set = train_set.merge(train_buy, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)

    last_day = '2018-04-08'
    for d in [1, 3, 5, 7, 14]:
        for gb_c in [['user_id'],
                     ['cate'],
                     ['shop_id'],
                     ['user_id', 'cate'],
                     ['user_id', 'shop_id'],
                     ['cate', 'shop_id'],
                     ['user_id', 'cate', 'shop_id']]:
            #
            day_ = str(datetime.datetime(*[int(i) for i in last_day.split('-')]) - datetime.timedelta(d)).split(' ')[0]
            action_temp = jdata_data[(jdata_data['action_time'] >= day_)
                                     & (jdata_data['action_time'] <= last_day)]

            features_dict = {
                'sku_id': [np.size, lambda x: len(set(x))],
                'module_id': lambda x: len(set(x)),
                'type': lambda x: len(set(x)),
                'brand': lambda x: len(set(x)),
                'shop_id': lambda x: len(set(x)),
                'cate': lambda x: len(set(x))
            }
            features_columns = [c + '_' + str(d) + '_' + '_'.join(gb_c)
                                for c in
                                ['sku_cnt', 'sku_nq', 'module_nq', 'type_nq', 'brand_nq', 'shop_nq', 'cate_nq']]
            f_temp = action_temp.groupby(gb_c).agg(features_dict).reset_index()
            f_temp.columns = gb_c + features_columns
            train_set = train_set.merge(f_temp, on=gb_c, how='left')

            for type_ in [1, 2, 3, 4, 5]:
                action_temp = jdata_data[(jdata_data['action_time'] >= day_)
                                         & (jdata_data['action_time'] <= last_day)
                                         & (jdata_data['type'] == type_)]
                features_dict = {
                    'sku_id': [np.size, lambda x: len(set(x))],
                    'module_id': lambda x: len(set(x)),
                    'type': lambda x: len(set(x)),
                    'brand': lambda x: len(set(x)),
                    'shop_id': lambda x: len(set(x)),
                    'cate': lambda x: len(set(x))
                }
                features_columns = [c + '_' + str(d) + '_' + '_'.join(gb_c) + '_type_' + str(type_)
                                    for c in
                                    ['sku_cnt', 'sku_nq', 'module_nq', 'type_nq', 'brand_nq', 'shop_nq', 'cate_nq']]
                f_temp = action_temp.groupby(gb_c).agg(features_dict).reset_index()
                if len(f_temp) == 0:
                    continue
                f_temp.columns = gb_c + features_columns
                train_set = train_set.merge(f_temp, on=gb_c, how='left')

    uid_info_col = ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level', 'province', 'city', 'county']
    train_set = train_set.merge(jdata_user[uid_info_col], on=['user_id'], how='left')

    shop_info_col = ['shop_id', 'fans_num', 'vip_num', 'shop_score']
    train_set = train_set.merge(jdata_shop[shop_info_col], on=['shop_id'], how='left')

    return train_set


def get_test_set():
    jdata_data = jdata_action.merge(jdata_product, on=['sku_id'])

    test_set = jdata_data[(jdata_data['action_time'] >= '2018-04-02')
                          & (jdata_data['action_time'] <= '2018-04-15')][
        ['user_id', 'cate', 'shop_id']].drop_duplicates()

    last_day = '2018-04-15'
    for d in [1, 3, 5, 7, 14]:
        for gb_c in [['user_id'],
                     ['cate'],
                     ['shop_id'],
                     ['user_id', 'cate'],
                     ['user_id', 'shop_id'],
                     ['cate', 'shop_id'],
                     ['user_id', 'cate', 'shop_id']]:
            day_ = str(datetime.datetime(*[int(i) for i in last_day.split('-')]) - datetime.timedelta(d)).split(' ')[0]
            action_temp = jdata_data[(jdata_data['action_time'] >= day_)
                                     & (jdata_data['action_time'] <= last_day)]

            features_dict = {
                'sku_id': [np.size, lambda x: len(set(x))],
                'module_id': lambda x: len(set(x)),
                'type': lambda x: len(set(x)),
                'brand': lambda x: len(set(x)),
                'shop_id': lambda x: len(set(x)),
                'cate': lambda x: len(set(x))
            }
            features_columns = [c + '_' + str(d) + '_' + '_'.join(gb_c)
                                for c in
                                ['sku_cnt', 'sku_nq', 'module_nq', 'type_nq', 'brand_nq', 'shop_nq', 'cate_nq']]
            f_temp = action_temp.groupby(gb_c).agg(features_dict).reset_index()
            f_temp.columns = gb_c + features_columns
            test_set = test_set.merge(f_temp, on=gb_c, how='left')

            for type_ in [1, 2, 3, 4, 5]:
                action_temp = jdata_data[(jdata_data['action_time'] >= day_)
                                         & (jdata_data['action_time'] <= last_day)
                                         & (jdata_data['type'] == type_)]
                features_dict = {
                    'sku_id': [np.size, lambda x: len(set(x))],
                    'module_id': lambda x: len(set(x)),
                    'type': lambda x: len(set(x)),
                    'brand': lambda x: len(set(x)),
                    'shop_id': lambda x: len(set(x)),
                    'cate': lambda x: len(set(x))
                }
                features_columns = [c + '_' + str(d) + '_' + '_'.join(gb_c) + '_type_' + str(type_)
                                    for c in
                                    ['sku_cnt', 'sku_nq', 'module_nq', 'type_nq', 'brand_nq', 'shop_nq', 'cate_nq']]
                f_temp = action_temp.groupby(gb_c).agg(features_dict).reset_index()
                if len(f_temp) == 0:
                    continue
                f_temp.columns = gb_c + features_columns
                test_set = test_set.merge(f_temp, on=gb_c, how='left')

    uid_info_col = ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level', 'province', 'city', 'county']
    test_set = test_set.merge(jdata_user[uid_info_col], on=['user_id'], how='left')

    shop_info_col = ['shop_id', 'fans_num', 'vip_num', 'shop_score']
    test_set = test_set.merge(jdata_shop[shop_info_col], on=['shop_id'])

    return test_set


train_set = get_train_set()
test_set = get_test_set()


class SBBTree():

    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
            for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(self.params,
                                lgb_train,
                                num_boost_round=self.num_boost_round,
                                valid_sets=lgb_eval,
                                early_stopping_rounds=self.early_stopping_rounds)

                self.stacking_model.append(gbm)

                pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                layer_train[test_index, 1] = pred_y

            X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        for bn in range(self.bagging_num):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=10000,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=200)

            self.bagging_model.append(gbm)

    def predict(self, X_pred):
        """ predict test data. """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                test_pred[:, sn] = pred
            X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass
        for bn, gbm in enumerate(self.bagging_model):
            pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred
        return pred_out / self.bagging_num


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 2 ** 5 - 1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': .7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 2018,
    'nthread': 4,
    'verbose': 0,
}

train_test_columns = list(set(train_set.columns) & set(test_set.columns))
X_train = train_set[train_test_columns].values
y_train = train_set['label'].values
X_test = test_set[train_test_columns].values
train_metrics = train_set[['user_id', 'cate', 'shop_id', 'label']]
submit_ID = test_set[['user_id', 'cate', 'shop_id']]

model = SBBTree(params=params,
                stacking_num=5,
                bagging_num=5,
                bagging_test_size=0.33,
                num_boost_round=10000,
                early_stopping_rounds=200)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

submit_ID['pred_prob'] = submit_ID
submit_ID[submit_ID['pred_prob'] >= 0.5][['user_id', 'cate', 'shop_id']].csv('submit_baseline.csv', header=True,
                                                                             sep=',', encoding='utf-8')
