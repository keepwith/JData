import pandas as pd

predict = pd.read_csv("result/0_5.csv", encoding='utf-8')
true = pd.read_csv("result/jdata_test_0409_0415.csv", encoding='utf-8')
predict_cate = predict[['user_id', 'cate']]
predict_cate = predict_cate.drop_duplicates()
true_cate = true[['user_id', 'cate']]
true_cate = true_cate.drop_duplicates()
pos = len(pd.merge(true_cate, predict_cate))
neg = len(true_cate)-pos
precision = 1.0 * pos / (pos + neg)
recall = 1.0 * pos / len(predict_cate)
f11 = 3 * recall * precision / (2 * recall + precision)
print("F11:", f11)

predict_shop = predict[['user_id', 'shop_id']]
predict_shop = predict_shop.drop_duplicates()
true_shop = true[['user_id', 'shop_id']]
true_shop = true_shop.drop_duplicates()
pos = len(pd.merge(true_shop, predict_shop))
neg = len(true_shop)-pos
precision = 1.0 * pos / (pos + neg)
recall = 1.0 * pos / len(predict_shop)
f12 = 5 * recall * precision / (2 * recall + 3 * precision)
print("F12:", f12)

score = 0.4 * f11 + 0.6 * f12
print("Final score:", score)
