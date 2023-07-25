import pandas as pd
# submit = pd.read_csv('data/submit.csv')
# submit['pred'] = submit['fold1']+submit['fold2']+submit['fold3']+submit['fold4']+submit['fold5']
# submit['pred'] = submit['pred']/5
# submit = submit[['user_id', 'cate', 'shop_id', 'pred']]
# submit.to_csv('data/submitpred.csv', encoding='utf-8', index=None)
# submit = pd.read_csv('data/submitpred.csv')
# submit = submit[submit['pred'] > 0.6]
# submit = submit[['user_id', 'cate', 'shop_id']]
# print(len(submit))
# print(submit)
# submit.to_csv('data/baseline.csv', encoding='utf-8', index=None)

result2 = pd.read_csv('data/result2.csv')
print(len(result2))
