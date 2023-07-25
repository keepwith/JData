import pandas as pd

result = pd.read_csv('result/submit.csv', encoding='utf-8')
result1 = result[result['pred'] > 0.5]
print(len(result1))
del result1['pred']
result1.to_csv('result/0_5.csv', index=None, encoding='utf-8')
result1 = result[result['pred'] > 0.6]
print(len(result1))
del result1['pred']
result1.to_csv('result/0_6.csv', index=None, encoding='utf-8')
result1 = result[result['pred'] > 0.7]
print(len(result1))
del result1['pred']
result1.to_csv('result/0_7.csv', index=None, encoding='utf-8')
result1 = result[result['pred'] > 0.8]
print(len(result1))
del result1['pred']
result1.to_csv('result/0_8.csv', index=None, encoding='utf-8')
result1 = result[result['pred'] > 0.9]
print(len(result1))
del result1['pred']
result1.to_csv('result/0_9csv', index=None, encoding='utf-8')
