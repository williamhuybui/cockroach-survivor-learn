import pandas as pd

df = pd.read_csv("/Users/huybui/Desktop/cockroach-survivor-learn/tmp/question_set.csv")
df[df['split'] == 'dev'].to_csv("dev_set.csv",index = False)