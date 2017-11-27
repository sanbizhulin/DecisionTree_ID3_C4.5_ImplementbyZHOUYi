import pandas as pd
data = pd.read_csv("hw2data.csv")
features = ['A','B','C','D','E','F']
med={}
data=data.dropna(axis=0)

for item in features:
    med[item]=data[item].median()
    print "med[",item,"] is",med[item]
Data = data

for item in features:
    def f(x):
        if x > med[item]:
            return 1
        else:
            return 0
    Data[item] = data[item].map(f)
Data.to_csv("tree_data_dellackvalue.csv")