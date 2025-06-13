import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data={
    'TID':[1,2,3,4,5],
    'MILK':[1,0,1,1,0],
    'BREAD':[1,1,0,1,1],
    'BUTTER':[1,0,0,1,1]
}
df=pd.DataFrame(data)
df.set_index('TID',inplace=True)
print("Transcation Data")
print(df)
frequentitemsets=apriori(df,min_support=0.4,use_colnames=True)
print("frequent itemsets \n")
print(frequentitemsets)
rules=association_rules(frequentitemsets,metric="confidence",min_threshold=0.6)
print("\n association Rules")
print(rules)