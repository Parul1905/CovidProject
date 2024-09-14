import pandas as pd
a = [1,7,2]
index_names = ["x","y","z"]
myvar = pd.Series(a, index_names)
print(myvar)