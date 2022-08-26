import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

enc = ['win', 'draw', 'lose', 'win']
dec = ['draw', 'draw', 'win']

# # 输入: 一维数组, 1D array
LE = LabelEncoder()
print("LE fit: ", LE.fit(enc))
print("LE.classes_: ", LE.classes_)
print("LE transform: ", LE.transform(dec))

# # 输入: 二维数组, DataFrame
OE = OrdinalEncoder()
encPd = pd.DataFrame(enc)       # 将一维数组转换为二维DataFrame
decPd = pd.DataFrame(dec)
print('enc:\n', enc, 'encPd:\n', encPd)
print('dec:\n', dec, 'decPd:\n', decPd)
print("OE fit: ", OE.fit(encPd))
print("OE.categories_:\n", OE.categories_)
print("OE transform:\n", OE.transform(decPd))

# # 输入: LE编码的一维数组 or DataFrame
OHE = OneHotEncoder()
num = LE.fit_transform(enc)     # shape(a,b)
print("num:\n", num)        # 1 * 4
print("num.reshape(-1, 1):\n", num.reshape(-1, 1))      # 4 * 1 (4 = 1*4 / 1)
yOHE = OHE.fit_transform(num.reshape(-1, 1))        # -1 代表 行数d自动计算， d = a*b/m
yArray = yOHE.toarray()
print("yArray\n", yArray)
