## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by : Arunmozhi Varman T
Reg No : 212223230022
```

```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![318691387-9a445ed3-f79e-46ed-8493-a0138abde135](https://github.com/user-attachments/assets/fd371c3e-e153-48cb-b6f0-dac9d50e1933)


```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![318692227-c5ae2314-6f2b-4d93-92b3-f44d1b74015a](https://github.com/user-attachments/assets/a95d2be5-ff16-49a6-b28d-6f4014a180d2)




```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![318692322-4ae17d2a-aa22-4340-9faf-8567549250f6](https://github.com/user-attachments/assets/e03f7814-f615-48d3-b77c-b180ad0002c2)




```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![318692437-2249ccf3-4a16-462b-b745-677312c7fd42](https://github.com/user-attachments/assets/fee98a97-89bf-4a9c-80c9-8c95d29a0236)




```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![318692763-d2714505-ceae-48c6-b428-fc421aaa735d](https://github.com/user-attachments/assets/a3e4b7ac-3d2e-4178-9dec-a6a97709e293)



```python
df2=pd.concat([df2,enc],axis=1)
df2
```

![318692827-b4b4c5b2-9bc8-4f41-8649-096999696847](https://github.com/user-attachments/assets/94e9bca9-642c-4552-91d9-340c23f128c7)


```python
pd.get_dummies(df2,columns=["nom_0"])
```

![318692921-e56e11b0-9489-41a5-973c-e32fca8f9840](https://github.com/user-attachments/assets/5bcb9151-fdd1-471f-a3b5-aaf1c2123834)




```python
pip install --upgrade category_encoders
```

![318693032-0711d42f-4456-4222-8334-f183bc7c2385](https://github.com/user-attachments/assets/12b50681-e958-4531-915b-f5d80d997066)




```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![318693230-3d2f8b4c-0ffc-4754-8c1b-ad637c727c9b](https://github.com/user-attachments/assets/5e5f7008-f765-4016-9a56-f1d0ae279086)



```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![318897767-781ddd71-1fc6-499b-9234-b83778405580](https://github.com/user-attachments/assets/9cf0a244-09f8-4a04-a519-0e2f7e8dfedb)



```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![318897871-6f1877a4-9ba9-45d6-8df2-38fdc103a0ef](https://github.com/user-attachments/assets/a913aa68-7346-4ae4-8ab3-6f456aa090dc)




```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![318897982-63cbb12a-e9eb-447e-855a-e56c706bbfa9](https://github.com/user-attachments/assets/157b430c-4d25-47b3-86b2-4b56d89fccf7)




```python
df.skew()
```

![318898092-3d04bbce-76dc-4571-8c8d-5aad234c1766](https://github.com/user-attachments/assets/4b25f782-3318-444a-9d0f-ac6f67786e48)

```python
np.log(df["Highly Positive Skew"])
```

![318898189-7247340c-6488-4b75-9deb-0ad3f10e03fd](https://github.com/user-attachments/assets/503706e0-27ec-4bdb-95c3-ee3563e9f9fb)




```python
np.reciprocal(df["Moderate Positive Skew"])
```

![318898261-71ae0399-a828-406a-93a6-0e36cc31e249](https://github.com/user-attachments/assets/8f9eb90c-1730-43d0-a7e8-323bd5b68739)



```python
np.sqrt(df["Highly Positive Skew"])
```

![318898327-9b500fd0-9b55-4397-b1e8-364652aca983](https://github.com/user-attachments/assets/0256c395-a9e4-4c38-ba99-d0ad9a2ae5fa)



```python
np.square(df["Highly Positive Skew"])
```


![318898423-d243323b-c97e-4c55-a41f-f76d176e6461](https://github.com/user-attachments/assets/85691a2f-09d8-494c-85ce-265689fe6683)



```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![318898509-758eaaba-b780-4fee-8487-d8242a9d6148](https://github.com/user-attachments/assets/ac1c56d6-a9d4-42f6-857c-80a79e03b7b8)



```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```

![318898927-4945b8c6-e27d-4526-9032-0c0aeb9ab576](https://github.com/user-attachments/assets/b66e714a-fcfa-4e45-9901-4633ee0f6df4)



```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![318899248-52a7553c-c1bd-4489-a0cb-b13a27684c23](https://github.com/user-attachments/assets/b2e8e190-4d6a-4d85-941a-bec2fa04d4ae)




```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

![318899545-3688ed78-4920-4cd4-9e33-4420fc790b8d](https://github.com/user-attachments/assets/e125d45d-9e70-458b-a65a-87d43b736d8c)




```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


![318899696-9ef5152c-d766-48e1-857c-a7dbfde4e648](https://github.com/user-attachments/assets/b2eceeb5-822d-4155-adeb-dd7ca54987ca)



```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![318899799-fde4b296-88ec-46ad-b6f3-2cf2b64a15f2](https://github.com/user-attachments/assets/7d02db56-390a-48c3-8631-8b8e0d5266a7)



```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![318899874-57bae70b-8ee0-4ab1-86bf-733d2597089d](https://github.com/user-attachments/assets/8338b60c-6b4e-46b4-8dcd-fc3b14a729c4)



```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![318900112-3987a28b-3816-41b2-9a9d-6a1cedf8382e](https://github.com/user-attachments/assets/25eb2d07-e226-44ef-9b14-bf023937678e)





## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.


       
