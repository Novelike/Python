# import numpy as np
#
# print(np.__version__)
#
# arr1 = np.array([1, 2, 3, 4, 5])
# print('arr1:', arr1)
# print('arr1.sum():', arr1.sum())
# print('arr1.mean():', arr1.mean())
# print('arr1.max():', arr1.max())
# print('arr1.min():', arr1.min())
# print('arr1.std():', arr1.std())
# print('arr1.var():', arr1.var())
# print(arr1[arr1%2==0])
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])
# print('행렬 곱:', a.dot(b))
# print(a*b)
#
# arr2 = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr2[0:, 1:3])
# print('arr2:', arr2)
# print('arr2.ndim:', arr2.ndim)
# print('arr2.shape:', arr2.shape)
# print('arr2.size:', arr2.size)
# print('arr2.dtype:', arr2.dtype)
# print('arr2.itemsize:', arr2.itemsize)
# print('arr2.nbytes:', arr2.nbytes)
# print('arr2.T:', arr2.T)
# # 배열의 형태변경
# arr1d = np.arange(12) # [0~11]
# print('arr1d:', arr1d)
# arr2d = arr1d.reshape(3, 4)
# print('arr2d:', arr2d)
# print('arr2d.flatten():', arr2d.flatten())
#
# zeros = np.zeros((3, 4))
# print(zeros)
#
# range_arr = np.arange(0, 10, 2)
# print(range_arr)
# linear_space = np.linspace(0, 1, 5)
# print(linear_space)
#
# x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# y = np.array([[10, 11, 12 ,13], [20, 21, 22 ,23]])
# print(np.concatenate((x, y), axis=0))
# print(np.vstack((x, y)))
# print(np.hstack((x, y)))

# pandas
import pandas as pd
import numpy as np
print(pd.__version__)

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)
print(s.values)
print(s.index)
print(s.apply(np.sqrt))
print(s['a'])
print(s[['a', 'c']])

country = {
	'Korea': 51,
	'China': 147,
	'USA': 327,
	'Japan': 225
}
country_series = pd.Series(country)
print(country_series)

data = {
	'Name': ['Kim', 'Lee', 'Park', 'Choi', 'Son'],
	'Age': [25, 30, 27, 40, 22],
	'Country': ['Korea', 'China', 'USA', 'Japan', 'South Korea'],
	'Salary' : [1000000, 2000000, 3000000, 4000000, 5000000],
	'Department': ['IT', 'Marketing', 'Finance', 'Sales', 'IT']
}

df = pd.DataFrame(data, columns=['Name', 'Age', 'Country', 'Salary', 'Department'])
print('크기(행, 열):', df.shape)
print('열 이름:', df.columns)
print('행 인덱스:', df.index)
print('데이터 타입:', df.dtypes)
print(df[df['Age']>=30])
print(df['Age'] + 1)
print(df.sort_values(by='Salary', ascending=False))
print(df.iloc[0:3])
print(df.loc[[1, 2, 3]])
print(df.loc[0:1, ['Name', 'Age']])