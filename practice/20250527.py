import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# df = pd.DataFrame({
# 	'A': ['1', '2', '3', '4', '5'],
# 	'B': [1.1, 2.2, 3.3, 4.4, 5.5],
# 	'C': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01'],
# 	'D': ['True', 'False', 'True', 'False', 'True']
# })
#
# df['A'] = df['A'].astype(int)
# df['C'] = pd.to_datetime(df['C'])
# # df['D'] = df['D'].astype(bool)
# # df['D'] = df['D'].map({'True':True, 'False':False})
# df['D'] = (df['D']=='True')
# # print(df)
# print(df.dtypes)

# np.random.seed(42)
# normal_data = np.random.normal(50, 10 ,95) # 평균, 표준편차, 개수
# outliers = [120, 130, -20, -10, 150] # 이상치 5개
# data_with_outliers = np.concatenate([normal_data, outliers])
#
# df = pd.DataFrame({
# 	'ID': range(1, 101),
# 	'Score': data_with_outliers,
# 	'Category': np.random.choice(['A', 'B', 'C'], 100)
# })
#
# print("원본 데이터 통계:")
# print(df['Score'].describe())
#
# def detect_outliers_iqr(data):
# 	"""IQR 방법으로 이상치 탐지"""
# 	Q1 = data.quantile(0.25) # 1사분위수
# 	Q3 = data.quantile(0.75) # 3사분위수
# 	IQR = Q3 - Q1            # 사분위수 범위
#
# 	# 이상치 경계값 계산
# 	lower_bound = Q1 - 1.5 * IQR
# 	upper_bound = Q3 + 1.5 * IQR
#
# 	# 이상치 식별
# 	outlierss = (data < lower_bound) | (data > upper_bound)
# 	return outlierss, lower_bound, upper_bound
#
# outliers_mask, lower, upper = detect_outliers_iqr(df['Score'])
# print(f"\nIQR 방법 - 이상치 경계 : {lower: .2f} ~ {upper: .2f}")
# print(f"이상치 개수 : {outliers_mask.sum()}개")
# print(f"이상치 값들 : {df[outliers_mask]['Score'].values}")
#
# df_no_outliers = df[~outliers_mask].copy()
# print(f"이상치 제거 전 데이터 크기 : {len(df)}")
# print(f"이상치 제거 후 데이터 크기 : {len(df_no_outliers)}")
# print(df_no_outliers['Score'].describe())
#
# """대체값으로 처리 (제거 대신)"""
# df_replaced = df.copy()
# # 이상치를 중앙값으로 대체
# median_score = df_replaced['Score'].median()
# df_replaced.loc[outliers_mask, 'Score'] = median_score
# print(f"\n이상치를 중앙값({median_score: .2f})으로 대체 후 통계:")
# print(df_replaced['Score'].describe())
#
#
# def detect_outliers_zscore(data, threshold = 3):
# 	"""Z-Score 방법으로 이상치 탐지"""
# 	z_scores = np.abs((data - data.mean()) / data.std())
# 	outliers = z_scores > threshold
# 	return outliers, z_scores
#
# outliers_zscore, z_scores = detect_outliers_zscore(df['Score'])
# print("Z-Score 방법")
# print(f"이상치 개수 : {outliers_zscore.sum()}개")
# print(f"이상치 값들 : {df[outliers_zscore]['Score'].values}")

# """중복 데이터 제거"""
# data = {
# 	'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob', 'Eve', 'Charlie'],
# 	'Age': [25, 30, 35, 25, 40, 30, 28, 35],
# 	'City': ['Seoul', 'Busan', 'Seoul', 'Seoul', 'Daegu', 'Busan', 'Daegu', 'Seoul'],
# 	'Gender': ['F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
# 	'Salary': [1000000, 2000000, 3000000, 1000000, 4000000, 2000000, 5000000, 3000000]
# }
#
# df = pd.DataFrame(data)
# print("원본 데이터:")
# print(df)
# print(f"\n원본 데이터 크기 : {len(df)}행")
#
# # 완전 중복 탐지
# duplicated_rows = df[df.duplicated()]
# print(duplicated_rows.sum())
# # print(duplicated_rows)
#
# # 특정열 중복
# name_duplicated = df[df.duplicated(subset=['Name'])]
# # print(name_duplicated)
# name_age_duplicated = df[df.duplicated(subset=['Name', 'Age'])]
# print(f"\n이름+나이 기준 중복 행\n{name_age_duplicated}")
#
# # 완전 중복 제거
# df_no_duplicates = df.drop_duplicates()
# print(f"\n완전 중복 제거\n{df_no_duplicates}")
#
# # 특정열 중복 제거
# print("\n특정열 중복 제거")
# df_unique_names = df.drop_duplicates(subset=['Name'])
# print(df_unique_names)
# print("마지막행 유지")
# df_unique_names_last_row = df.drop_duplicates(subset=['Name'], keep='last')
# print(df_unique_names_last_row)

# # 데이터 정규화 및 표준화
# # 다양한 스케일의 데이터가 포함된 샘플 데이터 생성
# np.random.seed(42)
# data = {
# 	'Age': np.random.randint(20, 70, 100),
# 	'Salary': np.random.normal(50000, 15000, 100),
# 	'Experience': np.random.exponential(5, 100),
# 	'Score': np.random.uniform(0, 100, 100)
# }
#
# df = pd.DataFrame(data)
# df['Salary'] = df['Salary'].clip(lower=20000)
# df['Experience'] = df['Experience'].clip(upper=20)
# print("원본 데이터 통계:")
# print(df.describe())
#
# # 표준화
# scaler_standard = StandardScaler()
# df_standardized = pd.DataFrame(
# 	scaler_standard.fit_transform(df),
# 	columns=df.columns
# )
# print("\n표준화 데이터 통계:")
# print(df_standardized.describe())
#
# # 정규화
# scaler_minmax = MinMaxScaler()
# df_normalized = pd.DataFrame(
# 	scaler_minmax.fit_transform(df),
# 	columns=df.columns
# )
# print("\n정규화 데이터 통계:")
# print(df_normalized.describe())
#
# # 로버스트 스케일링 (x-median)/IQR
# scaler_robust = RobustScaler()
# df_robust = pd.DataFrame(
# 	scaler_robust.fit_transform(df),
# 	columns=df.columns
# )
# print("\n로버스트 스케일링 데이터 통계:")
# print(df_robust.describe())
#
#
# def minmax_normalize(series):
# 	return (series - series.min()) / (series.max() - series.min())

# # 온라인 쇼핑몰 고객 데이터 전처리
# # 기본 고객 데이터 생성
# n_customers = 1000
# customer_data = {
# 	'customer_id': range(1, n_customers + 1),
# 	'name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
# 	'age': np.random.normal(35, 12, n_customers).astype(int),
# 	'gender': np.random.choice(['M', 'F', 'Male', 'Female', 'm', 'f', ''], n_customers),
# 	'city': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', ''], n_customers),
# 	'total_purchase': np.random.exponential(50000, n_customers),
# 	'purchase_count': np.random.poisson(5, n_customers),
# 	'last_purchase_days': np.random.randint(1, 365, n_customers),
# 	'membership_level': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum', ''], n_customers)
# }
#
# # customer_id : 1~1000
# # name : Customer_1~Customer_1000
# # age : 0~100
# # gender : M, F, Male, Female, m, f, ''
# # city : Seoul, Busan, Daegu, Incheon, Gwangju, ''
# # total_purchase : 0~100000
# # purchase_count : 0~10
# # last_purchase_days : 1~365
# # membership_level : Bronze, Silver, Gold, Platinum, ''
#
# # 데이터프레임 생성
# df = pd.DataFrame(customer_data)
#
# print("=== 원본 데이터 탐색 ===")
# print(f"데이터 크기: {df.shape}")
# print("\n데이터 타입:")
# print(df.dtypes)
# print("\n처음 5행:")
# print(df.head())
#
# # 데이터에 어떤 문제가 있을까
# print("\n1. 결측치 현황")
# # missing_data = df.isnull().sum()
# # missing_percentage = (missing_data/len(df))*100
# # missing_summary = pd.DataFrame({
# # 	'결측치 개수': missing_data,
# # 	'결측치 비율': missing_percentage
# # })
# # print(missing_summary[missing_summary['결측치 개수']>0])
#
# # 개별 데이터들의 문제점을 확인
# print(f"나이 데이터 타입: {df['age'].dtype}")
# print(f"나이 데이터 범위: {df['age'].min()}~{df['age'].max()}")
# print(f"비정상 나이 값 범위: {df[df['age'] < 0 | (df['age'] > 100)]['age'].tolist()}")
#
# print("성별 데이터 일관성 문제")
# print(f"성별의 고유값: {df['gender'].unique()}")
# print(f"성별 값 개수: {df['gender'].value_counts()}")
#
# print("구매 이상치 확인")
# # IQR 적용
# Q1 = df['total_purchase'].quantile(0.25)
# Q3 = df['total_purchase'].quantile(0.75)
# IQR = Q3 - Q1
# outlier_threashold_low = Q1 - 1.5 * IQR
# outlier_threashold_high = Q1 + 1.5 * IQR
# outliers = df[(df['total_purchase'] < outlier_threashold_low) | (df['total_purchase'] > outlier_threashold_high)]
#
# print(f"이상치 개수: {len(outliers)}개 ({len(outliers)/len(df)*100:.2f}%)")
# print(f"이상치 범위: {outlier_threashold_low:.2f} 미만 또는 {outlier_threashold_high:.2f} 초과")
#
# # 중복 데이터 확인
# print("완전 중복 확인")
# duplicates = df.duplicated()
# print(f"완전 중복 행: {duplicates.sum()}")
# name_duplicates = df.duplicated(subset=['name'])
# print(f"이름 중복 행: {name_duplicates.sum()}")
#
# # 원본 데이터 백업
# df_original = df.copy()
#
# print("나이 데이터 정제")
# median_age = df[(df['age']>=0)&(df['age']<=100)]['age'].median()
# df.loc[(df['age']<0)|(df['age']>100), 'age'] = median_age
# print(f"나이 데이터 정제 후 범위: {df['age'].min()} ~ {df['age'].max()}")
# print(f"중앙값: {median_age}")
#
# print("성별 데이터 표준화")
# gender_mapping = {
# 	'M':'Male',
# 	'F':'Female',
# 	'm':'Male',
# 	'f':'Female',
# 	'Male':'Male',
# 	'Female':'Female',
# 	'':'Unknown'
# }
# df['gender'] = df['gender'].map(gender_mapping).fillna('Unknown')
# print(f"표준화 후 성별의 고유값: {df['gender'].unique()}")
# print(f"표준화 후 성별 값 개수: \n{df['gender'].value_counts()}")
#
# # 최빈값으로 대체
# print("도시 빈 값 대체")
# df['city'] = df['city'].replace('', np.nan)
# most_common_city = df['city'].mode()[0]
# df['city'] = df['city'].fillna(most_common_city)
# print(f"최빈값으로 대체 후 도시의 고유값: {df['city'].unique()}")
# print(f"최빈값으로 대체 후 도시 값 개수: \n{df['city'].value_counts()}")
#
# print("멤버십 레벨 결측 처리")
# df['membership_level'] = df['membership_level'].replace('', 'Bronze')
# print(f"대체 후 멤버십 값 개수: {df['membership_level'].value_counts()}")
#
# print("\n구매 이상치 처리")
# # Q1 = df['total_purchase'].quantile(0.25)
# # Q3 = df['total_purchase'].quantile(0.75)
# # IQR = Q3 - Q1
# # outlier_threashold_low = Q1 - 1.5 * IQR
# # outlier_threashold_high = Q1 + 1.5 * IQR
# df.loc[df['total_purchase'] < outlier_threashold_low, 'total_purchase'] = outlier_threashold_low
# df.loc[df['total_purchase'] > outlier_threashold_high, 'total_purchase'] = outlier_threashold_high
# print(df['total_purchase'].describe())
#
# print("범주형 데이터 인코딩")
# print("라벨 인코딩")
# membership_order = {'Bronze':1, 'Silver':2, 'Gold':3, 'Platinum':4}
# df['membership_level_encoded'] = df['membership_level'].map(membership_order)
# print("멤버십 레벨")
# print(df[['membership_level', 'membership_level_encoded']].head())
#
# print("원 핫 인코딩")
# df_encoded = pd.get_dummies(df, columns=['gender', 'city'], prefix=['gender', 'city'])
# print(f"인코딩 후 열 개수 {len(df_encoded)}")
# print(f"새로 생성 된 열 {[col for col in df_encoded.columns if col not in df.columns]}")
# print(df_encoded.head())

# 데이터 시각화
plt.rc('font', family='Malgun Gothic')
# 선 그래프(Line Plot)
# x = np.linspace(0, 10, 20)
# plt.figure(figsize=(10, 6))
# plt.plot(x, x, color='red', linewidth=3, marker='o', markerfacecolor='blue', markersize=10, linestyle='dashed')
# plt.plot(x, x**2, color='green', linewidth=3, linestyle='dotted', marker='o', markerfacecolor='orange', markersize=10)
# plt.grid(True)
# plt.title('선 그래프')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# 산점도(Scatter Plot)
# x = np.random.rand(50)
# y = np.random.rand(50)
# colors = np.random.rand(50)
# sizes = 1000 * np.random.rand(50)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap='viridis', marker='o', edgecolors='black')
# plt.colorbar(label='color')
# plt.title('산점도')
# plt.xlabel('x 값')
# plt.ylabel('y 값')
# plt.show()

# 막대 그래프(Bar Plot)
# categories = ['범주 A', '범주 B', '범주 C', '범주 D', '범주 E']
# values = [5, 7, 3, 8, 6]
# plt.figure(figsize=(10, 6))
# plt.bar(categories, values, width=0.8, color=['blue', 'orange', 'green', 'pink', 'yellow'], alpha=0.5, edgecolor='black', align='center')
# plt.grid(True, axis='y')
# plt.title('막대 그래프')
# plt.xlabel('x 값')
# plt.ylabel('y 값')
# for i, v in enumerate(values):
# 	plt.text(i, v+0.1, str(v), ha='center')
# plt.show()
# 수평 막대 그래프
# plt.barh(categories, values, color=['blue', 'orange', 'green', 'pink', 'yellow'], alpha=0.5, edgecolor='black', align='center')
# plt.grid(True, axis='x')
# plt.show()

# 히스토그램(Histogram)
plt.figure(figsize=(10, 6))
plt.hist(np.random.normal(0, 1, 1000), bins=30)
plt.rc('axes', unicode_minus=False)
plt.show()