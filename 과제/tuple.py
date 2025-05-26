# • 주어진 데이터셋에서 튜플을 활용하여 다음 분석을 수행하세요
# • 연도별 판매량 계산
# • 제품별 평균 가격 계산
# • 최대 판매 지역 찾기
# • 분기별 매출 분석

# 데이터: (연도, 분기, 제품, 가격, 판매량, 지역)
sales_data = [
	(2020, 1, "노트북", 1200, 100, "서울"),
	(2020, 1, "스마트폰", 800, 200, "부산"),
	(2020, 2, "노트북", 1200, 150, "서울"),
	(2020, 2, "스마트폰", 800, 250, "대구"),
	(2020, 3, "노트북", 1300, 120, "인천"),
	(2020, 3, "스마트폰", 850, 300, "서울"),
	(2020, 4, "노트북", 1300, 130, "부산"),
	(2020, 4, "스마트폰", 850, 350, "서울"),
	(2021, 1, "노트북", 1400, 110, "대구"),
	(2021, 1, "스마트폰", 900, 220, "서울"),
	(2021, 2, "노트북", 1400, 160, "인천"),
	(2021, 2, "스마트폰", 900, 270, "부산"),
	(2021, 3, "노트북", 1500, 130, "서울"),
	(2021, 3, "스마트폰", 950, 320, "대구"),
	(2021, 4, "노트북", 1500, 140, "부산"),
	(2021, 4, "스마트폰", 950, 370, "서울"),
]


# 연도별 판매량 계산
def calculate_yearly_sales():
	yearly_sales = {}
	for sale in sales_data:
		year = sale[0]
		quantity = sale[4]
		yearly_sales[year] = yearly_sales.get(year, 0) + quantity
	return yearly_sales


# 제품별 평균 가격 계산
def calculate_average_prices():
	product_prices = {}
	product_counts = {}
	for sale in sales_data:
		product = sale[2]
		price = sale[3]
		product_prices[product] = product_prices.get(product, 0) + price
		product_counts[product] = product_counts.get(product, 0) + 1
	return {product: product_prices[product] / product_counts[product]
	        for product in product_prices}


# 최대 판매 지역 찾기
def find_top_sales_region():
	region_sales = {}
	for sale in sales_data:
		region = sale[5]
		quantity = sale[4]
		region_sales[region] = region_sales.get(region, 0) + quantity
	return max(region_sales.items(), key=lambda x: x[1])


# 분기별 매출 분석
def analyze_quarterly_revenue():
	quarterly_revenue = {}
	for sale in sales_data:
		year = sale[0]
		quarter = sale[1]
		revenue = sale[3] * sale[4]
		key = (year, quarter)
		quarterly_revenue[key] = quarterly_revenue.get(key, 0) + revenue
	return quarterly_revenue


# 결과 출력
print("연도별 판매량:", calculate_yearly_sales())
print("제품별 평균 가격:", calculate_average_prices())
print("최대 판매 지역:", find_top_sales_region())
print("분기별 매출:", analyze_quarterly_revenue())
