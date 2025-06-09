# items = [("name", "이영희"), ("age", 25), ("grade", "C")]
# print(items)
# print(items[0])
# print(items[0][1])
# print(dict(items))
#
# person = {
# 	"name": "홍길동",
# 	"age": 30,
# 	"city": "서울",
# 	"skills": ["python", "java", "c"]
# }
#
# for k, v in person.items():
# 	print(f"key : {k}, value : {v}")
#
# users = {
# 	"user1": {
# 		"name": "홍길동",
# 		"age": 35
# 	},
# 	"user2": {
# 		"name": "임꺽정",
# 		"age": 30
# 	}
# }
#
# for user_id, user_info in users.items():
# 	for k, v in user_info.items():
# 		print(f"user_id : {user_id}, {k} : {v}")
#
# fruits = ['apple', 'banana', 'orange']
# fruit_length = {fruit : len(fruit) for fruit in fruits}
# print(fruit_length)
# print(fruit_length.get('apple'))

# import requests
#
# response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
#
# if response.status_code == 200:
# 	data = response.json()
#
# 	print(f"게시물 ID : {data['id']}")
# 	print(f"제목 : {data['title']}")
# 	print(f"내용 : {data['body']}")
# 	print(f"작성자 ID : {data['userId']}")
# else:
# 	print(f"오류 발생 : {response.status_code}")

# fruits = {"사과", "바나나", "체리"}
# numbers = set([1, 2, 3, 4])
# chars = set("hello")
# empty_set = {} # 딕셔너리
# squares = {x**2 for x in range(1, 11)}

# # 기본 집합 생성
# A = {1, 2, 3, 4, 5}
# B = {4, 5, 6, 7, 8}
#
# # 합집합 (Union): A와 B의 모든 요소
# print(A | B)
# print(A.union(B))
#
# # 교집합 (Intersection): A와 B 모두에 있는 요소
# print(A & B)
# print(A.intersection(B))
#
# # 차집합 (Difference): A에는 있지만 B에 없는 요소
# print(A - B)
# print(A.difference(B))
#
# nmbs = [1, 2, 3, 2, 1, 4, 5, 4, 3, 2]
# print(nmbs)
# unique_nmbs = sorted(list(set(nmbs)))
# print(unique_nmbs)

# a = True
# b = False
# c = True
#
# print(a and b)
# print(a and c)
# print(b and c)
# print(a or b)
# print(a or c)
# print(b or c)
# print(a and (b or c))

# result1 = -2 ** 2
# result2 = -2 ** -2
# print(result1)
# print(result2)

age = 20

if age < 14:
    print("초등")
elif age < 20:
    if age == 18:
        print()
    print("미성년")
else:
    print("성인")


# 튜플 Quiz & 과제

# 1. 다음 코드의 실행 결과는?
t1 = (1, 2, 3)
t2 = (4, 5, 6)
result = t1 + t2
print(result[2:5])  # 3, 4, 5

# 2. 다음 코드에서 오류가 발생하는 라인은?
person = ("홍길동", 30, "서울")
name, age, city = person
age = age + 1
# person[1] = age # Error : 튜플은 대입을 허용하지 않는다.
print(
    f"{name}은 {age}세이고 {city}에 살고 있습니다."
)  # 홍길동은 31세이고 서울에 살고 있습니다.


# 3. 다음 코드의 실행 결과는?
def get_values():
    return 1, 2, 3


x, *y = get_values()
print(x)  # 1
print(y)  # [2, 3]

# 과제
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
