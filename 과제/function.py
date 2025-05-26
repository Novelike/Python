# • CSV 파일을 읽어 딕셔너리 리스트로 변환하는 함수 작성
# • 학생 중 성적이 80점 이상인 학생만 필터링
# • 필터링된 학생들의 평균 나이 계산
# • 모든 함수 호출 시간을 측정하는 데코레이터 적용

import csv

def csv_to_dict(path):
	data_dict = {}
	with open(path, 'r', encoding='utf-8') as csvfile:
		csv_reader = csv.DictReader(csvfile)
		for row in csv_reader:
			key = row[csv_reader.fieldnames[0]]
			data_dict[key] = row
	return data_dict

def filter_by_score(data_dict):
	filtered_dict = {}
	for key, value in data_dict.items():
		if int(value['score']) >= 80:
			filtered_dict[key] = value
	return filtered_dict

def get_average_age(filtered_dict):
	ages = []
	for key, value in filtered_dict.items():
		ages.append(int(value['age']))
	return sum(ages) / len(ages)

def measure_time(func):
	import time
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		print(f'모든 함수 호출 시간 {end_time - start_time} sec')
		return result
	return wrapper

@measure_time
def main():
	data_dict = csv_to_dict('students.csv')
	filtered_dict = filter_by_score(data_dict)
	average_age = get_average_age(filtered_dict)
	print(f'{'-'*10}Result{'-'*10}')
	print('전체 학생')
	for key, value in data_dict.items():
		print(value)
	print('80점 이상인 학생')
	for key, value in filtered_dict.items():
		print(value)
	print(f'80점 이상인 학생의 평균 나이: {average_age}')

if __name__ == '__main__':
	main()