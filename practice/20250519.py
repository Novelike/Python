# 1. 다음 코드의 실행 결과는?
numbers = [1, 2, 3, 4, 5]
numbers[1:4] = [10, 20]
print("1.", numbers) # 1 10 20 5

# 2. 다음 코드의 실행 결과는?
def modify_list(lst):
	lst.append(4)
	lst = [7, 8, 9]
	return lst

original = [1, 2, 3]
result = modify_list(original)
print("2.", original) # 1, 2, 3, 4
print("2.", result) # 7, 8, 9

# 3. 다음 코드를 실행하면 어떤 오류가 발생할까요?
matrix = [[0] * 3] * 3
# matrix[0][0] = 1
# matrix[1][1] = 2
print("3.", matrix) # 곱하기 연산자로 인하여 모든 행의 요소가 변경됨

# 과제
# • 학생들의 이름과 점수 정보를 리스트로 관리하는 코드 구현
# • 다음 기능을 구현하세요:
# • 학생 추가: 이름과 점수를 입력 받아 목록에 추가
# • 학생 삭제: 이름을 입력 받아 해당 학생 정보 삭제
# • 성적 수정: 이름을 입력 받아 해당 학생의 점수 수정
# • 전체 목록 출력: 모든 학생의 이름과 점수 출력
# • 통계 출력: 최고 점수, 최저 점수, 평균 점수 계산 및 출력

student_list = []

def add_student(name, score):
	student_list.append((name, score))

def delete_student(name):
	for i, (student_name, _) in enumerate(student_list):
		if student_name == name:
			del student_list[i]
			break

def modify_student(name, score):
	for i, (student_name, _) in enumerate(student_list):
		if student_name == name:
			student_list[i] = (student_name, score)
			break

def get_all_students():
	print("Name\tScore")
	for student_name, score in student_list:
		print(f"{student_name}\t{score}")

def get_statistics():
	if not student_list:
		print("No students in the list")
		return
	max_score = min_score = student_list[0][1]
	total_score = 0
	for _, score in student_list:
		max_score = max(max_score, score)
		min_score = min(min_score, score)
		total_score += score
	average_score = total_score / len(student_list)
	print(f"Max Score: {max_score}")
	print(f"Min Score: {min_score}")
	print(f"Average Score: {average_score}")

add_student("John", 90)
add_student("Jane", 80)
add_student("Jack", 70)
add_student("Jill", 60)
get_all_students()
get_statistics()

modify_student("John", 100)
modify_student("Jane", 90)
get_all_students()
get_statistics()

delete_student("Jill")
get_all_students()
get_statistics()
