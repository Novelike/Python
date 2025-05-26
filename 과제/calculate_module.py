# 심화 67p
# • 간단한 계산기 모듈을 만들어 보세요.
# • 모듈에는 덧셈, 뺄셈, 곱셈, 나눗셈 함수가 포함

def plus(a, b):
	return a + b


def minus(a, b):
	return a - b


def multiply(a, b):
	return a * b


def divide(a, b):
	return a / b


if __name__ == '__main__':
	n = 0
	loop = True
	while loop:
		print(f"{'-' * 5}계산기 모듈{'-' * 5}")
		print(f"현재값 : {n}")
		print("1. 더하기")
		print("2. 빼기")
		print("3. 곱하기")
		print("4. 나누기")
		print("5. 초기화")
		print("0. 종료")
		choice = input(": ")

		match choice:
			case "1":
				v = int(input(f"{n} + "))
				n = plus(n, v)
			case "2":
				v = int(input(f"{n} - "))
				n = minus(n, v)
			case "3":
				v = int(input(f"{n} * "))
				n = multiply(n, v)
			case "4":
				v = int(input(f"{n} / "))
				n = divide(n, v)
			case "5":
				n = 0
			case "0":
				break