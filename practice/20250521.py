from datetime import datetime, date, time, timedelta

# 현재 날짜와 시간
now = datetime.now()
print(f"현재 : {now}")

# 특정 날짜/시간 생성
specific_date = datetime(2025, 5, 21, 11, 13, 15)
print(f"특정 일시 : {specific_date}")

# 문자열 형식 변환
date_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"형식화된 날짜: {date_str}")

# 문자열을 날짜로 파싱
parsed_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
print(f"파싱된 날짜: {parsed_date}")


# 클래스와 객체지향
# - 실습: 은행 계좌 클래스 만들기


class BankAccount:

	def __init__(self, owner, balance=0):
		self.owner = owner
		self.balance = balance
		self.transaction_history = []

	# 입금
	def deposit(self, amount):
		self.balance += amount
		print(f"{amount}원이 입금되었습니다.")

	# 출금
	def withdraw(self, amount):
		self.balance -= amount
		print(f"{amount}원이 출급되었습니다.")

	def _log_transaction(self, type, amount):
		self.transaction_history.append({
			'타입': type,
			'금액': amount,
			'잔고': self.balance
		})
