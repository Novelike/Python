# 심화 129p
# • 다음 클래스를 구현하세요:
#   • `Book`: 도서 정보(제목, 저자, ISBN, 출판연도 등)를 관리  
#   • `Library`: 도서 컬렉션을 관리하고 대출/반납 기능 제공
#   • `Member`: 도서관 회원 정보와 대출 목록 관리
# • 다음 기능을 구현하세요:
#   • 도서 추가/삭제
#   • 도서 검색(제목, 저자, ISBN으로)
#   • 도서 대출/반납
#   • 회원 등록/관리
#   • 회원별 대출 현황 확인
# • 객체 지향 설계 원칙(SOLID)을 최소한 2가지 이상 적용하세요.
# • 적절한 캡슐화를 통해 데이터를 보호하세요.

class Book:

	def __init__(self, title, author, isbn, year, price):
		self._title = title
		self._author = author
		self._isbn = isbn
		self._year = year
		self._price = price
		self._is_borrowed = False

	def get_title(self):
		return self._title

	def get_author(self):
		return self._author

	def get_isbn(self):
		return self._isbn

	def get_year(self):
		return self._year

	def get_price(self):
		return self._price

	def is_borrowed(self):
		return self._is_borrowed

	def borrow(self):
		if self._is_borrowed:
			return False

		self._is_borrowed = True
		return True

	def return_book(self):
		if not self._is_borrowed:
			return False

		self._is_borrowed = False
		return True

	def __str__(self):
		status = "대출 중" if self._is_borrowed else "대출 가능"
		return f"[{self._isbn}] {self._title} - {self._author} ({self._year}) | {status}"


class Member:

	def __init__(self, member_id, name, phone_number, email, address):
		self._id = member_id
		self._name = name
		self._phone_number = phone_number
		self._email = email
		self._address = address
		self._borrowed_books = []
		self._is_active = True

	def get_id(self):
		return self._id

	def get_name(self):
		return self._name

	def is_active(self):
		return self._is_active

	def get_borrowed_books(self):
		return self._borrowed_books.copy()  # 복사본 반환하여 원본 보호

	def update_contact(self, phone=None, email=None, address=None):
		if phone:
			self._phone_number = phone
		if email:
			self._email = email
		if address:
			self._address = address

	def deactivate(self):
		self._is_active = False

	def activate(self):
		self._is_active = True

	def borrow_book(self, book):
		if book not in self._borrowed_books:
			self._borrowed_books.append(book)
			return True
		return False

	def return_book(self, book):
		if book in self._borrowed_books:
			self._borrowed_books.remove(book)
			return True
		return False

	def __str__(self):
		status = "활성" if self._is_active else "비활성"
		return f"회원 {self._id}: {self._name} ({status}) | 현재 대출 도서: {len(self._borrowed_books)}권"


class Library:

	def __init__(self, name):
		self._name = name
		self._books = []
		self._members = {}  # 회원 ID를 키로 사용

	def get_name(self):
		return self._name

	def get_total_books(self):
		return len(self._books)

	def get_available_books(self):
		count = 0
		for book in self._books:
			if not book.is_borrowed():
				count += 1
		return count

	def get_total_members(self):
		return len(self._members)

	def add_book(self, book):
		if book not in self._books:
			self._books.append(book)
			return True
		return False

	def delete_book(self, book):
		if book in self._books and not book.is_borrowed():
			self._books.remove(book)
			return True
		return False

	def register_member(self, member):
		if member.get_id() not in self._members:
			self._members[member.get_id()] = member
			return True
		return False

	def get_member(self, member_id):
		return self._members.get(member_id)

	def search_books(self, query, search_type='title'):
		results = []

		if search_type.lower() == 'title':
			for book in self._books:
				if query.lower() in book.get_title().lower():
					results.append(book)

		elif search_type.lower() == 'author':
			for book in self._books:
				if query.lower() in book.get_author().lower():
					results.append(book)

		elif search_type.lower() == 'isbn':
			for book in self._books:
				if query == book.get_isbn():
					results.append(book)

		else:
			raise ValueError(f"지원하지 않는 검색 유형: {search_type}")

		return results

	def borrow_book(self, book, member):
		if book not in self._books:
			raise ValueError("도서관에 없는 책입니다.")

		if member.get_id() not in self._members:
			raise ValueError("등록된 회원이 아닙니다.")

		if not member.is_active():
			raise ValueError("비활성화된 회원입니다.")

		if book.is_borrowed():
			raise ValueError("이미 대출 중인 책입니다.")

		if book.borrow() and member.borrow_book(book):
			return True

		return False

	def return_book(self, book, member):
		if book not in self._books:
			raise ValueError("도서관에 없는 책입니다.")

		if member.get_id() not in self._members:
			raise ValueError("등록된 회원이 아닙니다.")

		if not book.is_borrowed():
			raise ValueError("대출 상태가 아닌 책입니다.")

		if book in member.get_borrowed_books():
			if book.return_book() and member.return_book(book):
				return True
		else:
			raise ValueError("이 회원이 대출한 책이 아닙니다.")

		return False

	def get_member_borrowed_books(self, member_id):
		member = self.get_member(member_id)
		if not member:
			raise ValueError("등록된 회원이 아닙니다.")

		return member.get_borrowed_books()

	def __str__(self):
		return f"{self._name} | 보유 도서: {self.get_total_books()}권 (대출 가능: {self.get_available_books()}권) | 회원 수: {self.get_total_members()}명"

if __name__ == "__main__":
	# 도서관 생성
	library = Library("파이썬 도서관")

	# 도서 추가
	books = [
		Book("파이썬 프로그래밍", "이파이", "ISBN-001", 2023, 25000),
		Book("자료구조와 알고리즘", "김알고", "ISBN-002", 2022, 30000),
		Book("인공지능 입문", "박머신", "ISBN-003", 2024, 35000),
		Book("웹 개발의 정석", "정웹발", "ISBN-004", 2023, 28000),
		Book("데이터베이스 기초", "디비킴", "ISBN-005", 2024, 32000)
	]

	for book in books:
		library.add_book(book)

	# 회원 등록
	members = [
		Member("M001", "홍길동", "010-1234-5678", "hong@example.com", "서울시"),
		Member("M002", "김철수", "010-8765-4321", "kim@example.com", "부산시"),
		Member("M003", "이영희", "010-2345-6789", "lee@example.com", "대구시"),
		Member("M004", "박지민", "010-3456-7890", "park@example.com", "인천시")
	]

	for member in members:
		library.register_member(member)

	print("=== 도서관 현황 ===")
	print(library)

	# 도서 검색
	print("\n=== 도서 검색 결과 ===")
	print("제목 검색 - '파이썬':")
	for book in library.search_books("파이썬", "title"):
		print(book)

	print("\n저자 검색 - '김알고':")
	for book in library.search_books("김알고", "author"):
		print(book)

	# 도서 대출/반납
	try:
		print("\n=== 도서 대출/반납 테스트 ===")
		library.borrow_book(books[0], members[0])
		print(f"{members[0].get_name()}님이 '{books[0].get_title()}' 대출")

		library.borrow_book(books[1], members[0])
		print(f"{members[0].get_name()}님이 '{books[1].get_title()}' 대출")

		print(f"\n{members[0].get_name()}님의 대출 목록:")
		for book in library.get_member_borrowed_books("M001"):
			print(f"- {book.get_title()}")

		library.return_book(books[0], members[0])
		print(f"\n{members[0].get_name()}님이 '{books[0].get_title()}' 반납")

		print(f"\n{members[0].get_name()}님의 현재 대출 목록:")
		for book in library.get_member_borrowed_books("M001"):
			print(f"- {book.get_title()}")

	except ValueError as e:
		print(f"오류 발생: {e}")