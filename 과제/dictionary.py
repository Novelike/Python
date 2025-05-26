# • 딕셔너리를 활용하여 간단한 주소록 프로그램 작성
# • 연락처 이름을 키로 하고, 전화번호, 이메일, 주소 등의 정보를 값으로 저장
# • 중첩 딕셔너리 구조를 사용하여 각 연락처마다 여러 정보를 저장
# • 연락처 추가, 삭제, 검색, 수정, 모든 연락처 보기 기능을 구현

addressBook = {}

def create_address_book(name, phone, email, address):
	addressBook[name] = {
		'phone': phone,
		'email': email,
		'address': address
	}
	print(f'Successfully created {name} in address book.')

def delete_address_book(name):
	del addressBook[name]
	print(f'Successfully deleted {name} from address book.')

def search_address_book(name):
	print(f'Search {name} in address book...')
	if name in addressBook:
		for key, value in addressBook[name].items():
			print(f'{key} : {value}')
	else:
		print(f'{name} : No such contact')

def update_address_book(name, phone, email, address):
	addressBook[name] = {
		'phone': phone,
		'email': email,
		'address': address
	}
	print(f'Successfully updated {name} in address book.')

create_address_book('kjh', '010-1234-5678', 'kjh@example.com', 'Guri')
create_address_book('jnk', '010-1234-5679', 'jnk@example.com', 'Seoul')
create_address_book('john', '010-9876-5432', 'john@example.com', 'California')

search_address_book('kjh')
search_address_book('jnk')
search_address_book('john')

delete_address_book('john')
search_address_book('john')

update_address_book('kjh', '010-7612-8629', 'kimjh@example.com', 'Seoul')
search_address_book('kjh')