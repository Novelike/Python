# • 소셜 네트워크에서 사용자 간의 관계와 추천 시스템을 구현하는 프로그램을 작성
# • 공통 관심사를 갖는 친구 응답
# • 공통 관심사가 없는 친구 응답

def get_all_by_interests():
	all_interests = set().union(*[set(interests) for _, interests in my_sns.items()])

	interest_to_users = {}
	for interest in all_interests:
		users = {name for name, interests in my_sns.items() if interest in interests}
		interest_to_users[interest] = users

		print(f"[{interest}]")
		for user in sorted(users):
			print(f"  {user}")

	return interest_to_users


def find_common_interests(user1, user2):
	if user1 not in my_sns or user2 not in my_sns:
		return set()
	return set(my_sns[user1]) & set(my_sns[user2])


def find_users_with_no_common_interests(user):
	if user not in my_sns:
		return set()

	user_interests = set(my_sns[user])
	return {
		other_user for other_user in my_sns
		if other_user != user and not user_interests & set(my_sns[other_user])
	}


def add_user(name, interests):
	my_sns[name] = interests
	print(f"사용자 '{name}'이(가) 추가되었습니다.")


def remove_user(name):
	if name in my_sns:
		del my_sns[name]
		print(f"사용자 '{name}'이(가) 제거되었습니다.")
	else:
		print(f"사용자 '{name}'을(를) 찾을 수 없습니다.")


def display_menu():
	print("\n===== SNS 관심사 분석 프로그램 =====")
	print("1. 모든 관심사별 사용자 보기")
	print("2. 두 사용자 간의 공통 관심사 찾기")
	print("3. 특정 사용자와 공통 관심사가 없는 사용자 찾기")
	print("4. 새 사용자 추가")
	print("5. 사용자 제거")
	print("6. 모든 사용자 정보 보기")
	print("0. 종료")
	print("===================================")


def run_interactive_mode():
	while True:
		display_menu()
		choice = input("원하는 작업의 번호를 입력하세요: ")

		if choice == "0":
			print("프로그램을 종료합니다.")
			break

		elif choice == "1":
			print("\n===== 관심사별 사용자 목록 =====")
			get_all_by_interests()

		elif choice == "2":
			print("\n사용자 목록:")
			users = list(my_sns.keys())
			for idx, name in enumerate(users, 1):
				print(f"{idx}. {name}")

			try:
				user1_idx = int(input("\n첫 번째 사용자 번호를 선택하세요: ")) - 1
				user2_idx = int(input("두 번째 사용자 번호를 선택하세요: ")) - 1

				if not (0 <= user1_idx < len(users) and 0 <= user2_idx < len(users)):
					print("잘못된 사용자 번호입니다.")
					continue

				user1 = users[user1_idx]
				user2 = users[user2_idx]
				common = find_common_interests(user1, user2)

				print(f"\n'{user1}'와(과) '{user2}' 간의 공통 관심사:")
				if common:
					for interest in sorted(common):
						print(f"  - {interest}")
				else:
					print("  공통 관심사가 없습니다.")
			except ValueError:
				print("올바른 숫자를 입력해주세요.")

		elif choice == "3":
			user = input("사용자 이름: ")

			if user not in my_sns:
				print(f"사용자 '{user}'을(를) 찾을 수 없습니다.")
				continue

			no_common = find_users_with_no_common_interests(user)

			print(f"\n'{user}'와(과) 공통 관심사가 없는 사용자:")
			if no_common:
				for other_user in sorted(no_common):
					print(f"  - {other_user}")
			else:
				print("  모든 사용자와 적어도 하나의 공통 관심사가 있습니다.")

		elif choice == "4":
			name = input("새 사용자 이름: ")
			interests_input = input("관심사(쉼표로 구분): ")
			interests = [interest.strip() for interest in interests_input.split(",")]

			add_user(name, interests)

		elif choice == "5":
			name = input("제거할 사용자 이름: ")
			remove_user(name)

		elif choice == "6":
			print("\n===== 모든 사용자 정보 =====")
			for name, interests in my_sns.items():
				print(f"{name}: {', '.join(interests)}")

		else:
			print("잘못된 입력입니다. 다시 시도해주세요.")

		input("\n계속하려면 Enter 키를 누르세요...")


# 프로그램 실행
if __name__ == "__main__":
	my_sns = {
		"Alice": ["음악", "영화", "독서"],
		"Bob": ["스포츠", "여행", "음악"],
		"Charlie": ["프로그래밍", "게임", "영화"],
		"David": ["요리", "여행", "사진"],
		"Eve": ["프로그래밍", "독서", "음악"],
		"Frank": ["스포츠", "게임", "요리"],
		"Grace": ["영화", "여행", "독서"]
	}

	run_interactive_mode()
