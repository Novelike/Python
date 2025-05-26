import json
from datetime import datetime

import cloudscraper
from bs4 import BeautifulSoup


def crawl_steam_store():
	try:
		print("크롤러 초기화 중...")
		scraper = cloudscraper.create_scraper()

		print("페이지 로딩 시작...")
		# 검색 API를 사용하여 할인 게임 목록 가져오기
		response = scraper.get("https://store.steampowered.com/search/?specials=1&filter=topsellers")

		if response.status_code != 200:
			print(f"오류: 상태 코드 {response.status_code}")
			return None

		soup = BeautifulSoup(response.text, 'html.parser')
		games_list = []

		# 게임 요소 찾기 (검색 결과 행)
		game_elements = soup.find_all(class_="search_result_row")
		print(f"\n발견된 게임 수: {len(game_elements)}")

		for element in game_elements:
			try:
				# 제목
				title_elem = element.find(class_="title")
				if not title_elem:
					continue
				title = title_elem.text.strip()

				# 가격 및 할인 정보
				discount_block = element.find(class_="discount_block")
				if discount_block:
					# 데이터 속성에서 가격 정보 추출
					price_final = discount_block.get('data-price-final')
					discount_pct = discount_block.get('data-discount')

					# 최종 가격 추출
					final_price_elem = discount_block.find(class_="discount_final_price")
					if final_price_elem:
						price = final_price_elem.text.strip()
					elif price_final:
						# 데이터 속성에서 가격 추출 (단위: 원)
						price_in_won = int(price_final) / 100
						price = f"₩ {price_in_won:,.0f}"
					else:
						price = "가격 정보 없음"

					# 할인율 추출
					discount_pct_elem = discount_block.find(class_="discount_pct")
					if discount_pct_elem:
						discount = discount_pct_elem.text.strip()
					elif discount_pct:
						discount = f"-{discount_pct}%"
					else:
						discount = "할인 없음"
				else:
					# 할인 블록이 없는 경우 (정가 판매)
					price_container = element.find(class_="search_price_discount_combined")
					if price_container:
						price_final = price_container.get('data-price-final')
						if price_final:
							price_in_won = int(price_final) / 100
							price = f"₩ {price_in_won:,.0f}"
						else:
							price = "가격 정보 없음"
					else:
						price = "가격 정보 없음"

					discount = "할인 없음"

				# 태그/장르 - 검색 결과에서는 직접 태그를 가져올 수 없으므로 게임 페이지에서 가져오기
				genres = []

				game_data = {
					'title': title,
					'price': price,
					'discount': discount,
					'genres': genres,
					'crawled_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				}

				print(f"발견된 게임: {title}")
				games_list.append(game_data)

			except Exception as e:
				print(f"개별 게임 처리 중 오류: {str(e)}")
				continue

		# 중복 제거 및 정렬
		unique_games = {game['title']: game for game in games_list}.values()

		def extract_price(price_str):
			try:
				return float(price_str.replace('₩ ', '').replace(',', ''))
			except:
				return float('inf')

		sorted_games = sorted(unique_games,
		                      key=lambda x: extract_price(x['price'])
		                      if x['price'] != "가격 정보 없음" else float('inf'))

		# 결과 저장
		with open('steam_games.json', 'w', encoding='utf-8') as f:
			json.dump({
				'crawled_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				'total_games': len(sorted_games),
				'games': list(sorted_games)
			}, f, ensure_ascii=False, indent=2)

		print(f"\n크롤링 완료: {len(sorted_games)}개의 게임 정보를 저장했습니다.")

		# HTML 소스 저장 (디버깅용)
		with open('page_source.html', 'w', encoding='utf-8') as f:
			f.write(response.text)

		return sorted_games

	except Exception as e:
		print(f"크롤링 중 오류 발생: {str(e)}")
		return None

if __name__ == "__main__":
	games = crawl_steam_store()

	if games:
		print("\n=== 가격순으로 정렬된 게임 목록 ===")
		for game in games:
			print(f"\n제목: {game['title']}")
			print(f"가격: {game['price']}")
			print(f"할인: {game['discount']}")
			if game['genres']:
				print(f"장르: {', '.join(game['genres'])}")
