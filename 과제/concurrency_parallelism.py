# 심화 315p
# • 5개의 공개 API URL에 GET 요청을 보냄
# • 세 가지 방식으로 구현하고 성능을 비교합니다:
	# • 순차 처리
	# • ThreadPoolExecutor 사용
	# • asyncio와 aiohttp 사용
# • API_URLS
	# • "https://jsonplaceholder.typicode.com/posts/1",
	# • "https://jsonplaceholder.typicode.com/posts/2",
	# • "https://jsonplaceholder.typicode.com/posts/3",
	# • "https://jsonplaceholder.typicode.com/posts/4",
	# • "https://jsonplaceholder.typicode.com/posts/5"

import time
import requests
import concurrent.futures
import asyncio
import aiohttp

# API URL 목록
API_URLS = [
	"https://jsonplaceholder.typicode.com/posts/1",
	"https://jsonplaceholder.typicode.com/posts/2",
	"https://jsonplaceholder.typicode.com/posts/3",
	"https://jsonplaceholder.typicode.com/posts/4",
	"https://jsonplaceholder.typicode.com/posts/5"
]

# 1. 순차 처리 방식
def fetch_sequential():
	start_time = time.time()
	results = []

	for url in API_URLS:
		response = requests.get(url)
		results.append(response.json())

	end_time = time.time()
	elapsed_time = end_time - start_time

	print(f"순차 처리: {elapsed_time:.4f}초 소요")
	return results, elapsed_time

# 2. ThreadPoolExecutor 사용
def fetch_url(url):
	response = requests.get(url)
	return response.json()

def fetch_thread_pool():
	start_time = time.time()

	with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
		results = list(executor.map(fetch_url, API_URLS))

	end_time = time.time()
	elapsed_time = end_time - start_time

	print(f"ThreadPoolExecutor: {elapsed_time:.4f}초 소요")
	return results, elapsed_time

# 3. asyncio와 aiohttp 사용
async def fetch_async(url, session):
	async with session.get(url) as response:
		return await response.json()

async def fetch_all_async():
	async with aiohttp.ClientSession() as session:
		tasks = [fetch_async(url, session) for url in API_URLS]
		return await asyncio.gather(*tasks)

def fetch_asyncio():
	start_time = time.time()

	results = asyncio.run(fetch_all_async())

	end_time = time.time()
	elapsed_time = end_time - start_time

	print(f"asyncio와 aiohttp: {elapsed_time:.4f}초 소요")
	return results, elapsed_time

# 세 가지 방식 실행 및 결과 비교
def main():
	print("===== 성능 비교 시작 =====")

	# 순차 처리
	sequential_results, sequential_time = fetch_sequential()

	# ThreadPoolExecutor
	thread_pool_results, thread_pool_time = fetch_thread_pool()

	# asyncio와 aiohttp
	asyncio_results, asyncio_time = fetch_asyncio()

	# 결과 확인 (첫 번째 항목의 제목만 출력)
	print("\n===== 결과 확인 =====")
	print(f"순차 처리 결과 예시: {sequential_results[0]['title']}")
	print(f"ThreadPoolExecutor 결과 예시: {thread_pool_results[0]['title']}")
	print(f"asyncio 결과 예시: {asyncio_results[0]['title']}")

	# 성능 비교
	print("\n===== 성능 비교 =====")
	print(f"순차 처리: {sequential_time:.4f}초")
	print(f"ThreadPoolExecutor: {thread_pool_time:.4f}초")
	print(f"asyncio와 aiohttp: {asyncio_time:.4f}초")

	# 성능 향상 계산
	thread_speedup = sequential_time / thread_pool_time
	async_speedup = sequential_time / asyncio_time

	print(f"\nThreadPoolExecutor 성능 향상: {thread_speedup:.2f}배")
	print(f"asyncio와 aiohttp 성능 향상: {async_speedup:.2f}배")

	if thread_speedup > async_speedup:
		print("\n이 테스트에서는 ThreadPoolExecutor가 가장 빠릅니다.")
	elif async_speedup > thread_speedup:
		print("\n이 테스트에서는 asyncio와 aiohttp가 가장 빠릅니다.")
	else:
		print("\n두 병렬 처리 방식의 성능이 비슷합니다.")

if __name__ == "__main__":
	main()