# # iterator
# my_list = [1, 2, 3]
# iterator = iter(my_list)
# print(next(iterator))
# print(next(iterator))
# print(next(iterator))
#
# # generator
# def count_to_up(max):
# 	count = 1
# 	while count <= max:
# 		yield count
# 		count += 1
#
# counter = count_to_up(5)
# print(f"제네레이터 출력 : {next(counter)}")
# print(f"제네레이터 출력 : {next(counter)}")
#
# for num in count_to_up(5):
# 	print(f"for 루프 출력 : {num}")
#
# # 리스트 컴프리헨션 vs 제네레이터 표현식
# squares_list = [x**2 for x in range(1000)] # 리스트 컴프리헨션
# squares_gen = (x**2 for x in range(1000)) # 제네레이터 표현식
#
# import sys
# print(sys.getsizeof(squares_list))
# print(sys.getsizeof(squares_gen))
#
# for num2 in squares_gen:
# 	print(num2)
#
# # 제네레이터 함수의 상태 관리 예시
# def stateful_generator():
# 	print("첫 번째 값 생성")
# 	yield 1
#
# 	print("두 번째 값 생성")
# 	yield 2
#
# 	print("세 번째 값 생성")
# 	yield 3
#
# gen = stateful_generator()
# print(next(gen))
# print("중간 작업 수행")
# print(next(gen))
# print(next(gen))
#
# # yield from 사용하여 제네레이터 중첩 루프 표현
# def nested_generator():
# 	yield from "A"
# 	yield from "B"
# 	yield from "!"
#
# def main_generator():
# 	yield 1
# 	yield 2
# 	yield 3
# 	yield from nested_generator()
# 	yield 4
# 	yield 5
#
# for item in main_generator():
# 	print(item)
#
# # 무한 데이터 스트림
# import random
# import time
# # 시계열 데이터
# def sensor_date_stream():
# 	while True:
# 		temperature = 20+random.uniform(-5, 5)
# 		yield f"온도 : {temperature:.2f}도, 시간 : {time.strftime('%Y-%m-%d %H:%M:%S')}"
# 		time.sleep(1)
#
# stream = sensor_date_stream()
# for _ in range(10):
# 	print(next(stream))


# # Thread
# import threading
# import time
#
# def background_task():
# 	while True:
# 		print("background task is running...")
# 		time.sleep(1)
#
# # 데몬 스레드 생성
# my_thread = threading.Thread (target=background_task, daemon=True)
# my_thread.start()
#
# print("main thread is running...")
# time.sleep(3)
# print("main thread is finished...")

# # Thread event
# import threading
# import time
#
# event = threading.Event()
#
# def waiter():
# 	print("waiter is waiting...")
# 	event.wait()
# 	print("waiter is running...")
# 	time.sleep(1)
# 	print("waiter is finished...")
# 	event.clear()
#
# def setter():
# 	print("setter is running...")
# 	time.sleep(3)
# 	print("setter is setting event...")
# 	event.set()
# 	print("setter is finished...")
#
# # start thread
# t1 = threading.Thread(target=waiter)
# t2 = threading.Thread(target=setter)
#
# t1.start()
# t2.start()

# # Thread condition
# import threading
# import time
#
# # 데이터와 Condition 객체
# data = None
# condition = threading.Condition()
#
# def wait_for_data():
# 	print("wait_for_data is running...")
# 	with condition:
# 		condition.wait()
# 		print(f"wait_for_data is received data : {data}")

# # 데이터를 기다리는 스레드
# def prepare_data():
# 	global data
# 	print("prepare_data is running...")
# 	time.sleep(3)
# 	with condition:
# 		data = "data is ready"
# 		print("prepare_data is finished...")
# 		condition.notify()
#
# t1 = threading.Thread(target=wait_for_data)
# t2 = threading.Thread(target=prepare_data)
#
# t1.start()
# t2.start()
#
# t1.join()
# t2.join()

# # GIL(Global Interpreter Lock)
# import threading
# import time
#
# counter = 0
# counter_lock = threading.Lock()
#
# def increment(count, name):
# 	global counter
# 	for _ in range(count):
# 		with counter_lock:
# 			current = counter
# 			time.sleep(0.001)
# 			counter = current + 1
# 			print(f"{name} increment : {counter}")
#
# t1 = threading.Thread(target=increment, args=(1000, 't1'))
# t2 = threading.Thread(target=increment, args=(1000, 't2'))
#
# t1.start()
# t2.start()
#
# t1.join()
# t2.join()
#
# print(f"counter : {counter}")

# # Queue
# import queue
# import threading
# import time
# import random
#
# # 작업 큐
# task_queue = queue.Queue()
#
# # 결과 큐
# result_queue = queue.Queue()
#
# # 작업 생성 함수
# def create_tasks():
# 	print("create_tasks is running...\n")
# 	for i in range(10):
# 		task = f"task-{i}"
# 		task_queue.put(task)
# 		print(f"task-{i} is created.\n")
# 		time.sleep(random.uniform(0.1, 0.3))
#
# 	for _ in range(3):
# 		task_queue.put(None)
#
# 	print("create_tasks is finished...\n")
#
# # 작업 처리 함수
# def worker(worker_id):
# 	print(f"worker-{worker_id} is running...\n")
#
# 	while True:
# 		# 작업 가져오기
# 		task = task_queue.get()
#
# 		# 작업 끝났는지 확인
# 		if task is None:
# 			print(f"worker-{worker_id} is finished...\n")
# 			break
#
# 		# 작업 처리 중
# 		print(f"worker-{worker_id} is processing task : {task}\n")
# 		processing_time = random.uniform(0.5, 1.5)
# 		time.sleep(processing_time)
#
# 		# 결과 제출
# 		result = f"result of {task} is processed after {processing_time:.2f} seconds\n"
# 		result_queue.put((worker_id, result))
#
# 		# 작업 완료 표시
# 		task_queue.task_done()
# 		print(f"worker-{worker_id} is finished processing task : {task}\n")
# 		print(f"size of remaining tasks in task queue : {task_queue.qsize()}\n")
#
# # 결과 수집 함수
# def result_collector():
# 	print("result_collector is running...\n")
# 	results = []
#
# 	for _ in range(10):
# 		worker_id, result = result_queue.get()
# 		print(f"result of worker-{worker_id} : {result}")
# 		results.append(result)
# 		result_queue.task_done()
#
# 	print("result_collector is finished...\n")
# 	print(f"results : {results}")
#
# creator = threading.Thread(target=create_tasks)
# workers = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
# collector = threading.Thread(target=result_collector)
#
# creator.start()
# for w in workers:
# 	w.start()
# collector.start()
#
# creator.join()
# for w in workers:
# 	w.join()
# collector.join()
#
# print("All tasks are finished.")

# # ThreadPoolExecutor
# import concurrent.futures
# import time
#
# def task(param):
# 	name, duration = param
# 	print(f"작업 {name} 시작")
# 	time.sleep(duration)
# 	print(f"작업 {name} 완료")
# 	return f"{name} 리턴값"
#
# # 작업 목록
# params = [
# 	("A", 2),
# 	("B", 1),
# 	("C", 3)
# ]
#
# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
# 	results = list(executor.map(task, params))
#
# 	for result in results:
# 		print(result)

# # Process + Lock
# import multiprocessing
# import time
#
# def add_to_shared(shared_value, lock, increment):
# 	for i in range(5):
# 		with lock:
# 			shared_value.value += increment
# 		time.sleep(0.1)
# 	print(f"프로세스 {multiprocessing.current_process().name} 완료\n")
#
# if __name__ == "__main__":
# 	shared_number = multiprocessing.Value("i", 0)
# 	lock = multiprocessing.Lock()
#
# 	p1 = multiprocessing.Process(target=add_to_shared, args=(shared_number, lock, 10))
# 	p2 = multiprocessing.Process(target=add_to_shared, args=(shared_number, lock, 20))
#
# 	p1.start()
# 	p2.start()
#
# 	p1.join()
# 	p2.join()
#
# 	print(f"Final shared number : {shared_number.value}\n")

# # Process + Queue
# import multiprocessing
# import random
# import time
#
#
# def producer_process(queue):
# 	print(f"생산자 프로세스 시작 : {multiprocessing.current_process().name}\n")
# 	for i in range(5):
# 		item = f"데이터-{i}\n"
# 		queue.put(item)
# 		print(f"생산 : {item}\n")
# 		time.sleep(random.uniform(0.1, 0.5))
# 	# 작업 완료 신호
# 	queue.put(None)
# 	print("생산자 프로세스 종료\n")
#
#
# def consumer_process(queue):
# 	print(f"소비자 프로세스 시작 : {multiprocessing.current_process().name}\n")
# 	while True:
# 		item = queue.get()
# 		if item is None:
# 			break
# 		print(f"소비 : {item}\n")
# 		time.sleep(random.uniform(0.2, 0.7))
# 	print("소비자 프로세스 종료\n")
#
#
# if __name__ == "__main__":
# 	q = multiprocessing.Queue()
# 	prod = multiprocessing.Process(target=producer_process, args=(q,))
# 	cons = multiprocessing.Process(target=consumer_process, args=(q,))
#
# 	prod.start()
# 	cons.start()
#
# 	prod.join()
# 	cons.join()
#
# 	print("모든 프로세스 종료\n")


# Process + Pool
import multiprocessing
import time
import os



# 병렬 처리할 작업 함수
def process_task(task_id):
	process_id = os.getpid()
	print(f"Process {process_id} is running task {task_id}...")
	# CPU 작업 시뮬레이션
	result = 0
	for i in range(10000000):
		result += i
	print(f"Process {process_id} completed task {task_id}.")
	return task_id, result, process_id

if __name__ == "__main__":
	# 시스템의 CPU 코어 수 확인
	num_cores = multiprocessing.cpu_count()
	print(f"System has {num_cores} CPU cores.")

	# 작업 목록 생성
	tasks = range(10)

	# 순차 처리 시간 측정
	start_time = time.time()
	sequential_results = [process_task(i) for i in tasks]
	end_time = time.time()
	print(f"Sequential processing time: {end_time - start_time:.2f} sec")

	# 병렬 처리 시간 측정
	start_time = time.time()
	with multiprocessing.Pool(num_cores) as pool:
		parallel_results = pool.map(process_task, tasks)
	end_time = time.time()
	print(f"Parallel processing time: {end_time - start_time:.2f} sec")

	# 사용된 프로세스 ID 확인
	process_ids = set(result[2] for result in parallel_results)
	print(f"사용된 프로세스 수 : {len(process_ids)}")
	print(f"프로세스 ID 목록 : {process_ids}")
