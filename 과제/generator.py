# 심화 256p
# • 로그 파일을 한 줄씩 읽는 제너레이터 함수 작성
# • 특정 패턴(예: 'ERROR', 'WARNING' 등)이 포함된 줄만 필터링하는 제너레이터 작성
import logging

logging.basicConfig(filename='./log/generator.log', level=logging.DEBUG, encoding='utf-8',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def read_log(filename):
	with open(filename, 'r') as f:
		for line in f:
			yield line


def filter_log(filename, pattern):
	with open(filename, 'r') as f:
		for line in f:
			if pattern in line:
				yield line


def main():
	log_file = './log/generator.log'

	print("=" * 20, "read_log", "=" * 20)
	for line in read_log(log_file):
		print(line, end='')

	print("=" * 20, "error_log", "=" * 20)
	for line in filter_log(log_file, 'ERROR'):
		print(line, end='')

	print("=" * 20, "warning_log", "=" * 20)
	for line in filter_log(log_file, 'WARNING'):
		print(line, end='')

	print("=" * 20, "info_log", "=" * 20)
	for line in filter_log(log_file, 'INFO'):
		print(line, end='')

	print("=" * 20, "critical_log", "=" * 20)
	for line in filter_log(log_file, 'CRITICAL'):
		print(line, end='')

	print("=" * 20, "debug_log", "=" * 20)
	for line in filter_log(log_file, 'DEBUG'):
		print(line, end='')


if __name__ == '__main__':
	main()
