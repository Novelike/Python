# 심화 215p
# • 파일 처리기 구현
# • 다양한 유형의 파일(텍스트, CSV, JSON, 바이너리)을 읽고 쓸 수 있어야 합니다
# • 파일이 존재하지 않거나, 권한이 없거나, 형식이 잘못된 경우 등 다양한 오류 상황을 적절히 처리
# • 사용자 정의 예외 계층 구조를 설계하고 구현
# • 오류 발생 시 로깅을 통해 문제를 기록
# • 모든 파일 작업은 컨텍스트 매니저(`with` 구문)를 사용
import logging

logging.basicConfig(filename='./log/error.log', level=logging.ERROR, encoding='utf-8',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

folder = './sample/'

def read_file(file_name, binary=False):
	try:
		if binary:
			with open(folder + file_name, 'rb') as f:
				return f.read()
		else:
			with open(folder + file_name, 'r') as f:
				return f.read()
	except FileNotFoundError:
		print(f'File not found: {file_name}')
		logging.error(f'File not found: {file_name}')
		return None
	except PermissionError:
		print(f'Permission denied: {file_name}')
		logging.error(f'Permission denied: {file_name}')
		return None
	except Exception as e:
		print(f'An error occurred: {e}')
		logging.error(f'An error occurred: {e}')
		return None


def write_file(file_name, content, binary=False):
	try:
		if binary:
			with open(folder + file_name, 'wb') as f:
				f.write(content)
		else:
			with open(folder + file_name, 'w') as f:
				f.write(content)
	except Exception as e:
		print(f'An error occurred: {e}')
		logging.error(f'An error occurred: {e}')


if __name__ == '__main__':
	txt_file = 'sample.txt'
	content1 = 'This is a sample text file.'
	write_file(txt_file, content1, binary=False)

	csv_file = 'sample.csv'
	content2 = '1,2,3,4,5\n6,7,8,9,10'
	write_file(csv_file, content2, binary=False)

	json_file = 'sample.json'
	content3 = '{"name": "John", "age": 30}'
	write_file(json_file, content3, binary=False)

	bin_file = 'sample.bin'
	content4 = b'\x01\x02\x03\x04\x05\x06\x07\x08'
	write_file(bin_file, content4, binary=True)

	result1 = read_file(txt_file, binary=False)
	print(result1)

	result2 = read_file(csv_file, binary=False)
	print(result2)

	result3 = read_file(json_file, binary=False)
	print(result3)

	result4 = read_file(bin_file, binary=True)
	print(result4)

	error_result = read_file('error.txt', binary=False)
	print(error_result)
