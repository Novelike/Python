# file = open('test.txt', 'w')
# file.write('hello')
# file.close()
#
# file = open('test.txt', 'r')
# print(file.read())
# file.close()

# def xor_encrypt_decrypt(input_file, output_file, key):
#
# 	with open(input_file, 'rb') as infile:
# 		data = infile.read()
#
# 	key_bytes = key.encode() if isinstance(key, str) else bytes([key])
# 	key_len = len(key_bytes)
#
# 	encrypted_data = bytearray(len(data))
# 	for i in range(len(data)):
# 		encrypted_data[i] = data[i] ^ key_bytes[i % key_len]
#
# 	with open(output_file, 'wb') as outfile:
# 		outfile.write(encrypted_data)
#
#
# # 암호화
# xor_encrypt_decrypt('test.txt', 'test_encrypted.enc', 'myKey123')
# xor_encrypt_decrypt('test_encrypted.enc', 'test_decrypted.txt', 'myKey123')

################################################################################################

# 디렉토리를 압축해서 백업하는 코드를 만들고 싶다
# import zipfile
# import datetime
# import os
# from pathlib import Path
#
# def backup_directory(source_dir, backup_dir=None, backup_name=None):
#
# 	source_path = Path(source_dir)
# 	if backup_dir is None:
# 		backup_dir = Path.cwd()
# 	else:
# 		backup_dir = Path(backup_dir)
# 		backup_dir.mkdir(parents=True, exist_ok=True)
#
# 	if backup_name is None:
# 		timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# 		backup_name = f"{source_path.name}_backup_{timestamp}.zip"
#
# 	backup_path = backup_dir / backup_name
#
# 	with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
# 		for root, _, files in os.walk(source_dir):
# 			for file in files:
# 				file_path = os.path.join(root, file)
#
# 				arc_name = os.path.relpath(file_path, os.path.dirname(source_dir))
# 				zipf.write(file_path, arcname=arc_name)
#
# backup_directory('datas', 'backups')

################################################################################################

# try:
# 	num = int(input('숫자 입력'))
# 	result = 100 / num
# except ValueError:
# 	print('유효한 숫자 입력 필요')
# except ZeroDivisionError:
# 	print('0으로 나눌 수 없습니다.')

################################################################################################

from functools import partial

def power(base, exponent, multiplier):
	return base ** exponent * multiplier

square_and_double = partial(power, 2, multiplier=2)
# 2 ** 3 * 2

cube = partial(power,3)
print(cube(2, multiplier=1))



# 함수를 데이터처럼
# event_handlers = {
# 	"LOGIN": handle_login
# 	"LOGIN": handle_logout
# }
#
# def process_event(event, hanlers)