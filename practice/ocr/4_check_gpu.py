import platform
import subprocess


def check_gpu_info():
	print(f"운영체제: {platform.system()} {platform.release()}")

	try:
		result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
		if result.returncode == 0:
			print("NVIDIA GPU 감지됨")
			print("GPU 정보:")
			print(result.stdout)
		else:
			print("NVIDIA GPU가 없거나 드라이버 에러")
	except FileNotFoundError:
		print("NVIDIA 드라이버가 설치되지 않음")

check_gpu_info()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

device_cpu = torch.device("cpu")

if torch.cuda.is_available():
	device_gpu = torch.device("cuda")