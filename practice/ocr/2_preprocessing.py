import cv2
import numpy as np
import matplotlib.pyplot as plt


class OCRPreprocessor:

	def __init__(self):
		pass

	def convert_to_grayscale(self, image):
		"""
		컬러 이미지를 그레이스케일로 변환
		"""
		# 이미지 차원 확인 (3차원=컬러, 2차원=그레이스케일)
		if len(image.shape) == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			gray = image.copy()
		return gray

	# 이진화 처리
	def apply_threshold(self, image, method='adaptive'):

		gray = self.convert_to_grayscale(image)

		if method == 'simple':
			_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
		elif method == 'adaptive':
			thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		elif method == 'otsu':
			_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		return thresh

	def remove_noise(self, image):

		# 침식과 팽창으로 노이즈 점을 제거
		# 가우시안 블러
		kernel = np.ones((1, 1), np.uint8)

		opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
		denoised = cv2.GaussianBlur(opening, (1, 1), 0)

		return denoised

	def correct_skew(self, image):
		# Canny 에지 검출
		edges = cv2.Canny(image, 50, 150, apertureSize=3)
		# 허프변환 함수 -> 직선
		lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

		if lines is not None:
			angles = []
			for rho, theta in lines[:, 0]:
				# 허프 변환에서 theta의 의미:
				# theta = 0도 (0 radian)     -> 수직선 (|)
				# theta = 90도 (pi/2 radian) -> 수평선 (-)
				# theta = 180도 (pi radian)  -> 다시 수직선 (|)
				# 이미지 회전에서 각도의 의미:
				# 0도    -> 수평선 기준 (-)
				# 90도   -> 수직선 기준 (|)
				# 양수    -> 시계방향 회전
				# 음수    -> 반시계방향 회전
				# 즉, 허프 변환은 수직선을 0도로 보지만, 이미지 회전은 수평선을 0도로 보기 때문에 -90도 보정
				angle = np.degrees(theta) - 90
				angles.append(angle)

				median_angle = np.median(angles)

				(h, w) = image.shape[:2]
				center = (w // 2, h // 2)
				M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
			rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

			return rotated

		else:
			return image

	def resize_image(self, image, target_height=800):
		h, w = image.shape[:2]

		if h < target_height:
			scale = target_height / h
			new_w = int(w * scale)
			resized = cv2.resize(image, (new_w, target_height),
			                     interpolation=cv2.INTER_CUBIC)
		else:
			resized = image

		return resized

	def visualize_preprocessing_steps(self, steps, step_names):
		"""
		전처리 과정을 단계별로 시각화
		"""
		plt.rc('font', family='Malgun Gothic')
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))

		axes = axes.ravel()

		for i, (step, name) in enumerate(zip(steps, step_names)):
			if i < len(axes):
				if len(step.shape) == 3:
					axes[i].imshow(cv2.cvtColor(step, cv2.COLOR_BGR2RGB))
				else:
					axes[i].imshow(step, cmap='gray')

				axes[i].set_title(name)
				axes[i].axis('off')

		plt.tight_layout()
		plt.show()

	def preprocessing_pipeline(self, image, visualize=False):
		"""
		전체 이미지 전처리 파이프라인 실행
		"""
		steps = []
		step_names = []

		steps.append(image.copy())
		step_names.append("원본 이미지")

		gray = self.convert_to_grayscale(image)
		steps.append(gray)
		step_names.append("그레이스케일")

		resized = self.resize_image(gray)
		steps.append(resized)
		step_names.append("크기 조정")

		thresh = self.apply_threshold(resized, method="adaptive")
		steps.append(thresh)
		step_names.append("이진화")

		denoised = self.remove_noise(thresh)
		steps.append(denoised)
		step_names.append("노이즈 제거")

		corrected = self.correct_skew(denoised)
		steps.append(corrected)
		step_names.append("기울기 보정")

		if visualize:
			self.visualize_preprocessing_steps(steps, step_names)

		return corrected

def create_noisy_sample_image():
	"""
	테스트용 노이즈가 있는 샘플 이미지 생성
	"""
	image = np.ones((300, 800, 3), dtype=np.uint8) * 255

	font = cv2.FONT_HERSHEY_SIMPLEX

	cv2.putText(image, "Noisy OCR Test Image", (50, 100), font, 1.5, (0, 0, 0), 2)

	cv2.putText(image, "Preprocessing improves accuracy.", (50, 150), font, 1, (0, 0, 0), 2)

	cv2.putText(image, "Machine Learning & AI", (50, 200), font, 1, (0, 0, 0), 2)

	noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)

	noisy_image = cv2.add(image, noise)

	h, w = noisy_image.shape[:2]

	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, 5, 1.0)
	skewed_image = cv2.warpAffine(noisy_image, M, (w, h))
	return skewed_image

def preprocessing_example():
	"""
	OCR 전처리 예제 실행 함수
	"""
	preprocessor = OCRPreprocessor()

	image = create_noisy_sample_image()

	processed_image = preprocessor.preprocessing_pipeline(image, visualize=True)

	try:
		import pytesseract

		original_text = pytesseract.image_to_string(image)

		processed_text = pytesseract.image_to_string(processed_image)

		print("전처리 전 OCR 결과:")
		print(repr(original_text))
		print("\n전처리 후 OCR 결과:")
		print(repr(processed_text))

	except ImportError:
		print("pytesseract가 설치되지 않아 OCR 비교를 건너뜁니다.")
		print("설치: pip install pytesseract")

	return processed_image

if __name__ == "__main__":
	preprocessing_example()