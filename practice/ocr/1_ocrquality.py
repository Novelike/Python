import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt


def create_sample_image():
	image = np.ones((200, 600, 3), dtype=np.uint8) * 255

	font = cv2.FONT_HERSHEY_SIMPLEX

	cv2.putText(image, "Hello OCR World!", (50, 100), font, 2, (0, 0, 0), 3)
	cv2.putText(image, "This is a test image.", (50, 150), font, 1, (0, 0, 0), 2)

	cv2.imwrite("sample_text.png", image)
	return image

def basic_ocr_sample():
	image_path = "sample_text.png"

	image = cv2.imread(image_path)

	if image is None:
		print("이미지를 찾을 수 없습니다. 샘플 이미지를 생성합니다.")
		image = create_sample_image()

	text = pytesseract.image_to_string(image, lang='eng')
	print("인식된 텍스트:", text)

	return text, image

def create_high_quality_sample():
	image = np.ones((100, 400, 3), dtype=np.uint8) * 255

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image, "Perfect OCR Text", (29, 60), font, 1.2, (0, 0, 0), 2)
	cv2.imwrite("high_quality_sample.png", image)
	return image

def create_midium_quality_sample():
	image = np.ones((100, 400, 3), dtype=np.uint8) * 255

	noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
	image = cv2.add(image, noise)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image, "Noisy OCR Text", (29, 60), font, 1.2, (0, 0, 0), 2)
	cv2.imwrite("midium_quality_sample.png", image)
	return image

def create_low_quality_sample():
	image = np.ones((100, 400, 3), dtype=np.uint8) * 255

	noise = np.random.normal(0, 30, image.shape).astype(np.uint8)
	image = cv2.add(image, noise)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image, "Bad OCR Text", (29, 60), font, 1.2, (0, 0, 0), 2)

	rows, cols = image.shape[:2]
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)
	image = cv2.warpAffine(image, M, (cols, rows))

	image = cv2.GaussianBlur(image, (3, 3), 0)

	cv2.imwrite("low_quality_sample.png", image)
	return image

def ocr_quality_demo():
	samples = {
		'high_quality': create_high_quality_sample(),
		'midium_quality': create_midium_quality_sample(),
		'low_quality': create_low_quality_sample()
	}

	results = {}

	for quality, image in samples.items():
		text = pytesseract.image_to_string(image, lang='eng')
		data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
		print(data)

		confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
		avg_confidence = np.mean(confidences) if confidences else 0

		results[quality] = {
			'text': text.strip(),
			'confidence': avg_confidence,
			'image': image
		}

		print(f"{quality.upper()}:")
		print(f"    텍스트: '{text.strip()}'")
		print(f"    평균 신뢰도: {avg_confidence:.2f}%")
		print()

	return results

if __name__ == "__main__":
	text, image = basic_ocr_sample()
	quality_results = ocr_quality_demo()

	plt.figure(figsize=(15, 10))
	plt.rc('font', family='Malgun Gothic')
	plt.subplot(2, 3, 1)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title('기본 OCR 예제')
	plt.axis('off')

	for i, (quality, result) in enumerate(quality_results.items(), 2):
		plt.subplot(2, 3, i)
		plt.imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
		plt.title(f"{quality.title()}\n신뢰도: {result['confidence']:.2f}%")
		plt.axis('off')

	plt.tight_layout()
	plt.show()