import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


class TesseractOCR:
	"""
	Tesseract OCR 클래스
	"""

	def __init__(self):
		pass

	def get_available_languages(self):
		"""
		지원 가능한 언어 목록 확인
		"""
		langs = pytesseract.get_languages()
		return langs

	def ocr_with_bbox(self, image, lang="eng"):
		"""
		바운딩 박스 포함 OCR
		"""
		data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

		result_image = image.copy()

		for i in range(len(data["text"])):
			if int(data["conf"][i]) > 60:
				x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
				text = data["text"][i]

				if text.strip():
					cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
					cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		return result_image, data

class DocumentOCRDemo:
	"""
	문서 OCR 데모 클래스
	"""

	def __init__(self):
		pass

	def process_receipt_image(self, image_path):
		"""
		영수증 이미지 처리 예제
		"""
		image = cv2.imread(image_path)
		if image == None:
			image = self.create_sample_receipt()

		processed = self.preprocess_receipt(image)

		config = r"--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$ "
		text = pytesseract.image_to_string(processed, config=config)

		parsed_info = self.parse_receipt_info(text)

		return text, parsed_info, processed

	def preprocess_receipt(self, image):
		"""
		영수증 전용 전처리
		"""
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		enhanced = clahe.apply(gray)

		_, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		return thresh

	def parse_receipt_info(self, text):
		"""
		영수증 정보 파싱
		"""
		lines = text.split('\n')
		info = {
			'items': [],
			'total': None,
			'date': None,
			'store_name': None
		}

		for line in lines:
			line = line.strip()
			if not line:
				continue
			import re
			price_pattern = r"\$?\d+\.\d{2}"
			if re.search(price_pattern, line):
				if "total" in line.lower() or "sum" in line.lower():
					prices = re.findall(price_pattern, line)
					if prices:
						info['total'] = prices[-1]
				else:
					info['items'].append(line)

			date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"
			if re.search(date_pattern, line):
				info['date'] = line

		return info

	def create_sample_receipt(self):
		"""
		샘플 영수증 이미지 생성
		"""
		image = np.ones((600, 400, 3), dtype=np.uint8) * 255

		font = cv2.FONT_HERSHEY_SIMPLEX
		texts = [
			('GROCERY STORE', (100, 50), 0.8, 2),
			('Date: 01/15/2024', (50, 100), 0.6, 1),
			('Item 1        $5.99', (50, 150), 0.6, 1),
			('Item 2        $3.50', (50, 180), 0.6, 1),
			('Item 3        $12.25', (50, 210), 0.6, 1),
			('Tax           $1.89', (50, 260), 0.6, 1),
			('Total         $32.14', (50, 310), 0.7, 2),
			('Thank you for shopping!', (120, 360), 0.6, 1)
		]

		for text, pos, scale, thickness in texts:
			cv2.putText(image, text, pos, font, scale, (0, 0, 0), thickness)

		return image

def tesseract_practice():
	"""
	Tesseract OCR 실습
	"""

	print("=== Tesseract OCR 실습 ===\n")

	print("1. 기본 OCR 테스트")
	ocr = TesseractOCR()

	print("지원 언어:", ocr.get_available_languages()[:10])

	print("\n2. 영수증 OCR 데모")
	demo = DocumentOCRDemo()
	receipt_text, parsed_info, processed_image = demo.process_receipt_image(None)

	print("영수증 텍스트:")
	print(receipt_text)
	print("\n파싱된 정보:")
	for key, value in parsed_info.items():
		print(f"{key}: {value}")

	sample_image = demo.create_sample_receipt()

	return sample_image, processed_image

if __name__ == "__main__":
	sample_image, processed_image = tesseract_practice()

	# 결과 시각화
	plt.rc('font', family='Malgun Gothic')
	plt.figure(figsize=(15, 5))

	plt.subplot(1, 3, 1)
	plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
	plt.title("원본 영수증")
	plt.axis('off')

	plt.subplot(1, 3, 2)
	plt.imshow(processed_image, cmap="gray")
	plt.title("전처리된 이미지")
	plt.axis('off')

	ocr = TesseractOCR()
	bbox_image, _ = ocr.ocr_with_bbox(sample_image)

	plt.subplot(1, 3, 3)
	plt.imshow(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))
	plt.title("OCR 결과 (바운딩 박스)")
	plt.axis('off')

	plt.tight_layout()
	plt.show()