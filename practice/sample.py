from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QFont
import sys

app = QApplication(sys.argv)

label = QLabel('한글 테스트: 안녕하세요! 😊')
label.setFont(QFont("NanumGothic", 16))  # 또는 "Noto Sans CJK KR", "Malgun Gothic" 등
label.show()

sys.exit(app.exec())
