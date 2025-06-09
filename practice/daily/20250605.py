# -*- coding: utf-8 -*-
# import numpy as np
#
# X = np.array([
# 	[0, 0],
# 	[0, 1],
# 	[1, 0],
# 	[1, 1]
# ])
#
# y = np.array([
# 	[0],
# 	[1],
# 	[1],
# 	[0]
# ])
#
# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))
#
# class SimpleNeuralNetwork:
# 	def __init__(self, input_size, hidden_size, output_size):
# 		self.W1 = np.random.randn(input_size, hidden_size) * 0.5
# 		print(f"W1 형태: {self.W1.shape}")
# 		print(f"W1: \n{self.W1}")
#
# 		self.b1 = np.zeros((1, hidden_size))
# 		print(f"b1 형태: {self.b1.shape}")
# 		print(f"b1: \n{self.b1}")
#
# 		self.W2 = np.random.randn(hidden_size, output_size) * 0.5
# 		print(f"W2 형태: {self.W2.shape}")
# 		print(f"W2: \n{self.W2}")
#
# 		self.b2 = np.zeros((1, output_size))
# 		print(f"b2 형태: {self.b2.shape}")
# 		print(f"b2: \n{self.b2}")
#
# 		# 중간 계산값 저장용 (역전파에서 사용)
# 		# z1은 은닉층의 선형 변환 결과입니다.
# 		self.z1 = None
# 		# a1은 은닉층의 활성화 함수 적용 결과입니다.
# 		self.a1 = None
# 		# z2는 출력층의 선형 변환 결과입니다.
# 		self.z2 = None
# 		# a2는 출력층의 활성화 함수 적용 결과입니다.
# 		self.a2 = None
#
# 	def forward(self, x):
# 		"""순전파 과정"""
# 		print("=== 순전파 과정 ===")
#
# 		# 1단계: 입력층 -> 은닉층
# 		self.z1 = np.dot(x, self.W1) + self.b1 # 선형 변환

import tkinter as tk
from tkinter import font

root = tk.Tk()
print("사용가능 폰트 패밀리:")
for f in font.families():
	print(f)

label = tk.Label(root, text="한글테스트: 안녕하세요", font=("NanumGothic", 16))
label.pack()
root.mainloop()
