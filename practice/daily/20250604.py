# # 거리 - 시간
# def distance_to_time(distance):
# 	speed = 80
# 	return distance / speed
#
# # 시간 - 비용
# def time_to_cost(time):
# 	hourly_cost = 10000
# 	return time * hourly_cost
#
# distance = 400
# time = distance_to_time(distance)
# cost = time_to_cost(time)
#
# # 거리가 1km 늘어나면 비용이 얼마나 늘어날까
# d_cost_d_time = 10000 # d(비용)/d(시간)
# d_time_d_distance = 1/80 # d(시간)/d(거리)
#
# d_cost_d_distance = d_cost_d_time * d_time_d_distance
# import numpy as np
#
#
# def simple_neuron_chain_rule():
# 	"""단일 뉴런에서의 연쇄법칙
# 	z = w*x + b (선형 변환)
# 	a = a(z) (시그모이드)
# 	L = (a-t)**2 (MSE)"""
#
# 	def sigmoid(z):
# 		return 1 / (1 + np.exp(-z))
#
# 	def sigmoid_derivative(s):
# 		s = sigmoid(s)
# 		return s * (1 - s)
#
# 	# 입력
# 	x = 1.0
# 	t = 0.8
#
# 	# 가중치 설정
# 	w = 0.5
# 	b = 0.2
#
# 	# 순전파
# 	z = w*x + b
# 	a = sigmoid(z)
#
# 	# 손실함수
# 	L = (a-t)**2
#
# 	# 역전파
# 	# dl/dw를 구하기 위해 연쇄법칙 활용
# 	# dl/da를 구함
# 	# u = a-t, L = u**2
# 	# dl/da = dl/du X du/da = 2u x 1 = 2(a-t)
# 	dl_da = 2*(a-t)
#
# 	da_dz = sigmoid_derivative(z)
#
# 	dz_dw = x
# 	# z = w*x+b
#
# 	# 연쇄법칙 적용
# 	dl_dw = dl_da * da_dz * dz_dw
# 	learning_rate = 0.1
# 	w_new = w - learning_rate * dl_dw
#
# 	print("가중치 업데이트")
# 	print(f"L : {L}")
# 	print(f"w_new : {w_new}")
#
# simple_neuron_chain_rule()
import numpy as np

X = np.array([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
])

y = np.array([
	[0],
	[1],
	[1],
	[0]
])


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))


class SimpleNeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):
		# 가중치 초기화 (랜덤 값)
		self.W1 = np.random.randn(input_size, hidden_size) * 0.5
		print(f"W1 형태 : {self.W1.shape}")

		self.b1 = np.zeros((1, hidden_size))
		print(f"b1 형태 : {self.b1.shape}")

		self.W2 = np.random.randn(hidden_size, output_size) * 0.5
		print(f"W2 형태 : {self.W2.shape}")

		self.b2 = np.zeros((1, output_size))
		print(f"b2 형태 : {self.b2.shape}")

		# 중간 계산 저장
		self.z1 = None
		# 활성화 함수 적용 결과
		self.a1 = None
		self.z2 = None
		self.a2 = None

	def forward(self, x):
		# 순전파 과정
		print("=== 순전파 과정 ===")

		self.z1 = np.dot(x, self.W1) + self.b1  # 선형 변환
		print(f"1단계 - 은닉층 입력 z1 형태: {self.z1.shape}")
		print(f"1단계 - 은닉층 입력 z1 결과: \n{self.z1}")

		self.a1 = sigmoid(self.z1)
		print(f"1단계 - 은닉층 출력 a1 형태: {self.a1.shape}")
		print(f"1단계 - 은닉층 출력 a1 결과: \n{self.a1}")

		self.z2 = np.dot(self.a1, self.W2) + self.b2
		print(f"2단계 - 출력층 입력 z2 형태: {self.z2.shape}")
		print(f"2단계 - 출력층 입력 z2 결과: \n{self.z2}")

		self.a2 = sigmoid(self.z2)
		print(f"2단계 - 출력층 출력 a2 형태: {self.a2.shape}")
		print(f"2단계 - 출력층 출력 a2 결과: \n{self.a2}")

		return self.a2

	def backward(self, X, y, learning_rate):
		# 손실함수의 역전파
		print("backward")

		m = X.shape[0]
		dL_da2 = 2 * (self.a2 - y) / m  # MSE 미분
		da2_dz2 = sigmoid_derivative(self.z2)
		dz2 = dL_da2 * da2_dz2

		dw2 = self.a1.T @ dz2
		db2 = np.sum(dz2, axis=0, keepdims=True)

		da1_dz1 = sigmoid_derivative(self.z1)
		dz1 = (dz2 @ self.W2.T) * da1_dz1

		dw1 = X.T @ dz1
		db1 = np.sum(dz1, axis=0, keepdims=True)

		self.W2 -= learning_rate * dw2
		self.b2 -= learning_rate * db2
		self.W1 -= learning_rate * dw1
		self.b1 -= learning_rate * db1

		return dw1, db1, dw2, db2

	def train(self, X, y, epochs, learning_rate=0.1):
		# 순전파
		losses = []
		for epoch in range(epochs):
			output = self.forward(X)

			# 손실 계산
			loss = np.mean((output - y) ** 2)
			losses.append(loss)

			self.backward(X, y, learning_rate)
		return losses


nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
before_output = nn.forward(X)
print(f"before_output.flatten(): {before_output.flatten()}")

loss = nn.train(X, y, epochs=1000)
# print(f"loss: {loss}")
after_output = nn.forward(X)
print(f"after_output.flatten(): {after_output.flatten()}")
print(f"정답 : {y.flatten()}")
