# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, mixed_precision
from keras.mixed_precision import global_policy, set_global_policy, Policy
import random
from collections import deque
import time
import tkinter as tk
from tkinter import messagebox
from minesweeper import MinesweeperGame, MainMenu
import itertools
import platform

# DirectML TensorFlow 설정 및 메모리 최적화
try:
	print("TensorFlow 버전:", tf.__version__)

	# DirectML 디바이스 설정
	# DirectML은 기본적으로 사용 가능한 모든 GPU를 사용합니다
	# 특별한 설정이 필요하지 않습니다

	# 사용 가능한 DirectML 디바이스 확인
	physical_devices = tf.config.list_physical_devices()
	print("사용 가능한 물리적 디바이스:", physical_devices)

	# GPU 디바이스 확인
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		print(f"DirectML GPU 감지됨: {len(gpus)}개")
		for gpu in gpus:
			# GPU 메모리 증가 설정 (DirectML에서 지원하는 경우)
			try:
				tf.config.experimental.set_memory_growth(gpu, True)
				print(f"메모리 증가 설정 활성화: {gpu}")
			except:
				print(f"메모리 증가 설정을 지원하지 않음: {gpu}")

		# 혼합 정밀도 설정 (DirectML에서 지원하는 경우)
		try:
			mixed_precision.set_global_policy('mixed_float16')
			print("혼합 정밀도 활성화: mixed_float16")
		except:
			print("혼합 정밀도를 지원하지 않습니다. 기본 정밀도 사용")
	else:
		print("DirectML GPU를 찾을 수 없습니다. CPU만 사용합니다.")

	# CPU 설정 (Intel CPU 최적화)
	if platform.processor().startswith('Intel'):
		# Intel CPU 최적화 설정
		os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
		print("Intel CPU 최적화 활성화")

	# DirectML 디버그 정보 출력 (문제 해결에 도움이 됨)
	os.environ['TF_DIRECTML_VERBOSE'] = '1'

except Exception as e:
	print(f"DirectML 설정 중 오류 발생: {e}")
	print("기본 설정으로 계속합니다.")


# 메타 학습을 위한 클래스
class MetaLearningAgent:
	"""여러 게임 구성에서 학습하고 지식을 전이하는 메타 학습 에이전트"""

	def __init__(self, configurations=None):
		"""
		Args:
			configurations: 학습할 게임 구성 목록 [(width, height, mines), ...]
						   기본값은 초급, 중급, 고급 난이도
		"""
		if configurations is None:
			# 기본 구성: 초급, 중급, 고급
			self.configurations = [
				(9, 9, 10),  # 초급
				(16, 16, 40),  # 중급
				(30, 16, 99)  # 고급
			]
		else:
			self.configurations = configurations

		# 각 구성별 AI 에이전트
		self.agents = {}

		# 공유 경험 메모리
		self.shared_memory = deque(maxlen=50000)

	def initialize_agents(self, use_inference=True, architecture='cnn'):
		"""모든 구성에 대한 에이전트 초기화"""
		for width, height, mines in self.configurations:
			config_key = f"{width}x{height}_{mines}"
			self.agents[config_key] = MinesweeperAI(
				width=width,
				height=height,
				mines=mines,
				use_inference=use_inference,
				architecture=architecture
			)

	def train_all(self, episodes_per_config=500, visualize=False, self_play=True):
		"""모든 구성에 대해 순차적으로 학습"""
		for width, height, mines in self.configurations:
			config_key = f"{width}x{height}_{mines}"
			print(f"\n학습 시작: 구성 {config_key} ({episodes_per_config} 에피소드)")

			# 현재 구성에 대한 에이전트 학습
			self.agents[config_key].train(
				episodes=episodes_per_config,
				visualize=visualize,
				self_play=self_play
			)

			# 학습된 경험 공유
			self._share_experiences(config_key)

		# 공유된 경험으로 모든 에이전트 추가 학습
		self._train_from_shared_experiences()

	def _share_experiences(self, source_config_key):
		"""특정 구성의 경험을 공유 메모리에 추가"""
		source_agent = self.agents[source_config_key]

		# 에이전트의 경험 메모리에서 일부 샘플링하여 공유 메모리에 추가
		if len(source_agent.agent.memory) > 0:
			# 최대 1000개 경험 샘플링
			sample_size = min(1000, len(source_agent.agent.memory))
			experiences = random.sample(list(source_agent.agent.memory), sample_size)

			# 공유 메모리에 추가
			for experience in experiences:
				self.shared_memory.append(experience)

	def _train_from_shared_experiences(self, batch_size=64, epochs=5):
		"""공유된 경험으로 모든 에이전트 추가 학습"""
		if len(self.shared_memory) < batch_size:
			print("공유 메모리에 충분한 경험이 없습니다.")
			return

		print(f"\n공유 경험으로 모든 에이전트 추가 학습 (공유 메모리 크기: {len(self.shared_memory)})")

		for config_key, agent in self.agents.items():
			print(f"에이전트 {config_key} 추가 학습 중...")

			# 각 에이전트에 대해 공유 메모리에서 배치 학습 수행
			for _ in range(epochs):
				# 배치 샘플링
				minibatch = random.sample(self.shared_memory, batch_size)

				# 배치 학습
				for state, action, reward, next_state, done in minibatch:
					# 상태 크기 조정 (필요한 경우)
					if state.shape[0] != agent.height or state.shape[1] != agent.width:
						# 크기가 다른 경우 스킵
						continue

					action_type, row, col = action

					# 행/열 범위 확인
					if row >= agent.height or col >= agent.width:
						continue

					# 에이전트 학습
					agent.agent.remember(state, action, reward, next_state, done)

				# 리플레이 수행
				agent.agent.replay()

	def play_game(self, width=9, height=9, mines=10):
		"""특정 구성으로 게임 플레이"""
		config_key = f"{width}x{height}_{mines}"

		if config_key in self.agents:
			self.agents[config_key].play_game()
		else:
			print(f"구성 {config_key}에 대한 에이전트가 없습니다.")

			# 가장 가까운 구성 찾기
			closest_key = self._find_closest_config(width, height, mines)

			if closest_key:
				print(f"가장 가까운 구성 {closest_key}의 에이전트로 플레이합니다.")
				self.agents[closest_key].play_game()
			else:
				print("플레이할 수 있는 에이전트가 없습니다.")

	def _find_closest_config(self, width, height, mines):
		"""주어진 구성과 가장 가까운 구성 찾기"""
		if not self.agents:
			return None

		closest_key = None
		min_distance = float('inf')

		for config_key in self.agents:
			w, h, m = map(int, config_key.replace('x', '_').split('_'))

			# 유클리드 거리 계산
			distance = ((w - width) ** 2 + (h - height) ** 2 + (m - mines) ** 2) ** 0.5

			if distance < min_distance:
				min_distance = distance
				closest_key = config_key

		return closest_key


# 확률적 추론을 위한 클래스
class ProbabilisticInference:
	"""지뢰찾기 게임에서 확률적 추론을 수행하는 클래스"""

	def __init__(self, width, height):
		self.width = width
		self.height = height

	def analyze_board(self, state_matrix, board=None):
		"""게임 보드를 분석하여 각 셀의 지뢰 확률 계산"""
		# 결과 저장용 확률 맵 초기화 (기본값: -1 = 알 수 없음)
		probability_map = np.full((self.height, self.width), -1.0)

		# 이미 열린 셀 또는 깃발이 있는 셀은 확률 설정
		for row in range(self.height):
			for col in range(self.width):
				if state_matrix[row, col] > 0:  # 숫자가 있는 열린 셀
					probability_map[row, col] = 0.0  # 지뢰 확률 0%
				elif state_matrix[row, col] == BOARD_REPRESENTATION['flag']:
					probability_map[row, col] = 1.0  # 지뢰 확률 100%
				elif state_matrix[row, col] == BOARD_REPRESENTATION['empty']:
					probability_map[row, col] = 0.0  # 지뢰 확률 0%

		# 숫자 셀 주변의 닫힌 셀에 대한 제약 조건 수집
		constraints = []
		for row in range(self.height):
			for col in range(self.width):
				if state_matrix[row, col] > 0:  # 숫자가 있는 열린 셀
					# 주변 닫힌 셀 찾기
					closed_cells = []
					for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
						nr, nc = row + dr, col + dc
						if 0 <= nr < self.height and 0 <= nc < self.width:
							if state_matrix[nr, nc] == BOARD_REPRESENTATION['unknown']:
								closed_cells.append((nr, nc))

					# 주변 깃발 수 계산
					flag_count = 0
					for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
						nr, nc = row + dr, col + dc
						if 0 <= nr < self.height and 0 <= nc < self.width:
							if state_matrix[nr, nc] == BOARD_REPRESENTATION['flag']:
								flag_count += 1

					# 제약 조건 추가: 주변 닫힌 셀 중 (숫자 - 깃발 수) 개가 지뢰
					if closed_cells:
						constraints.append((closed_cells, int(state_matrix[row, col]) - flag_count))

		# 단순 제약 조건 처리
		self._apply_simple_constraints(probability_map, constraints)

		# 고급 제약 조건 처리 (제약 조건 간의 관계 분석)
		if len(constraints) > 1:
			self._apply_advanced_constraints(probability_map, constraints)

		# 전역 확률 계산 (남은 지뢰 수 기반)
		if board is not None:
			self._apply_global_probability(probability_map, state_matrix, board)

		return probability_map

	def _apply_simple_constraints(self, probability_map, constraints):
		"""단순 제약 조건 적용 (확실한 지뢰/안전 셀 식별)"""
		for cells, mines in constraints:
			# 모든 닫힌 셀이 지뢰인 경우
			if len(cells) == mines and mines > 0:
				for row, col in cells:
					probability_map[row, col] = 1.0

			# 지뢰가 없는 경우 (모든 셀이 안전)
			elif mines == 0:
				for row, col in cells:
					probability_map[row, col] = 0.0

	def _apply_advanced_constraints(self, probability_map, constraints):
		"""고급 제약 조건 적용 (제약 조건 간의 관계 분석)"""
		# 제약 조건 간의 부분집합 관계 확인
		for (cells1, mines1), (cells2, mines2) in itertools.combinations(constraints, 2):
			# cells1이 cells2의 부분집합인 경우
			if set(cells1).issubset(set(cells2)):
				# cells2 - cells1에 있는 셀들의 지뢰 수는 mines2 - mines1
				diff_cells = [cell for cell in cells2 if cell not in cells1]
				diff_mines = mines2 - mines1

				# 모든 차이 셀이 지뢰인 경우
				if len(diff_cells) == diff_mines and diff_mines > 0:
					for row, col in diff_cells:
						probability_map[row, col] = 1.0

				# 차이 셀에 지뢰가 없는 경우
				elif diff_mines == 0:
					for row, col in diff_cells:
						probability_map[row, col] = 0.0

			# cells2가 cells1의 부분집합인 경우
			elif set(cells2).issubset(set(cells1)):
				# cells1 - cells2에 있는 셀들의 지뢰 수는 mines1 - mines2
				diff_cells = [cell for cell in cells1 if cell not in cells2]
				diff_mines = mines1 - mines2

				# 모든 차이 셀이 지뢰인 경우
				if len(diff_cells) == diff_mines and diff_mines > 0:
					for row, col in diff_cells:
						probability_map[row, col] = 1.0

				# 차이 셀에 지뢰가 없는 경우
				elif diff_mines == 0:
					for row, col in diff_cells:
						probability_map[row, col] = 0.0

	def _apply_global_probability(self, probability_map, state_matrix, board):
		"""전역 확률 계산 (남은 지뢰 수 기반)"""
		# 남은 지뢰 수 계산
		total_mines = sum(1 for row in board for cell in row if cell == 'X')
		flagged_mines = np.sum(state_matrix == BOARD_REPRESENTATION['flag'])
		remaining_mines = total_mines - flagged_mines

		# 아직 확률이 결정되지 않은 셀 수
		unknown_cells = np.sum(probability_map == -1.0)

		# 남은 셀이 있고 남은 지뢰가 있는 경우 전역 확률 계산
		if unknown_cells > 0 and remaining_mines > 0:
			global_probability = remaining_mines / unknown_cells

			# 아직 확률이 결정되지 않은 셀에 전역 확률 적용
			for row in range(self.height):
				for col in range(self.width):
					if probability_map[row, col] == -1.0:
						probability_map[row, col] = global_probability

	def get_safe_moves(self, state_matrix, board=None):
		"""안전한 이동(확률 0%)과 지뢰가 확실한 위치(확률 100%) 반환"""
		probability_map = self.analyze_board(state_matrix, board)

		safe_moves = []  # 확률 0%인 셀 (안전한 이동)
		mine_cells = []  # 확률 100%인 셀 (지뢰 확실)

		for row in range(self.height):
			for col in range(self.width):
				if state_matrix[row, col] == BOARD_REPRESENTATION['unknown']:
					if probability_map[row, col] == 0.0:
						safe_moves.append((ACTION_LEFT_CLICK, row, col))
					elif probability_map[row, col] == 1.0:
						mine_cells.append((ACTION_RIGHT_CLICK, row, col))

		return safe_moves, mine_cells, probability_map


# 자동 재시작을 위한 MinesweeperGame 확장 클래스
class AutoRestartMinesweeperGame(MinesweeperGame):
	"""지뢰찾기 게임 클래스 확장 - 자동 재시작 기능 추가"""

	def __init__(self, master, width=10, height=10, mines=15, main_menu_callback=None):
		super().__init__(master, width, height, mines, main_menu_callback)
		# 메시지 박스 자동 처리를 위한 플래그
		self.auto_restart = True

	def handle_left_click(self, row, col):
		"""왼쪽 클릭 이벤트 처리 - 메시지 박스 자동 처리"""
		# 게임이 끝났으면 아무 동작 안함
		if self.game_over:
			return

		# 플래그가 있는 칸이면 아무 동작 안함
		if self.buttons[row][col].cget('text') == '🚩':
			return

		# 이미 열린 셀인지 확인
		if (row, col) in self.opened_cells:
			# 이미 열린 셀이면 chord_click 실행
			self.chord_click(row, col)
			return

		# 첫 클릭 시 타이머 시작
		if self.start_time is None:
			self.start_time = time.time()

		# 지뢰 클릭 시 게임 오버
		if self.board[row][col] == 'X':
			self.buttons[row][col].config(text='💣', bg='red')
			self.game_over = True
			self.reveal_all()

			if self.auto_restart:
				# 메시지 박스 표시 대신 자동으로 게임 재시작
				self.master.after(500, self.reset_game)
			else:
				messagebox.showinfo("게임 오버", "지뢰를 밟았습니다!")
			return

		# 빈 칸(0) 클릭 시 주변 빈 칸들 자동 열기
		self.reveal(row, col)

		# 승리 조건 확인
		self.check_win()

	def chord_click(self, row, col):
		"""코드 클릭 처리 - 메시지 박스 자동 처리"""
		# 숫자가 있는 셀만 처리
		cell_text = self.buttons[row][col].cget('text')
		if not cell_text or not cell_text.isdigit():
			return

		cell_number = int(cell_text)

		# 주변 깃발 수 계산
		flag_count = 0
		for i in range(max(0, row - 1), min(self.height, row + 2)):
			for j in range(max(0, col - 1), min(self.width, col + 2)):
				if self.buttons[i][j].cget('text') == '🚩':
					flag_count += 1

		# 주변 셀 좌표 리스트 (깃발이 아닌 닫힌 셀)
		surrounding_cells = []
		for i in range(max(0, row - 1), min(self.height, row + 2)):
			for j in range(max(0, col - 1), min(self.width, col + 2)):
				if (i, j) != (row, col) and (i, j) not in self.opened_cells and self.buttons[i][j].cget('text') != '🚩':
					surrounding_cells.append((i, j))

		# 클릭 효과 보여주기
		self.show_click_effect(surrounding_cells)

		# 깃발 수가 셀의 숫자와 일치하면 주변 셀들 열기
		if flag_count == cell_number:
			for i, j in surrounding_cells:
				# 지뢰 클릭 시 게임 오버
				if self.board[i][j] == 'X':
					self.buttons[i][j].config(text='💣', bg='red')
					self.game_over = True
					self.reveal_all()

					if self.auto_restart:
						# 메시지 박스 표시 대신 자동으로 게임 재시작
						self.master.after(500, self.reset_game)
					else:
						messagebox.showinfo("게임 오버", "지뢰를 밟았습니다!")
					return

				# 셀 열기
				self.reveal(i, j)

			# 승리 조건 확인
			self.check_win()

	def check_win(self):
		"""승리 조건 확인 - 메시지 박스 자동 처리"""
		# 승리 조건 1: 모든 지뢰가 아닌 칸이 열림
		unopened = 0
		for i in range(self.height):
			for j in range(self.width):
				if self.buttons[i][j].cget('state') != 'disabled' and self.board[i][j] != 'X':
					unopened += 1

		# 승리 조건 2: 모든 지뢰에 플래그가 있음
		correct_flags = 0
		for i in range(self.height):
			for j in range(self.width):
				if self.buttons[i][j].cget('text') == '🚩' and self.board[i][j] == 'X':
					correct_flags += 1

		if unopened == 0 or correct_flags == self.mines:
			self.game_over = True
			self.reveal_all()
			elapsed_time = int(time.time() - self.start_time)

			if self.auto_restart:
				# 메시지 박스 표시 대신 자동으로 게임 재시작
				self.master.after(500, self.reset_game)
			else:
				messagebox.showinfo("승리", f"축하합니다! 게임에서 이겼습니다!\n소요 시간: {elapsed_time}초")


# Constants
BOARD_REPRESENTATION = {
	'unknown': 0,  # 아직 열리지 않은 셀
	'flag': -1,  # 깃발이 꽂힌 셀
	'mine': -2,  # 지뢰 (학습용, 실제 게임에서는 보이지 않음)
	'empty': 9,  # 빈 셀 (숫자가 0인 셀)
	# 1-8: 주변 지뢰 수
}

# 행동 정의
ACTION_LEFT_CLICK = 0  # 왼쪽 클릭 (셀 열기)
ACTION_RIGHT_CLICK = 1  # 오른쪽 클릭 (깃발 설치/제거)
ACTION_CHORD_CLICK = 2  # 코드 클릭 (주변 셀 열기)


class MinesweeperState:
	"""지뢰찾기 게임 상태를 AI가 처리할 수 있는 형태로 변환"""

	def __init__(self, board, opened_cells, flags, width, height):
		self.width = width
		self.height = height
		self.board = board  # 실제 게임 보드 (지뢰 위치 포함)
		self.opened_cells = opened_cells  # 열린 셀 집합
		self.flags = flags  # 깃발이 설치된 위치 집합

	def get_state_matrix(self, include_mines=False):
		"""현재 게임 상태를 행렬로 변환"""
		state = np.zeros((self.height, self.width), dtype=np.int8)

		# 열린 셀 표시
		for row, col in self.opened_cells:
			if self.board[row][col] == '0':
				state[row][col] = BOARD_REPRESENTATION['empty']
			else:
				state[row][col] = int(self.board[row][col])

		# 깃발 표시
		for row, col in self.flags:
			state[row][col] = BOARD_REPRESENTATION['flag']

		# 학습을 위해 지뢰 위치 포함 (실제 게임에서는 사용하지 않음)
		if include_mines:
			for row in range(self.height):
				for col in range(self.width):
					if self.board[row][col] == 'X' and (row, col) not in self.opened_cells and (row,
					                                                                            col) not in self.flags:
						state[row][col] = BOARD_REPRESENTATION['mine']

		return state

	def get_valid_actions(self):
		"""현재 상태에서 가능한 행동 목록 반환"""
		valid_actions = []

		for row in range(self.height):
			for col in range(self.width):
				# 이미 열린 셀은 코드 클릭만 가능
				if (row, col) in self.opened_cells:
					# 숫자가 있는 셀만 코드 클릭 가능
					if self.board[row][col] not in ['0', 'X']:
						valid_actions.append((ACTION_CHORD_CLICK, row, col))
				else:
					# 닫힌 셀은 왼쪽/오른쪽 클릭 가능
					if (row, col) not in self.flags:
						valid_actions.append((ACTION_LEFT_CLICK, row, col))
					valid_actions.append((ACTION_RIGHT_CLICK, row, col))

		return valid_actions


class DQNAgent:
	"""Deep Q-Network 기반 지뢰찾기 AI 에이전트"""

	def __init__(self, width, height, memory_size=10000, batch_size=64, gamma=0.95, architecture='cnn'):
		self.width = width
		self.height = height
		self.state_shape = (height, width, 1)  # CNN 입력용 형태
		self.action_size = 3  # 왼쪽 클릭, 오른쪽 클릭, 코드 클릭

		# 하이퍼파라미터
		self.memory = deque(maxlen=memory_size)
		self.batch_size = batch_size
		self.gamma = gamma  # 할인율
		self.epsilon = 1.0  # 탐험률
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.architecture = architecture  # 신경망 아키텍처 선택

		# 모델 생성
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_target_model()

	def _build_model(self):
		"""신경망 아키텍처에 따른 Q-네트워크 모델 생성"""
		if self.architecture == 'cnn':
			return self._build_cnn_model()
		elif self.architecture == 'resnet':
			return self._build_resnet_model()
		elif self.architecture == 'densenet':
			return self._build_densenet_model()
		else:
			print(f"Unknown architecture: {self.architecture}, using CNN instead")
			return self._build_cnn_model()

	def _build_cnn_model(self):
		"""기본 CNN 아키텍처"""
		model = models.Sequential()

		# 컨볼루션 레이어
		model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
		                        input_shape=self.state_shape))
		model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
		model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))

		# 완전 연결 레이어
		model.add(layers.Flatten())
		model.add(layers.Dense(256, activation='relu'))
		model.add(layers.Dense(self.width * self.height * self.action_size))
		model.add(layers.Reshape((self.height, self.width, self.action_size)))

		model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
		return model

	def _build_resnet_model(self):
		"""ResNet 아키텍처 (잔차 연결 포함)"""
		inputs = layers.Input(shape=self.state_shape)

		# 초기 컨볼루션 레이어
		x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

		# 잔차 블록 1
		residual = x
		x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
		x = layers.Conv2D(64, (3, 3), padding='same')(x)
		x = layers.add([x, residual])
		x = layers.Activation('relu')(x)

		# 잔차 블록 2
		residual = x
		x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
		x = layers.Conv2D(128, (3, 3), padding='same')(x)
		residual = layers.Conv2D(128, (1, 1), padding='same')(residual)  # 차원 맞추기
		x = layers.add([x, residual])
		x = layers.Activation('relu')(x)

		# 완전 연결 레이어
		x = layers.Flatten()(x)
		x = layers.Dense(256, activation='relu')(x)
		x = layers.Dense(self.width * self.height * self.action_size)(x)
		outputs = layers.Reshape((self.height, self.width, self.action_size))(x)

		model = models.Model(inputs=inputs, outputs=outputs)
		model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
		return model

	def _build_densenet_model(self):
		"""DenseNet 아키텍처 (밀집 연결 포함)"""
		inputs = layers.Input(shape=self.state_shape)

		# 초기 컨볼루션 레이어
		x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

		# 밀집 블록 1
		dense1_outputs = [x]
		for _ in range(3):
			x = layers.Concatenate()(dense1_outputs)
			x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
			dense1_outputs.append(x)

		# 전환 레이어
		x = layers.Concatenate()(dense1_outputs)
		x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)

		# 밀집 블록 2
		dense2_outputs = [x]
		for _ in range(3):
			x = layers.Concatenate()(dense2_outputs)
			x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
			dense2_outputs.append(x)

		# 완전 연결 레이어
		x = layers.Concatenate()(dense2_outputs)
		x = layers.Flatten()(x)
		x = layers.Dense(256, activation='relu')(x)
		x = layers.Dense(self.width * self.height * self.action_size)(x)
		outputs = layers.Reshape((self.height, self.width, self.action_size))(x)

		model = models.Model(inputs=inputs, outputs=outputs)
		model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
		return model

	def update_target_model(self):
		"""타겟 모델 업데이트"""
		self.target_model.set_weights(self.model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		"""경험 저장"""
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state, valid_actions):
		"""현재 상태에서 행동 선택 (GPU 가속 최적화)"""
		if np.random.rand() <= self.epsilon:
			# 무작위 행동 선택
			return random.choice(valid_actions)

		# 모델을 사용하여 행동 선택
		state_tensor = np.expand_dims(state, axis=0)  # 배치 차원 추가
		state_tensor = np.expand_dims(state_tensor, axis=-1)  # 채널 차원 추가

		# DirectML GPU 가속 사용 시도
		try:
			# DirectML은 자동으로 적절한 디바이스를 선택합니다
			q_values = self.model.predict(state_tensor, verbose=0)[0]
		except Exception as e:
			print(f"예측 중 오류 발생: {e}")
			# 오류 발생 시 기본 CPU 연산으로 시도
			with tf.device('/CPU:0'):
				q_values = self.model.predict(state_tensor, verbose=0)[0]

		# 유효한 행동 중에서 Q값이 가장 높은 행동 선택
		best_action = None
		best_q_value = float('-inf')

		for action_type, row, col in valid_actions:
			if q_values[row, col, action_type] > best_q_value:
				best_q_value = q_values[row, col, action_type]
				best_action = (action_type, row, col)

		return best_action

	def replay(self):
		"""경험 리플레이를 통한 학습 (배치 처리 최적화)"""
		if len(self.memory) < self.batch_size:
			return

		# 배치 학습을 위한 준비
		minibatch = random.sample(self.memory, self.batch_size)

		# 배치 데이터 준비
		states = []
		targets = []

		# 배치 내 모든 다음 상태에 대한 예측을 한 번에 수행
		next_states = []
		next_states_indices = []
		done_indices = []

		# 첫 번째 패스: 상태 및 다음 상태 데이터 수집
		for i, (state, action, reward, next_state, done) in enumerate(minibatch):
			# 상태 텐서 준비
			state_tensor = np.expand_dims(state, axis=-1)  # 채널 차원 추가
			states.append(state_tensor)

			# 완료되지 않은 상태에 대해 다음 상태 수집
			if not done:
				next_state_tensor = np.expand_dims(next_state, axis=-1)
				next_states.append(next_state_tensor)
				next_states_indices.append(i)
			else:
				done_indices.append(i)

		# 배치로 변환
		states = np.array(states)

		# 모든 상태에 대한 예측을 한 번에 수행
		try:
			# DirectML은 자동으로 적절한 디바이스를 선택합니다
			predictions = self.model.predict(states, batch_size=self.batch_size, verbose=0)
		except Exception as e:
			print(f"배치 예측 중 오류 발생: {e}")
			# 오류 발생 시 기본 CPU 연산으로 시도
			with tf.device('/CPU:0'):
				predictions = self.model.predict(states, batch_size=self.batch_size, verbose=0)

		# 다음 상태가 있는 경우 타겟 모델로 예측
		if next_states:
			next_states = np.array(next_states)
			try:
				# DirectML은 자동으로 적절한 디바이스를 선택합니다
				next_state_predictions = self.target_model.predict(next_states, batch_size=len(next_states), verbose=0)
			except Exception as e:
				print(f"다음 상태 예측 중 오류 발생: {e}")
				# 오류 발생 시 기본 CPU 연산으로 시도
				with tf.device('/CPU:0'):
					next_state_predictions = self.target_model.predict(next_states, batch_size=len(next_states),
					                                                   verbose=0)

		# 두 번째 패스: 타겟 값 계산
		for i, (state, action, reward, next_state, done) in enumerate(minibatch):
			action_type, row, col = action
			target = predictions[i].copy()

			if done:
				target[row, col, action_type] = reward
			else:
				# 해당 인덱스의 다음 상태 예측 찾기
				next_state_idx = next_states_indices.index(i)
				next_state_pred = next_state_predictions[next_state_idx]
				target[row, col, action_type] = reward + self.gamma * np.amax(next_state_pred)

			targets.append(target)

		# 배치로 변환
		targets = np.array(targets)

		# 배치 학습 수행 (한 번의 fit 호출로 모든 데이터 학습)
		try:
			# DirectML은 자동으로 적절한 디바이스를 선택합니다
			self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=0)
		except Exception as e:
			print(f"모델 학습 중 오류 발생: {e}")
			# 오류 발생 시 기본 CPU 연산으로 시도
			with tf.device('/CPU:0'):
				self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=0)

		# 탐험률 감소
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		"""모델 로드"""
		if os.path.exists(name):
			self.model.load_weights(name)
			self.update_target_model()

	def save(self, name):
		"""모델 저장"""
		self.model.save_weights(name)


class MinesweeperAI:
	"""지뢰찾기 게임을 플레이하는 AI 클래스"""

	def __init__(self, width=9, height=9, mines=10, use_inference=True, architecture='cnn'):
		self.width = width
		self.height = height
		self.mines = mines
		self.architecture = architecture
		self.agent = DQNAgent(width, height, architecture=architecture)
		self.model_path = f"minesweeper_model_{width}x{height}_{mines}_{architecture}.weights.h5"

		# 확률적 추론 사용 여부
		self.use_inference = use_inference
		if use_inference:
			self.inference = ProbabilisticInference(width, height)

		# 모델 로드 시도
		self.agent.load(self.model_path)

	def train(self, episodes=1000, visualize=False, visualize_every=50, self_play=False):
		"""AI 에이전트 학습"""
		# 학습 결과 기록
		scores = []
		win_count = 0

		for episode in range(episodes):
			# 게임 환경 초기화
			if visualize and episode % visualize_every == 0:
				# 시각화 모드
				self._train_episode_visualized(episode)
			elif self_play and episode % 5 == 0:  # 5 에피소드마다 자가 대전
				# 자가 대전 모드
				self._train_episode_self_play()
			else:
				# 빠른 학습 모드
				score = self._train_episode_fast()
				scores.append(score)

				if score > 0:  # 승리
					win_count += 1

			# 학습 진행 상황 출력
			if (episode + 1) % 10 == 0:
				win_rate = win_count / 10 if episode >= 9 else win_count / (episode + 1)
				print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Epsilon: {self.agent.epsilon:.4f}")
				win_count = 0

			# 주기적으로 타겟 모델 업데이트 및 모델 저장
			if (episode + 1) % 100 == 0:
				self.agent.update_target_model()
				self.agent.save(self.model_path)

		# 최종 모델 저장
		self.agent.save(self.model_path)
		return scores

	def _train_episode_self_play(self):
		"""자가 대전을 통한 학습 (CPU/GPU 최적화)"""
		# 게임 환경 초기화
		game = self._create_headless_game()

		# 게임 상태 초기화
		state = MinesweeperState(
			game.board,
			set(),
			set(),
			self.width,
			self.height
		)

		done = False
		moves = []  # 게임 중 수행한 모든 행동 기록
		states = []  # 게임 중 모든 상태 기록
		step_count = 0  # 스텝 카운터 추가

		# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
		row = random.randint(0, self.height - 1)
		col = random.randint(0, self.width - 1)

		# 첫 번째 클릭 실행
		self._perform_action(game, ACTION_LEFT_CLICK, row, col)

		# 게임 상태 업데이트
		state = MinesweeperState(
			game.board,
			game.opened_cells,
			{(r, c) for r in range(self.height) for c in range(self.width)
			 if game.buttons[r][c].cget('text') == '🚩'},
			self.width,
			self.height
		)

		# 게임 루프
		while not done:
			# 현재 상태 행렬 얻기
			state_matrix = state.get_state_matrix()
			states.append(state_matrix.copy())

			# 유효한 행동 목록 얻기
			valid_actions = state.get_valid_actions()

			if not valid_actions:
				break

			# 하이브리드 행동 선택 (확률적 추론 + DQN)
			action = self.hybrid_act(state_matrix, valid_actions, game.board)
			action_type, row, col = action
			moves.append(action)
			step_count += 1

			# 행동 실행
			reward, done = self._perform_action(game, action_type, row, col)

			# 다음 상태 얻기
			next_state = MinesweeperState(
				game.board,
				game.opened_cells,
				{(r, c) for r in range(self.height) for c in range(self.width)
				 if game.buttons[r][c].cget('text') == '🚩'},
				self.width,
				self.height
			)

			# 상태 업데이트
			state = next_state

		# 게임 결과에 따른 보상 계산
		final_reward = 10 if len(game.opened_cells) == (self.width * self.height - self.mines) else -10

		# 자가 대전 학습: 게임 결과를 바탕으로 모든 행동 평가
		# 경험 메모리에 저장만 하고 학습은 한 번만 수행 (CPU 부하 감소)
		for i in range(len(moves)):
			# 현재 상태와 행동
			state_matrix = states[i]
			action = moves[i]
			action_type, row, col = action

			# 다음 상태 (마지막 행동인 경우 현재 상태 사용)
			next_state_matrix = states[i + 1] if i < len(moves) - 1 else state_matrix

			# 보상 계산: 마지막 행동에는 최종 보상, 나머지는 작은 보상
			if i == len(moves) - 1:
				reward = final_reward
			else:
				reward = 0.1  # 중간 행동에 대한 작은 보상

			# 경험 저장
			self.agent.remember(state_matrix, action, reward, next_state_matrix, i == len(moves) - 1)

		# 배치 학습 (한 번만 수행)
		try:
			# DirectML은 자동으로 적절한 디바이스를 선택합니다
			self.agent.replay()
		except Exception as e:
			print(f"자가 대전 학습 중 오류 발생: {e}")
			# 오류 발생 시 기본 CPU 연산으로 시도
			with tf.device('/CPU:0'):
				self.agent.replay()

	def hybrid_act(self, state_matrix, valid_actions, board=None, exploration=True):
		"""확률적 추론과 딥러닝을 결합한 행동 선택"""
		if not self.use_inference:
			# 확률적 추론을 사용하지 않는 경우 DQN 에이전트만 사용
			return self.agent.act(state_matrix, valid_actions)

		# 확률적 추론으로 안전한 이동과 지뢰 위치 찾기
		safe_moves, mine_cells, probability_map = self.inference.get_safe_moves(state_matrix, board)

		# 안전한 이동이 있으면 그 중 하나 선택
		if safe_moves:
			return random.choice(safe_moves)

		# 지뢰가 확실한 위치가 있으면 깃발 설치
		if mine_cells:
			return random.choice(mine_cells)

		# 확률적 추론으로 결정할 수 없는 경우 DQN 에이전트 사용
		if exploration:
			return self.agent.act(state_matrix, valid_actions)
		else:
			# 탐험 없이 최적의 행동 선택 (게임 플레이 모드)
			original_epsilon = self.agent.epsilon
			self.agent.epsilon = 0
			action = self.agent.act(state_matrix, valid_actions)
			self.agent.epsilon = original_epsilon
			return action

	def _train_episode_fast(self):
		"""시각화 없이 빠르게 한 에피소드 학습 (CPU/GPU 최적화)"""
		# 게임 환경 초기화
		game = self._create_headless_game()

		# 게임 상태 초기화
		state = MinesweeperState(
			game.board,
			set(),
			set(),
			self.width,
			self.height
		)

		done = False
		score = 0
		step_count = 0  # 스텝 카운터 추가
		update_frequency = 4  # 4스텝마다 학습 수행 (CPU 부하 감소)

		# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
		row = random.randint(0, self.height - 1)
		col = random.randint(0, self.width - 1)

		# 첫 번째 클릭 실행
		self._perform_action(game, ACTION_LEFT_CLICK, row, col)

		# 게임 상태 업데이트
		state = MinesweeperState(
			game.board,
			game.opened_cells,
			{(r, c) for r in range(self.height) for c in range(self.width)
			 if game.buttons[r][c].cget('text') == '🚩'},
			self.width,
			self.height
		)

		# 게임 루프
		while not done:
			# 현재 상태 행렬 얻기
			state_matrix = state.get_state_matrix()

			# 유효한 행동 목록 얻기
			valid_actions = state.get_valid_actions()

			if not valid_actions:
				break

			# 하이브리드 행동 선택 (확률적 추론 + DQN)
			action = self.hybrid_act(state_matrix, valid_actions, game.board)
			action_type, row, col = action

			# 행동 실행
			reward, done = self._perform_action(game, action_type, row, col)
			score += reward
			step_count += 1

			# 다음 상태 얻기
			next_state = MinesweeperState(
				game.board,
				game.opened_cells,
				{(r, c) for r in range(self.height) for c in range(self.width)
				 if game.buttons[r][c].cget('text') == '🚩'},
				self.width,
				self.height
			)
			next_state_matrix = next_state.get_state_matrix()

			# 경험 저장
			self.agent.remember(state_matrix, action, reward, next_state_matrix, done)

			# 상태 업데이트
			state = next_state

			# 일정 주기마다 또는 에피소드 종료 시 학습 수행 (CPU 부하 감소)
			if step_count % update_frequency == 0 or done:
				self.agent.replay()

		return score

	def _train_episode_visualized(self, episode):
		"""시각화와 함께 한 에피소드 학습 (CPU/GPU 최적화)"""
		# tkinter 루트 생성
		root = tk.Tk()
		root.title(f"지뢰찾기 AI 학습 - 에피소드 {episode + 1}")

		# 게임 생성
		game = MinesweeperGame(root, width=self.width, height=self.height, mines=self.mines)

		# 게임 상태 초기화
		state = MinesweeperState(
			game.board,
			game.opened_cells,
			{(r, c) for r in range(self.height) for c in range(self.width)
			 if game.buttons[r][c].cget('text') == '🚩'},
			self.width,
			self.height
		)

		done = False
		score = 0
		step_count = 0  # 스텝 카운터 추가
		update_frequency = 4  # 4스텝마다 학습 수행 (CPU 부하 감소)

		# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
		row = random.randint(0, self.height - 1)
		col = random.randint(0, self.width - 1)

		# 첫 번째 클릭 실행 (시각화)
		def first_click():
			self._perform_action_visualized(game, ACTION_LEFT_CLICK, row, col)
			root.after(100, game_loop)

		# 게임 루프 함수
		def game_loop():
			nonlocal state, done, score, step_count

			# 게임이 재시작되었는지 확인 (game_over가 False로 변경됨)
			if hasattr(game, 'was_game_over') and game.was_game_over and not game.game_over:
				# 게임이 재시작됨
				print("게임이 재시작되었습니다.")
				game.was_game_over = False

				# 상태 초기화
				state = MinesweeperState(
					game.board,
					game.opened_cells,
					{(r, c) for r in range(self.height) for c in range(self.width)
					 if game.buttons[r][c].cget('text') == '🚩'},
					self.width,
					self.height
				)

				# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
				row = random.randint(0, self.height - 1)
				col = random.randint(0, self.width - 1)

				# 첫 번째 클릭 실행
				self._perform_action_visualized(game, ACTION_LEFT_CLICK, row, col)

				# 다음 행동 예약
				root.after(300, game_loop)
				return

			if done or game.game_over:
				# 게임 종료 처리
				if game.game_over and not done:
					# 승리 여부에 따른 보상
					reward = 10 if len(game.opened_cells) == (self.width * self.height - self.mines) else -10
					score += reward
					done = True

					# 게임 오버 상태 기록 (재시작 감지용)
					game.was_game_over = True

					# 학습 수행 (에피소드 종료 시)
					self.agent.replay()

					# 일정 시간 후 다시 게임 루프 실행 (자동 재시작 확인)
					root.after(1000, game_loop)
					return

				# 에피소드 종료 (창 닫기)
				root.after(1000, root.destroy)
				return

			# 현재 상태 행렬 얻기
			state_matrix = state.get_state_matrix()

			# 유효한 행동 목록 얻기
			valid_actions = state.get_valid_actions()

			if not valid_actions:
				# 학습 수행 (에피소드 종료 시)
				self.agent.replay()
				root.after(1000, root.destroy)
				return

			# 하이브리드 행동 선택 (확률적 추론 + DQN)
			action = self.hybrid_act(state_matrix, valid_actions, game.board)
			action_type, row, col = action

			# 행동 실행 (시각화)
			reward, done = self._perform_action_visualized(game, action_type, row, col)
			score += reward
			step_count += 1

			# 다음 상태 얻기
			next_state = MinesweeperState(
				game.board,
				game.opened_cells,
				{(r, c) for r in range(self.height) for c in range(self.width)
				 if game.buttons[r][c].cget('text') == '🚩'},
				self.width,
				self.height
			)
			next_state_matrix = next_state.get_state_matrix()

			# 경험 저장
			self.agent.remember(state_matrix, action, reward, next_state_matrix, done)

			# 상태 업데이트
			state = next_state

			# 일정 주기마다 학습 수행 (CPU 부하 감소)
			if step_count % update_frequency == 0:
				self.agent.replay()

			# 다음 행동 예약 (지연 추가)
			root.after(300, game_loop)

		# 첫 번째 클릭 예약
		root.after(500, first_click)

		# tkinter 메인 루프 실행
		root.mainloop()

		return score

	def _create_headless_game(self):
		"""UI 없이 게임 로직만 실행하는 게임 객체 생성"""
		# 임시 tkinter 루트 생성
		root = tk.Tk()
		root.withdraw()  # UI 숨기기

		# 자동 재시작 게임 생성
		game = AutoRestartMinesweeperGame(root, width=self.width, height=self.height, mines=self.mines)

		return game

	def _perform_action(self, game, action_type, row, col):
		"""행동 실행 및 보상 계산 (헤드리스 모드)"""
		reward = 0
		done = False

		# 행동 실행 전 열린 셀 수
		opened_before = len(game.opened_cells)

		# 행동 타입에 따라 다른 메서드 호출
		if action_type == ACTION_LEFT_CLICK:
			game.handle_left_click(row, col)
		elif action_type == ACTION_RIGHT_CLICK:
			game.right_click(row, col)
		elif action_type == ACTION_CHORD_CLICK:
			game.chord_click(row, col)

		# 행동 실행 후 열린 셀 수
		opened_after = len(game.opened_cells)

		# 보상 계산
		if game.game_over:
			# 게임 종료
			if len(game.opened_cells) == (self.width * self.height - self.mines):
				# 승리
				reward = 10
			else:
				# 패배 (지뢰 밟음)
				reward = -10
			done = True
		else:
			# 새로 열린 셀 수에 비례하는 보상
			cells_opened = opened_after - opened_before
			reward = cells_opened * 0.1

			# 깃발 관련 보상
			if action_type == ACTION_RIGHT_CLICK:
				# 깃발을 설치한 경우
				if game.buttons[row][col].cget('text') == '🚩':
					# 실제 지뢰 위치에 깃발을 설치하면 작은 보상
					if game.board[row][col] == 'X':
						reward = 0.2
					else:
						# 지뢰가 아닌 곳에 깃발을 설치하면 작은 패널티
						reward = -0.1

		return reward, done

	def _perform_action_visualized(self, game, action_type, row, col):
		"""행동 실행 및 보상 계산 (시각화 모드)"""
		# 행동 실행 전 열린 셀 수
		opened_before = len(game.opened_cells)

		# 행동 타입에 따라 다른 메서드 호출
		if action_type == ACTION_LEFT_CLICK:
			# 왼쪽 클릭 이벤트 생성
			event = tk.Event()
			event.widget = game.buttons[row][col]
			game.handle_left_click(row, col)
		elif action_type == ACTION_RIGHT_CLICK:
			# 오른쪽 클릭 이벤트 생성
			event = tk.Event()
			event.widget = game.buttons[row][col]
			game.right_click(row, col)
		elif action_type == ACTION_CHORD_CLICK:
			# 코드 클릭 실행
			game.chord_click(row, col)

		# 행동 실행 후 열린 셀 수
		opened_after = len(game.opened_cells)

		# 보상 계산
		reward = 0
		done = False

		if game.game_over:
			# 게임 종료
			if len(game.opened_cells) == (self.width * self.height - self.mines):
				# 승리
				reward = 10
			else:
				# 패배 (지뢰 밟음)
				reward = -10
			done = True
		else:
			# 새로 열린 셀 수에 비례하는 보상
			cells_opened = opened_after - opened_before
			reward = cells_opened * 0.1

			# 깃발 관련 보상
			if action_type == ACTION_RIGHT_CLICK:
				# 깃발을 설치한 경우
				if game.buttons[row][col].cget('text') == '🚩':
					# 실제 지뢰 위치에 깃발을 설치하면 작은 보상
					if game.board[row][col] == 'X':
						reward = 0.2
					else:
						# 지뢰가 아닌 곳에 깃발을 설치하면 작은 패널티
						reward = -0.1

		return reward, done

	def play_game(self):
		"""학습된 AI로 게임 플레이 (시각화)"""
		# tkinter 루트 생성
		root = tk.Tk()
		root.title("지뢰찾기 AI 플레이")

		# 게임 생성
		game = MinesweeperGame(root, width=self.width, height=self.height, mines=self.mines)

		# 게임 상태 초기화
		state = MinesweeperState(
			game.board,
			game.opened_cells,
			{(r, c) for r in range(self.height) for c in range(self.width)
			 if game.buttons[r][c].cget('text') == '🚩'},
			self.width,
			self.height
		)

		# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
		row = random.randint(0, self.height - 1)
		col = random.randint(0, self.width - 1)

		# 첫 번째 클릭 실행 (시각화)
		def first_click():
			self._perform_action_visualized(game, ACTION_LEFT_CLICK, row, col)
			root.after(500, game_loop)

		# 게임 루프 함수
		def game_loop():
			nonlocal state

			# 게임이 재시작되었는지 확인 (game_over가 False로 변경됨)
			if hasattr(game, 'was_game_over') and game.was_game_over and not game.game_over:
				# 게임이 재시작됨
				print("게임이 재시작되었습니다.")
				game.was_game_over = False

				# 상태 초기화
				state = MinesweeperState(
					game.board,
					game.opened_cells,
					{(r, c) for r in range(self.height) for c in range(self.width)
					 if game.buttons[r][c].cget('text') == '🚩'},
					self.width,
					self.height
				)

				# 첫 번째 행동은 항상 무작위 위치에 왼쪽 클릭 (첫 클릭은 항상 안전)
				row = random.randint(0, self.height - 1)
				col = random.randint(0, self.width - 1)

				# 첫 번째 클릭 실행
				self._perform_action_visualized(game, ACTION_LEFT_CLICK, row, col)

				# 다음 행동 예약
				root.after(500, game_loop)
				return

			if game.game_over:
				# 게임 종료 메시지
				result = "승리" if len(game.opened_cells) == (self.width * self.height - self.mines) else "패배"
				print(f"게임 종료: {result}")

				# 게임 오버 상태 기록 (재시작 감지용)
				game.was_game_over = True

				# 일정 시간 후 다시 게임 루프 실행 (자동 재시작 확인)
				root.after(1000, game_loop)
				return

			# 현재 상태 행렬 얻기
			state_matrix = state.get_state_matrix()

			# 유효한 행동 목록 얻기
			valid_actions = state.get_valid_actions()

			if not valid_actions:
				return

			# 하이브리드 행동 선택 (확률적 추론 + DQN, 탐험 없이)
			action = self.hybrid_act(state_matrix, valid_actions, game.board, exploration=False)
			action_type, row, col = action

			# 행동 실행 (시각화)
			self._perform_action_visualized(game, action_type, row, col)

			# 다음 상태 얻기
			next_state = MinesweeperState(
				game.board,
				game.opened_cells,
				{(r, c) for r in range(self.height) for c in range(self.width)
				 if game.buttons[r][c].cget('text') == '🚩'},
				self.width,
				self.height
			)

			# 상태 업데이트
			state = next_state

			# 다음 행동 예약 (지연 추가)
			root.after(500, game_loop)

		# 첫 번째 클릭 예약
		root.after(1000, first_click)

		# tkinter 메인 루프 실행
		root.mainloop()


# 메인 함수
if __name__ == "__main__":
	print("=" * 50)
	print("지뢰찾기 AI - 딥러닝 기반 자동 플레이")
	print("=" * 50)

	# 고급 기능 사용 여부
	use_advanced = input("\n고급 기능 사용? (y/n, 기본: n): ").lower() == 'y'

	if use_advanced:
		print("\n고급 기능 메뉴:")
		print("1. 단일 에이전트 (고급 기능 포함)")
		print("2. 메타 학습 에이전트 (여러 난이도 학습)")

		advanced_mode = input("\n선택 (기본: 1): ") or "1"

		if advanced_mode == "2":
			# 메타 학습 에이전트 사용
			meta_agent = MetaLearningAgent()

			print("\n신경망 아키텍처 선택:")
			print("1. CNN (기본)")
			print("2. ResNet")
			print("3. DenseNet")

			arch_choice = input("\n선택 (기본: 1): ") or "1"
			architecture = {
				"1": "cnn",
				"2": "resnet",
				"3": "densenet"
			}.get(arch_choice, "cnn")

			# 확률적 추론 사용 여부
			use_inference = input("\n확률적 추론 사용? (y/n, 기본: y): ").lower() != 'n'

			# 에이전트 초기화
			meta_agent.initialize_agents(use_inference=use_inference, architecture=architecture)

			print("\n메타 학습 모드:")
			print("1. 모든 난이도 학습")
			print("2. 학습된 모델로 게임 플레이")

			meta_mode = input("\n선택 (기본: 1): ") or "1"

			if meta_mode == "1":
				# 학습 에피소드 수 설정
				episodes = int(input("\n난이도별 학습 에피소드 수 (기본: 500): ") or "500")

				# 시각화 여부 설정
				visualize = input("학습 과정 시각화? (y/n, 기본: n): ").lower() == 'y'

				# 자가 대전 사용 여부
				self_play = input("자가 대전 사용? (y/n, 기본: y): ").lower() != 'n'

				# 학습 시작
				print("\n메타 학습 시작")
				meta_agent.train_all(episodes_per_config=episodes, visualize=visualize, self_play=self_play)
				print("\n메타 학습 완료")

				# 학습 후 게임 플레이
				play_after_train = input("\n학습 후 게임 플레이? (y/n, 기본: y): ").lower() != 'n'
				if play_after_train:
					# 난이도 선택
					print("\n난이도를 선택하세요:")
					print("1. 초급 (9x9, 지뢰 10개)")
					print("2. 중급 (16x16, 지뢰 40개)")
					print("3. 고급 (30x16, 지뢰 99개)")
					print("4. 커스텀 설정")

					difficulty = input("\n선택 (기본: 1): ") or "1"

					# 난이도에 따른 설정
					if difficulty == "1":
						width, height, mines = 9, 9, 10
					elif difficulty == "2":
						width, height, mines = 16, 16, 40
					elif difficulty == "3":
						width, height, mines = 30, 16, 99
					elif difficulty == "4":
						# 커스텀 설정
						width = int(input("가로 크기 (8-30): ") or "9")
						height = int(input("세로 크기 (8-24): ") or "9")
						max_mines = (width * height) - 9  # 최소 9칸은 비워둠
						mines = int(input(f"지뢰 수 (1-{max_mines}): ") or "10")
					else:
						width, height, mines = 9, 9, 10

					meta_agent.play_game(width, height, mines)
			else:
				# 게임 플레이
				# 난이도 선택
				print("\n난이도를 선택하세요:")
				print("1. 초급 (9x9, 지뢰 10개)")
				print("2. 중급 (16x16, 지뢰 40개)")
				print("3. 고급 (30x16, 지뢰 99개)")
				print("4. 커스텀 설정")

				difficulty = input("\n선택 (기본: 1): ") or "1"

				# 난이도에 따른 설정
				if difficulty == "1":
					width, height, mines = 9, 9, 10
				elif difficulty == "2":
					width, height, mines = 16, 16, 40
				elif difficulty == "3":
					width, height, mines = 30, 16, 99
				elif difficulty == "4":
					# 커스텀 설정
					width = int(input("가로 크기 (8-30): ") or "9")
					height = int(input("세로 크기 (8-24): ") or "9")
					max_mines = (width * height) - 9  # 최소 9칸은 비워둠
					mines = int(input(f"지뢰 수 (1-{max_mines}): ") or "10")
				else:
					width, height, mines = 9, 9, 10

				meta_agent.play_game(width, height, mines)
		else:
			# 단일 에이전트 고급 기능
			# 난이도 선택
			print("\n난이도를 선택하세요:")
			print("1. 초급 (9x9, 지뢰 10개)")
			print("2. 중급 (16x16, 지뢰 40개)")
			print("3. 고급 (30x16, 지뢰 99개)")
			print("4. 커스텀 설정")

			difficulty = input("\n선택 (기본: 1): ") or "1"

			# 난이도에 따른 설정
			if difficulty == "1":
				width, height, mines = 9, 9, 10
			elif difficulty == "2":
				width, height, mines = 16, 16, 40
			elif difficulty == "3":
				width, height, mines = 30, 16, 99
			elif difficulty == "4":
				# 커스텀 설정
				width = int(input("가로 크기 (8-30): ") or "9")
				height = int(input("세로 크기 (8-24): ") or "9")
				max_mines = (width * height) - 9  # 최소 9칸은 비워둠
				mines = int(input(f"지뢰 수 (1-{max_mines}): ") or "10")
			else:
				width, height, mines = 9, 9, 10

			print("\n신경망 아키텍처 선택:")
			print("1. CNN (기본)")
			print("2. ResNet")
			print("3. DenseNet")

			arch_choice = input("\n선택 (기본: 1): ") or "1"
			architecture = {
				"1": "cnn",
				"2": "resnet",
				"3": "densenet"
			}.get(arch_choice, "cnn")

			# 확률적 추론 사용 여부
			use_inference = input("\n확률적 추론 사용? (y/n, 기본: y): ").lower() != 'n'

			# 자가 대전 사용 여부
			self_play = input("\n자가 대전 사용? (y/n, 기본: y): ").lower() != 'n'

			# AI 에이전트 생성
			ai = MinesweeperAI(width, height, mines, use_inference=use_inference, architecture=architecture)

			# 학습 모드 또는 플레이 모드 선택
			mode = input("\n모드 선택 (1: 학습, 2: 플레이): ")

			if mode == "1":
				# 학습 에피소드 수 설정
				episodes = int(input("\n학습 에피소드 수 (기본: 1000): ") or "1000")

				# 시각화 여부 설정
				visualize = input("학습 과정 시각화? (y/n, 기본: n): ").lower() == 'y'

				# 학습 시작
				print(f"\n학습 시작: {episodes} 에피소드")
				ai.train(episodes=episodes, visualize=visualize, self_play=self_play)
				print("\n학습 완료")

				# 학습 후 게임 플레이
				play_after_train = input("\n학습 후 게임 플레이? (y/n, 기본: y): ").lower() != 'n'
				if play_after_train:
					ai.play_game()
			else:
				# 학습된 모델로 게임 플레이
				ai.play_game()
	else:
		# 기본 기능 (이전 버전과 동일)
		# 난이도 설정
		width, height, mines = 9, 9, 10  # 초급 난이도

		# AI 에이전트 생성
		ai = MinesweeperAI(width, height, mines)

		# 학습 모드 또는 플레이 모드 선택
		mode = input("\n모드 선택 (1: 학습, 2: 플레이): ")

		if mode == "1":
			# 학습 에피소드 수 설정
			episodes = int(input("\n학습 에피소드 수 (기본: 1000): ") or "1000")

			# 시각화 여부 설정
			visualize = input("학습 과정 시각화? (y/n, 기본: n): ").lower() == 'y'

			# 학습 시작
			print(f"\n학습 시작: {episodes} 에피소드")
			ai.train(episodes=episodes, visualize=visualize)
			print("\n학습 완료")

			# 학습 후 게임 플레이
			play_after_train = input("\n학습 후 게임 플레이? (y/n, 기본: y): ").lower() != 'n'
			if play_after_train:
				ai.play_game()
		else:
			# 학습된 모델로 게임 플레이
			ai.play_game()
