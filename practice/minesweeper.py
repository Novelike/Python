# -*- coding: utf-8 -*-
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
import random
import time
import platform


# 플랫폼에 따른 기본 버튼 색상 설정
def get_default_button_color():
	"""플랫폼에 따라 적절한 기본 버튼 색상 반환"""
	if platform.system() == 'Windows':
		return 'SystemButtonFace'  # Windows 기본 버튼 색상
	else:
		return '#f0f0f0'  # Linux/macOS 등에서 사용할 기본 회색


class MinesweeperGame:
	def __init__(self, master, width=10, height=10, mines=15, main_menu_callback=None):
		self.master = master
		self.width = width
		self.height = height
		self.mines = mines
		self.game_over = False
		self.flags = 0
		self.start_time = None

		# 메인 메뉴로 돌아가는 콜백 함수 저장
		self.main_menu_callback = main_menu_callback

		# 열린 셀 추적을 위한 변수 추가
		self.opened_cells = set()

		# 현재 선택된 셀 위치 (키보드 조작용)
		self.current_row = 0
		self.current_col = 0

		# 버튼 크기 설정 (난이도에 따라 조정)
		self.button_width = 2  # 기본 버튼 너비
		self.button_height = 1  # 기본 버튼 높이

		# 난이도에 따라 버튼 크기 조정
		# if width > 20:  # 고급 난이도
		# 	self.button_width = 1
		# 	self.button_height = 1

		# 윈도우 제목 설정
		self.master.title(f"지뢰찾기 minesweeper - {width}x{height}, 지뢰 {mines}개")

		# 게임 상태 UI
		self.frame_top = tk.Frame(master)
		self.frame_top.pack(pady=5)

		self.label_mines = tk.Label(self.frame_top, text=f"남은 지뢰: {mines}")
		self.label_mines.pack(side=tk.LEFT, padx=20)

		self.label_time = tk.Label(self.frame_top, text="시간: 0")
		self.label_time.pack(side=tk.RIGHT, padx=20)

		# 게임 보드 UI
		self.frame_board = tk.Frame(master)
		self.frame_board.pack(pady=10)

		# 버튼 그리드 생성
		self.buttons = []
		for i in range(height):
			row = []
			for j in range(width):
				button = tk.Button(self.frame_board, width=self.button_width, height=self.button_height)
				button.grid(row=i, column=j)
				# 왼쪽 클릭과 오른쪽 클릭에 이벤트 바인딩
				button.bind("<Button-1>", lambda event, r=i, c=j: self.handle_left_click(r, c))
				button.bind("<Button-3>", lambda event, r=i, c=j: self.right_click(r, c))
				row.append(button)
			self.buttons.append(row)

		# 하단 버튼 프레임
		self.frame_bottom = tk.Frame(master)
		self.frame_bottom.pack(pady=10)

		# 재시작 버튼
		self.restart_button = tk.Button(self.frame_bottom, text="재시작", command=self.reset_game)
		self.restart_button.pack(side=tk.LEFT, padx=10)

		# 메인 화면 버튼
		self.main_menu_button = tk.Button(self.frame_bottom, text="메인 화면으로", command=self.return_to_main_menu)
		self.main_menu_button.pack(side=tk.LEFT, padx=10)

		# 게임 초기화
		self.reset_game()

		# 타이머 시작
		self.update_timer()

		# 키보드 이벤트 바인딩
		master.bind("<Up>", self.move_up)
		master.bind("<Down>", self.move_down)
		master.bind("<Left>", self.move_left)
		master.bind("<Right>", self.move_right)
		master.bind("<comma>", self.key_left_click)  # , 키로 왼쪽 클릭
		master.bind("<period>", self.key_right_click)  # . 키로 오른쪽 클릭
		master.bind("<Escape>", lambda event: self.return_to_main_menu())  # ESC 키로 메인 메뉴

		# 포커스 설정 (키보드 이벤트를 받기 위해)
		master.focus_set()

		# 초기 선택 셀 표시
		self.highlight_current_cell()

		# 창 크기 조정 (지연시켜 모든 위젯이 배치된 후 실행)
		self.master.after(100, self.adjust_window_size)

	def create_board(self):
		# 게임 보드 초기화
		self.board = []
		for i in range(self.height):
			self.board.append(['0'] * self.width)

		# 지뢰 배치
		mines_placed = 0
		while mines_placed < self.mines:
			x = random.randint(0, self.height - 1)
			y = random.randint(0, self.width - 1)
			if self.board[x][y] != 'X':
				self.board[x][y] = 'X'
				mines_placed += 1

		# 숫자 계산
		for i in range(self.height):
			for j in range(self.width):
				if self.board[i][j] != 'X':
					count = self.count_nearby_mines(i, j)
					self.board[i][j] = str(count)

	def count_nearby_mines(self, row, col):
		count = 0
		for i in range(max(0, row - 1), min(self.height, row + 2)):
			for j in range(max(0, col - 1), min(self.width, col + 2)):
				if self.board[i][j] == 'X':
					count += 1
		return count

	def handle_left_click(self, row, col):
		"""왼쪽 클릭 이벤트 처리"""
		print(f"왼쪽 클릭: ({row}, {col})")

		# 게임이 끝났으면 아무 동작 안함
		if self.game_over:
			return

		# 플래그가 있는 칸이면 아무 동작 안함
		if self.buttons[row][col].cget('text') == '🚩':
			return

		# 이미 열린 셀인지 확인
		if (row, col) in self.opened_cells:
			# 이미 열린 셀이면 chord_click 실행
			print(f"이미 열린 셀 클릭: ({row}, {col})")
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
			# messagebox.showinfo("게임 오버", "지뢰를 밟았습니다!")
			return

		# 빈 칸(0) 클릭 시 주변 빈 칸들 자동 열기
		self.reveal(row, col)

		# 승리 조건 확인
		self.check_win()

	def chord_click(self, row, col):
		"""이미 열린 셀을 클릭했을 때 주변 셀들을 자동으로 열기"""
		print(f"chord_click 실행: ({row}, {col})")

		# 숫자가 있는 셀만 처리
		cell_text = self.buttons[row][col].cget('text')
		if not cell_text or not cell_text.isdigit():
			print("숫자 셀이 아님")
			return

		cell_number = int(cell_text)
		print(f"셀 숫자: {cell_number}")

		# 주변 깃발 수 계산
		flag_count = 0
		for i in range(max(0, row - 1), min(self.height, row + 2)):
			for j in range(max(0, col - 1), min(self.width, col + 2)):
				if self.buttons[i][j].cget('text') == '🚩':
					flag_count += 1

		print(f"주변 깃발 수: {flag_count}")

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
			print("깃발 수 일치. 주변 셀 열기")
			for i, j in surrounding_cells:
				# 지뢰 클릭 시 게임 오버
				if self.board[i][j] == 'X':
					print("지뢰 발견! 게임 오버")
					self.buttons[i][j].config(text='💣', bg='red')
					self.game_over = True
					self.reveal_all()
					messagebox.showinfo("게임 오버", "지뢰를 밟았습니다!")
					return

				# 셀 열기
				self.reveal(i, j)

			# 승리 조건 확인
			self.check_win()
		else:
			print(f"깃발 수({flag_count})와 셀 숫자({cell_number}) 불일치")

	def reveal(self, row, col):
		# 범위를 벗어나거나 이미 열려있으면 무시
		if (row < 0 or row >= self.height or col < 0 or col >= self.width or
				(row, col) in self.opened_cells):
			return

		# 플래그가 있는 칸은 무시
		if self.buttons[row][col].cget('text') == '🚩':
			return

		# 현재 칸 열기
		self.opened_cells.add((row, col))
		self.buttons[row][col].config(relief=tk.SUNKEN, state='disabled')

		# 숫자 표시
		if self.board[row][col] != '0':
			self.buttons[row][col].config(text=self.board[row][col])
			# 숫자에 따라 색상 설정
			colors = ['blue', 'green', 'red', 'purple', 'maroon', 'turquoise', 'black', 'gray']
			if self.board[row][col].isdigit() and int(self.board[row][col]) > 0:
				self.buttons[row][col].config(fg=colors[int(self.board[row][col]) - 1])

			# 현재 선택된 셀이면 배경색 업데이트
			if row == self.current_row and col == self.current_col:
				self.buttons[row][col].config(bg='lightblue')

			return

		# 빈 칸이면 주변 칸도 열기
		self.buttons[row][col].config(text='')

		# 현재 선택된 셀이면 배경색 업데이트
		if row == self.current_row and col == self.current_col:
			self.buttons[row][col].config(bg='lightblue')

		# 주변 8개 칸 재귀적으로 열기
		for i in range(max(0, row - 1), min(self.height, row + 2)):
			for j in range(max(0, col - 1), min(self.width, col + 2)):
				if (i != row or j != col):
					self.reveal(i, j)

	def right_click(self, row, col):
		if self.game_over:
			return

		if self.buttons[row][col].cget('state') == 'disabled':
			return

		if self.buttons[row][col].cget('text') == '🚩':
			self.buttons[row][col].config(text='')
			self.flags -= 1
		else:
			self.buttons[row][col].config(text='🚩')
			self.flags += 1

		self.label_mines.config(text=f"남은 지뢰: {self.mines - self.flags}")

		# 승리 조건 확인
		self.check_win()

		return "break"  # 기본 동작 방지

	def reveal_all(self):
		for i in range(self.height):
			for j in range(self.width):
				if self.board[i][j] == 'X':
					if self.buttons[i][j].cget('text') != '🚩':
						self.buttons[i][j].config(text='💣')
				elif self.buttons[i][j].cget('text') == '🚩':
					self.buttons[i][j].config(text='❌', bg='orange')
				else:
					self.buttons[i][j].config(state='disabled')
					if self.board[i][j] != '0':
						self.buttons[i][j].config(text=self.board[i][j])

	def check_win(self):
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
			messagebox.showinfo("승리", f"축하합니다! 게임에서 이겼습니다!\n소요 시간: {elapsed_time}초")

	def update_timer(self):
		"""게임 타이머 업데이트"""
		if self.start_time is not None and not self.game_over:
			elapsed_time = int(time.time() - self.start_time)
			self.label_time.config(text=f"시간: {elapsed_time}")

		# after 메서드의 반환값을 저장하여 나중에 취소할 수 있도록 함
		self.timer_id = self.master.after(1000, self.update_timer)

	# 키보드 조작 관련 메서드들
	def move_up(self, event):
		"""위쪽 방향키: 선택 셀을 위로 이동"""
		if self.current_row > 0:
			self.unhighlight_current_cell()
			self.current_row -= 1
			self.highlight_current_cell()

	def move_down(self, event):
		"""아래쪽 방향키: 선택 셀을 아래로 이동"""
		if self.current_row < self.height - 1:
			self.unhighlight_current_cell()
			self.current_row += 1
			self.highlight_current_cell()

	def move_left(self, event):
		"""왼쪽 방향키: 선택 셀을 왼쪽으로 이동"""
		if self.current_col > 0:
			self.unhighlight_current_cell()
			self.current_col -= 1
			self.highlight_current_cell()

	def move_right(self, event):
		"""오른쪽 방향키: 선택 셀을 오른쪽으로 이동"""
		if self.current_col < self.width - 1:
			self.unhighlight_current_cell()
			self.current_col += 1
			self.highlight_current_cell()

	def key_left_click(self, event):
		"""쉼표(,) 키: 현재 선택된 셀에 왼쪽 클릭 동작"""
		self.handle_left_click(self.current_row, self.current_col)
		self.highlight_current_cell()  # 선택 셀 다시 강조

	def key_right_click(self, event):
		"""마침표(.) 키: 현재 선택된 셀에 오른쪽 클릭 동작"""
		self.right_click(self.current_row, self.current_col)
		self.highlight_current_cell()  # 선택 셀 다시 강조

	def highlight_current_cell(self):
		"""현재 선택된 셀 강조"""
		current_button = self.buttons[self.current_row][self.current_col]

		# 셀이 열려있는지 확인
		is_opened = (self.current_row, self.current_col) in self.opened_cells

		# 셀의 원래 배경색 저장
		if hasattr(self, 'original_bg'):
			self.original_bg = current_button.cget('bg')
		else:
			self.original_bg = get_default_button_color()

		# 열린 셀이면 더 밝은 색으로, 닫힌 셀이면 다른 색으로 강조
		if is_opened:
			current_button.config(bg='lightblue')
		else:
			current_button.config(bg='yellow')

	def unhighlight_current_cell(self):
		"""현재 선택된 셀의 강조 해제"""
		current_button = self.buttons[self.current_row][self.current_col]

		# 셀이 열려있는지 확인
		is_opened = (self.current_row, self.current_col) in self.opened_cells

		# 셀이 이미 열려있거나 게임 오버 상태인 경우 원래 배경색으로 복원
		if is_opened or self.game_over:
			current_button.config(bg=self.original_bg)
		else:
			# 닫힌 셀은 기본 배경색으로
			current_button.config(bg=get_default_button_color())

	def reset_game(self):
		# 게임 상태 초기화
		self.game_over = False
		self.flags = 0
		self.start_time = None
		self.opened_cells = set()  # 열린 셀 추적 집합 초기화
		self.label_mines.config(text=f"남은 지뢰: {self.mines}")
		self.label_time.config(text="시간: 0")

		# 현재 선택 셀 위치 초기화
		self.current_row = 0
		self.current_col = 0

		# 보드 초기화
		self.create_board()

		# 버튼 초기화
		for i in range(self.height):
			for j in range(self.width):
				self.buttons[i][j].config(text='', state='normal', relief=tk.RAISED, bg=get_default_button_color())

		# 현재 선택 셀 강조
		self.highlight_current_cell()

	def show_click_effect(self, cells):
		"""셀 리스트에 클릭 효과 적용"""
		# 원래 relief 상태 저장
		original_relief = {}

		for i, j in cells:
			original_relief[(i, j)] = self.buttons[i][j].cget('relief')
			# 누른 상태로 변경
			self.buttons[i][j].config(relief=tk.SUNKEN)

		# 화면 업데이트 (애니메이션 효과를 위해)
		self.master.update()

		# 약간의 지연 후 원래 상태로 복원
		self.master.after(100, lambda: self.restore_relief(cells, original_relief))

	def restore_relief(self, cells, original_relief):
		"""셀 리스트의 relief 상태 복원"""
		for i, j in cells:
			# 아직 열리지 않은 셀만 원래 상태로 복원
			if (i, j) not in self.opened_cells:
				self.buttons[i][j].config(relief=original_relief[(i, j)])

	def return_to_main_menu(self):
		"""메인 화면으로 돌아가기"""
		# 타이머 중지 (update_timer에서 사용하는 after 콜백 취소)
		if hasattr(self, 'timer_id'):
			self.master.after_cancel(self.timer_id)

		# 메인 메뉴 콜백이 제공된 경우 실행
		if self.main_menu_callback:
			self.main_menu_callback()
		else:
			# 콜백이 없는 경우의 기본 동작: 현재 창 닫기
			self.master.destroy()

	def adjust_window_size(self):
		"""게임 난이도에 맞게 창 크기 조정"""
		# 프레임의 필요한 크기를 계산
		self.master.update_idletasks()  # 위젯 크기 업데이트

		# 모든 프레임의 필요한 크기 계산
		top_height = self.frame_top.winfo_reqheight()
		board_width = self.frame_board.winfo_reqwidth()
		board_height = self.frame_board.winfo_reqheight()
		bottom_height = self.frame_bottom.winfo_reqheight()

		# 창 크기 계산 (여백 포함)
		window_width = board_width + 40  # 좌우 여백
		window_height = top_height + board_height + bottom_height + 60  # 상하 여백

		# 화면 크기 확인
		screen_width = self.master.winfo_screenwidth()
		screen_height = self.master.winfo_screenheight()

		# 화면 크기를 초과하지 않도록 조정
		window_width = min(window_width, screen_width - 100)
		window_height = min(window_height, screen_height - 100)

		# 창 크기 설정
		self.master.geometry(f"{window_width}x{window_height}")

		# 창 최소 크기 설정
		self.master.minsize(window_width, window_height)

		# 창을 화면 중앙에 배치
		x = (screen_width - window_width) // 2
		y = (screen_height - window_height) // 2
		self.master.geometry(f"+{x}+{y}")


class DifficultySelector:
	def __init__(self, master):
		self.master = master
		self.width = None
		self.height = None
		self.mines = None

		# 난이도 선택 화면
		self.frame = tk.Frame(master)
		self.frame.pack(pady=20)

		tk.Label(self.frame, text="지뢰찾기 난이도 선택", font=("Malgun Gothic", 16)).pack(pady=10)

		# 난이도 버튼
		tk.Button(self.frame, text="초급 (9x9, 10개 지뢰)",
		          command=lambda: self.set_difficulty(9, 9, 10)).pack(pady=5)

		tk.Button(self.frame, text="중급 (16x16, 40개 지뢰)",
		          command=lambda: self.set_difficulty(16, 16, 40)).pack(pady=5)

		tk.Button(self.frame, text="고급 (16x30, 99개 지뢰)",
		          command=lambda: self.set_difficulty(16, 30, 99)).pack(pady=5)

		# 커스텀 난이도
		custom_frame = tk.Frame(self.frame)
		custom_frame.pack(pady=10)

		tk.Label(custom_frame, text="가로:").grid(row=0, column=0)
		self.width_entry = tk.Entry(custom_frame, width=5)
		self.width_entry.grid(row=0, column=1)
		self.width_entry.insert(0, "10")

		tk.Label(custom_frame, text="세로:").grid(row=0, column=2)
		self.height_entry = tk.Entry(custom_frame, width=5)
		self.height_entry.grid(row=0, column=3)
		self.height_entry.insert(0, "10")

		tk.Label(custom_frame, text="지뢰:").grid(row=0, column=4)
		self.mines_entry = tk.Entry(custom_frame, width=5)
		self.mines_entry.grid(row=0, column=5)
		self.mines_entry.insert(0, "15")

		tk.Button(custom_frame, text="시작", command=self.custom_difficulty).grid(row=0, column=6, padx=5)

	def set_difficulty(self, width, height, mines):
		self.width = width
		self.height = height
		self.mines = mines
		self.frame.destroy()
		self.start_game()

	def custom_difficulty(self):
		try:
			width = int(self.width_entry.get())
			height = int(self.height_entry.get())
			mines = int(self.mines_entry.get())

			# 입력값 검증
			if width < 5 or height < 5 or width > 30 or height > 30:
				messagebox.showerror("오류", "가로와 세로는 5에서 30 사이여야 합니다.")
				return

			if mines < 1 or mines >= width * height:
				messagebox.showerror("오류", "지뢰 수는 1개 이상, 칸 수보다 적어야 합니다.")
				return

			self.set_difficulty(width, height, mines)
		except ValueError:
			messagebox.showerror("오류", "숫자를 입력해주세요.")

	def start_game(self):
		game = MinesweeperGame(self.master, self.width, self.height, self.mines)


class MainMenu:
	def __init__(self, master):
		self.master = master
		tfont = tkfont.Font(family="Nanum Gothic", size=12)
		self.master.title("지뢰찾기 게임")

		# 메인 프레임
		self.frame = tk.Frame(master)
		self.frame.pack(padx=20, pady=20)

		# 게임 제목
		title_label = tk.Label(self.frame, text="지뢰찾기", font=tfont)
		title_label.pack(pady=20)

		# 난이도 버튼
		self.difficulty_frame = tk.Frame(self.frame)
		self.difficulty_frame.pack(pady=10)

		# 초급
		beginner_button = tk.Button(self.difficulty_frame, text="초급", width=15,
		                            command=lambda: self.start_game(9, 9, 10), font=("Malgun Gothic", 9))
		beginner_button.pack(pady=5)

		# 중급
		intermediate_button = tk.Button(self.difficulty_frame, text="중급", width=15,
		                                command=lambda: self.start_game(16, 16, 40), font=("Malgun Gothic", 9))
		intermediate_button.pack(pady=5)

		# 고급
		expert_button = tk.Button(self.difficulty_frame, text="고급", width=15,
		                          command=lambda: self.start_game(30, 16, 99), font=("Malgun Gothic", 9))
		expert_button.pack(pady=5)

		# 커스텀 난이도
		custom_button = tk.Button(self.difficulty_frame, text="커스텀", width=15,
		                          command=self.show_custom_dialog, font=("Malgun Gothic", 9))
		custom_button.pack(pady=5)

		# 종료 버튼
		exit_button = tk.Button(self.frame, text="종료", width=15,
		                        command=master.quit, font=("Malgun Gothic", 9))
		exit_button.pack(pady=10)

		# 메인 메뉴 창 크기 조정
		self.adjust_window_size()

	def adjust_window_size(self):
		"""메인 메뉴 창 크기 조정"""
		# 프레임 크기 업데이트
		self.master.update_idletasks()

		# 필요한 크기 계산
		width = self.frame.winfo_reqwidth() + 40
		height = self.frame.winfo_reqheight() + 40

		# 창 크기 설정
		self.master.geometry(f"{width}x{height}")

		# 화면 중앙에 배치
		screen_width = self.master.winfo_screenwidth()
		screen_height = self.master.winfo_screenheight()
		x = (screen_width - width) // 2
		y = (screen_height - height) // 2
		self.master.geometry(f"+{x}+{y}")

	def start_game(self, width, height, mines):
		"""선택한 난이도로 게임 시작"""
		# 기존 프레임 제거
		self.frame.pack_forget()

		# 새로운 게임 시작
		self.game = MinesweeperGame(self.master, width=width, height=height, mines=mines,
		                            main_menu_callback=self.show_main_menu)

	def show_main_menu(self):
		"""메인 메뉴로 돌아가기"""
		# 게임 위젯 모두 제거
		for widget in self.master.winfo_children():
			widget.destroy()

		# 메인 메뉴 다시 생성
		self.__init__(self.master)

	def show_custom_dialog(self):
		"""커스텀 난이도 설정 대화상자 표시"""
		dialog = tk.Toplevel(self.master)
		dialog.title("커스텀 난이도")
		dialog.geometry("300x200")
		dialog.resizable(False, False)

		# 가운데 정렬
		dialog.transient(self.master)
		dialog.grab_set()

		# 프레임
		frame = tk.Frame(dialog, padx=10, pady=10)
		frame.pack(fill=tk.BOTH, expand=True)

		# 너비 입력
		tk.Label(frame, text="가로 크기 (8-30):").grid(row=0, column=0, sticky="w", pady=5)
		width_var = tk.StringVar(value="16")
		width_entry = tk.Entry(frame, textvariable=width_var, width=10)
		width_entry.grid(row=0, column=1, pady=5)

		# 높이 입력
		tk.Label(frame, text="세로 크기 (8-24):").grid(row=1, column=0, sticky="w", pady=5)
		height_var = tk.StringVar(value="16")
		height_entry = tk.Entry(frame, textvariable=height_var, width=10)
		height_entry.grid(row=1, column=1, pady=5)

		# 지뢰 수 입력
		tk.Label(frame, text="지뢰 수:").grid(row=2, column=0, sticky="w", pady=5)
		mines_var = tk.StringVar(value="40")
		mines_entry = tk.Entry(frame, textvariable=mines_var, width=10)
		mines_entry.grid(row=2, column=1, pady=5)

		def validate_and_start():
			try:
				width = int(width_var.get())
				height = int(height_var.get())
				mines = int(mines_var.get())

				# 유효성 검사
				if width < 8 or width > 30:
					messagebox.showerror("오류", "가로 크기는 8에서 30 사이여야 합니다.")
					return

				if height < 8 or height > 24:
					messagebox.showerror("오류", "세로 크기는 8에서 24 사이여야 합니다.")
					return

				max_mines = (width * height) - 9  # 최소 9칸은 비워둠
				if mines < 1 or mines > max_mines:
					messagebox.showerror("오류", f"지뢰 수는 1에서 {max_mines} 사이여야 합니다.")
					return

				dialog.destroy()
				self.start_game(width, height, mines)

			except ValueError:
				messagebox.showerror("오류", "모든 값은 숫자여야 합니다.")

		# 확인 버튼
		ok_button = tk.Button(frame, text="확인", command=validate_and_start)
		ok_button.grid(row=3, column=0, pady=10, padx=5)

		# 취소 버튼
		cancel_button = tk.Button(frame, text="취소", command=dialog.destroy)
		cancel_button.grid(row=3, column=1, pady=10, padx=5)

		# 초기 포커스 설정
		width_entry.focus_set()


def set_korean_font(root):
	# 리눅스에서 사용 가능한 한글 폰트 지정
	default_font = tkfont.nametofont("TkDefaultFont")
	text_font = tkfont.nametofont("TkTextFont")
	fixed_font = tkfont.nametofont("TkFixedFont")

	# 가능한 한글 폰트들을 시도
	korean_fonts = ['NanumGothic', 'Malgun Gothic', 'Gulim', 'Batang', 'AppleGothic', 'UnDotum']

	# 시스템에 설치된 폰트 확인
	available_fonts = tkfont.families()

	# 설치된 한글 폰트 중 하나 선택
	selected_font = None
	for font in korean_fonts:
		if font in available_fonts:
			selected_font = font
			break

	# 적합한 폰트가 없으면 기본 폰트 사용
	if not selected_font:
		return

	# 폰트 설정
	default_font.configure(family=selected_font)
	text_font.configure(family=selected_font)
	fixed_font.configure(family=selected_font)


# 게임 시작 코드
if __name__ == "__main__":
	root = tk.Tk()
	set_korean_font(root)
	# font = tkfont.Font(family="나눔 고딕", size=12)
	root.title("지뢰찾기: minesweeper(test)")

	# 창 크기를 고정하지 않음 (게임 난이도에 따라 조절될 수 있도록)
	root.resizable(True, True)

	# 메인 메뉴 시작
	menu = MainMenu(root)

	root.mainloop()
