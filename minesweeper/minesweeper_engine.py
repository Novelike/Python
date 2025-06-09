# minesweeper_engine.py
from PyQt6.QtCore import QObject, pyqtSignal
import random, time

class MinesweeperEngine(QObject):
	cell_revealed = pyqtSignal(int, int, str)   # (i, j, value)
	cell_flagged = pyqtSignal(int, int, bool)
	game_over = pyqtSignal(bool, int)           # (is_win, elapsed_sec)

	def __init__(self, width, height, mines):
		super().__init__()
		self.width, self.height, self.mines = width, height, mines
		self.reset()

	def reset(self):
		self.flags = 0
		self.opened = set()
		self.flagged_cells = set()
		self.game_over_flag = False
		self.start_time = None
		self.board = [['0']*self.width for _ in range(self.height)]
		m = 0
		while m < self.mines:
			x, y = random.randint(0, self.height-1), random.randint(0, self.width-1)
			if self.board[x][y] != 'X':
				self.board[x][y] = 'X'
				m += 1
		for i in range(self.height):
			for j in range(self.width):
				if self.board[i][j] != 'X':
					cnt = sum(
						self.board[r][c] == 'X'
						for r in range(max(0,i-1),min(self.height,i+2))
						for c in range(max(0,j-1),min(self.width,j+2))
					)
					self.board[i][j] = str(cnt)

	def reveal(self, i, j):
		if self.game_over_flag or (i, j) in self.opened or (i, j) in self.flagged_cells:
			return
		if not self.opened and self.start_time is None:
			self.start_time = time.time()
		if self.board[i][j] == 'X':
			self.cell_revealed.emit(i, j, '💣')
			self.game_over_flag = True
			self.game_over.emit(False, int(time.time()-self.start_time))
			return
		self._reveal_dfs(i, j)
		if self.check_win():
			self.game_over_flag = True
			self.game_over.emit(True, int(time.time()-self.start_time))

	def _reveal_dfs(self, i, j):
		if (i, j) in self.opened: return
		self.opened.add((i, j))
		val = self.board[i][j]
		self.cell_revealed.emit(i, j, val)
		if val == '0':
			for r in range(max(0,i-1),min(self.height,i+2)):
				for c in range(max(0,j-1),min(self.width,j+2)):
					if (r, c) != (i, j):
						self._reveal_dfs(r, c)

	def flag(self, i, j):
		if (i, j) in self.opened: return
		if (i, j) in self.flagged_cells:
			self.flagged_cells.remove((i, j))
			self.cell_flagged.emit(i, j, False)
		else:
			self.flagged_cells.add((i, j))
			self.cell_flagged.emit(i, j, True)

	def check_win(self):
		return len(self.opened) == self.width*self.height - self.mines
