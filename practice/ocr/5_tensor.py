import torch

# 0차원
scalar = torch.tensor(5)
print(scalar)

# 1차원
vector = torch.tensor([1, 2, 3])
print(vector)

# 2차원
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix)

# 3차원
tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor)