import torch

tensor = torch.rand(2, 1024, 3).cuda()
print(tensor)
print(tensor.shape)

print(tensor ** 2)
print(tensor.sqrt())