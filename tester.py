import torch.nn as nn
import torch
import torch.nn.functional as F

t = torch.tensor([0])
a = torch.tensor([[-0.1399, -0.0398, -0.3521]])
arr = F.one_hot(t, num_classes=3)
output = arr * a
print(output)
print(torch.sum(output, dim=-1))