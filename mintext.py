import torch

preds = torch.rand(2,2)
print(preds)

targets = torch.rand(2,2)
print(targets)


a = torch.min(preds[:,:2], targets[:,:2])

print(a)