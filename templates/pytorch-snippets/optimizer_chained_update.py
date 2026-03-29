import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(4):
    print(epoch, scheduler2.get_last_lr()[0])
    optimizer.step()
    scheduler1.step()
    scheduler2.step()

# 从1.4版本开始，torch.optim.lr_scheduler 支持链式更新（chaining），即用户可以定义两个 schedulers，并交替在训练中使用。
