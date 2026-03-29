# If there is one global learning rate (which is the common case).
lr = next(iter(optimizer.param_groups))['lr']

# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
# 另一种方法，在一个batch训练代码里，当前的lr是optimizer.param_groups[0]['lr']
