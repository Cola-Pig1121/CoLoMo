class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss
