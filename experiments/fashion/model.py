import torch
from experiments.conf import device


class LocalModel(torch.nn.Module):
    def __init__(self, input_dim, learning_rate=0.01, bias=True):
        super(LocalModel, self).__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(28 * input_dim, 100),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(100, 10))
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        self.is_debug = False
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")
        # x = torch.tensor(x).float()
        return self.classifier(x.to(device))

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        x = x.clone().detach().requires_grad_(True).float().to(device)
        grads = grads.clone().detach().float()
        output = self.classifier(x)
        output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

