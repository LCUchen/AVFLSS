import math
from experiments.conf import device
import torch


class LocalModel(torch.nn.Module):
    def __init__(self, input_dim, learning_rate=0.01, bias=True):
        super(LocalModel, self).__init__()
        self.size = int(math.sqrt(input_dim // 3))
        # self.classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(input_dim),
        #                                       torch.nn.Linear(input_dim, input_dim, bias=bias),
        #                                       torch.nn.Softmax(dim=1),
        #                                       torch.nn.BatchNorm1d(input_dim),
        #                                       torch.nn.Linear(input_dim, input_dim // 2),
        #                                       torch.nn.BatchNorm1d(input_dim // 2),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(input_dim // 2, input_dim // 4),
        #                                       torch.nn.BatchNorm1d(input_dim // 4),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(input_dim // 4, input_dim // 8),
        #                                       torch.nn.BatchNorm1d(input_dim // 8),
        #                                       torch.nn.ReLU(),
        #                                       torch.nn.Linear(input_dim // 8, 10))
        self.classifier = torch.nn.Sequential(
            # IN : 3*32*32
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.ReLU(),
            # IN : 96*16*16
            torch.nn.MaxPool2d(kernel_size=2, stride=2),              # 论文中为kernel_size = 3,stride = 2
            # IN : 96*8*8
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            torch.nn.ReLU(),
            # IN :256*8*8
            torch.nn.MaxPool2d(kernel_size=2, stride=2),              # 论文中为kernel_size = 3,stride = 2
            # IN : 256*4*4
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.ReLU(),
            # IN : 384*4*4
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.ReLU(),
            # IN : 384*4*4
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.ReLU(),
            # IN : 384*4*4
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),              # 论文中为kernel_size = 3,stride = 2
            # # OUT : 384*2*2
            torch.nn.Flatten(),
            # torch.nn.Linear(384 * 4 * 4, 4096),
            torch.nn.Linear(1536, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 10)
        )
        # self.linear = torch.nn.Sequential(
        #     nn.Linear(in_features=384 * 2 * 2, out_features=4096),
        #     nn.ReLU(),
        #     nn.Linear(in_features=4096, out_features=4096),
        #     nn.ReLU(),
        #     nn.Linear(in_features=4096, out_features=10),
        # )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        self.is_debug = False
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")
        # x = torch.tensor(x).float()
        x = x.reshape(x.shape[0], 3, self.size, self.size).to(device)
        # x = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))(x)
        # x = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))(x)
        # x = torch.nn.MaxPool2d(2, 2)(x)
        # x = torch.nn.Flatten()(x)
        return self.classifier(x)

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        x = x.clone().detach().requires_grad_(True).float().cuda()
        x = x.reshape(x.shape[0], 3, self.size, self.size)
        grads = grads.clone().detach().float()
        output = self.classifier(x)
        output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()