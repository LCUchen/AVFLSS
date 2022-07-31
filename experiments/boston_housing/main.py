import os
import sys
import argparse
import threading
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from model import LocalModel
import numpy as np
# from experiments.client import Participant
from experiments.he_client import Participant, Server
from experiments.asycn_client import Participant as Async_Participant
from experiments.client import Participant as Sync_Participant
from experiments.dp_client import Participant as DP_Participant
from dataset.boston_housing.load_boston_data import vertical_partition_data, make_data
from torch.utils.data import DataLoader, TensorDataset
from experiments.conf import dataset_config, max_epoch, home, device


def run_experiment(party, mode):
    epoch = 0
    while epoch < max_epoch:
        party.train(epoch)
        epoch += 1
    party.save_result(home + 'result/boston-%d-%d-%d' % (party.id, party.party_num, mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--epoch', type=int, help='epoch', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    # to do
    parser.add_argument('--batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('--train_rate', type=float, help='training data', default=.7)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--party_num', type=int, default=3)
    parser.add_argument('--name', type=str, default='a')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--label', type=bool, default=True)
    parser.add_argument('--mode', type=int, help='0: Synchronous mode; 1: Homomorphic Encryption mode; 2: '
                                                 'Asynchronous mode; 3: dp mode', default=2)
    opt = parser.parse_args()
    # server
    if opt.id == -1:
        server = Server('server', opt.id, opt.party_num)
        thread = threading.Thread(target=server.accept_connect)
        thread.setDaemon(True)
        thread.start()
        while True:
            server.train()
            time.sleep(0.1)
    else:
        train_data, train_label, test_data, test_label = make_data()
        train_data = vertical_partition_data(train_data, opt.party_num)[opt.id]
        test_data = vertical_partition_data(test_data, opt.party_num)[opt.id]
        local_model = LocalModel(input_dim=len(train_data[0])).to(device)
        train_data, train_label = torch.tensor(train_data).float(), torch.tensor(train_label).float()
        train_data = TensorDataset(train_data, train_label)
        train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        # test_images, test_labels = load_mnist_data('/home/chen/Code/', 't10k')
        test_data, test_label = torch.tensor(test_data).float(), torch.tensor(test_label).float()
        # test_images, test_labels = torch.from_numpy(self.test_x).float(), torch.from_numpy(self.test_y).long()
        test_data = TensorDataset(test_data, test_label)
        test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
        # 所有参与者都有标签，但只有部分参与者有使用标签的权限，因此可以看成只有部分参与者拥有标签
        if opt.mode == 0:
            party = Sync_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                     local_model, dataset_config['boston_housing']['loss_function'], opt.batch_size)
        elif opt.mode == 1:
            party = Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                local_model, dataset_config['boston_housing']['loss_function'], opt.batch_size)
        elif opt.mode == 2:
            party = Async_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                      local_model, dataset_config['boston_housing']['loss_function'], opt.batch_size)
        else:
            party = Async_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                   local_model, dataset_config['boston_housing']['loss_function'], opt.batch_size, opt.mode)
        thread = threading.Thread(target=party.accept_connect)
        thread.setDaemon(True)
        thread.start()
        run_experiment(party, opt.mode)
