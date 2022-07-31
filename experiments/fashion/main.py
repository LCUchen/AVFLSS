import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import torch
from model import LocalModel
import numpy as np
# from experiments.client import Participant
from experiments.he_client import Participant, Server
from experiments.asycn_client import Participant as Async_Participant
from experiments.client import Participant as Sync_Participant
from experiments.dp_client import Participant as DP_Participant
from dataset.fashion.load_mnist_data import load_mnist_data
from torch.utils.data import DataLoader, TensorDataset
from experiments.conf import dataset_config, max_epoch, home, device


def vertical_partition_data(x, party_num):
    partitioned_data = []
    # split data by column
    # feature_num = int(len(x[0][0][0]) / party_num)
    # for i in range(party_num):
    #     tmp = []
    #     for d in x:
    #         row = []
    #         for j in d:
    #             col = []
    #             for k in j:
    #                 if i < party_num - 1:
    #                     col.append(k[i * feature_num:i * feature_num + feature_num])
    #                 else:
    #                     col.append(k[i * feature_num:])
    #             row.append(col)
    #         tmp.append(row)
    #     tmp = np.array(tmp)
    #     partitioned_data.append(tmp)
    # split data by row
    feature_num = int(len(x[0][0]) / party_num)
    for i in range(party_num):
        tmp = []
        for n in x:
            for c in n:
                row = []
                if i < party_num - 1:
                    for r in c[i * feature_num:i * feature_num + feature_num]:
                        row.append(r)
                else:
                    for r in c[i * feature_num:]:
                        row.append(r)
                tmp.append([row])
        tmp = np.array(tmp)
        partitioned_data.append(tmp)
    # split by row (gap)
    # feature_num = int(len(x[0][0]) / party_num)
    # for i in range(party_num):
    #     tmp = []
    #     for n in x:
    #         for c in n:
    #             row = []
    #             for r in range(len(c)):
    #                 if r % party_num == i:
    #                     row.append(c[r])
    #             tmp.append([row])
    #     tmp = np.array(tmp)
    #     partitioned_data.append(tmp)
    return partitioned_data


def run_experiment(party, mode):
    epoch = 0
    while epoch < max_epoch:
        party.train(epoch)
        epoch += 1
    party.save_result(home + 'result/mnist-%d-%d-%d' % (party.id, party.party_num, mode))
    if party.id == 0:
        time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--epoch', type=int, help='epoch', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('--train_rate', type=float, help='training data', default=.7)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--party_num', type=int, default=1)
    parser.add_argument('--name', type=str, default='a')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--label', type=bool, default=True)
    parser.add_argument('--mode', type=int, help='0: Synchronous mode; 1: Homomorphic Encryption mode; 2: '
                                                 'Asynchronous mode; 3: dp mode', default=0)
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
        train_data, train_label = load_mnist_data('../../dataset/fashion', 'train')
        train_data = vertical_partition_data(train_data, opt.party_num)[opt.id]
        test_data, test_label = load_mnist_data('../../dataset/fashion', 't10k')
        test_data = vertical_partition_data(test_data, opt.party_num)[opt.id]
        local_model = LocalModel(input_dim=len(train_data[0][0])).to(device)
        train_data, train_label = torch.tensor(train_data).float(), torch.tensor(train_label).long()
        train_data = TensorDataset(train_data, train_label)
        train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        test_data, test_label = torch.tensor(test_data).float(), torch.tensor(test_label).long()
        test_data = TensorDataset(test_data, test_label)
        test_data_loader = DataLoader(test_data, batch_size=len(test_label), shuffle=False)
        # 所有参与者都有标签，但只有部分参与者有使用标签的权限，因此可以看成只有部分参与者拥有标签
        if opt.mode == 0:
            party = Sync_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                     local_model, dataset_config['MNIST']['loss_function'], opt.batch_size)
        elif opt.mode == 1:
            party = Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                local_model, dataset_config['MNIST']['loss_function'], opt.batch_size)
        elif opt.mode == 2:
            party = Async_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader,
                                      local_model, dataset_config['MNIST']['loss_function'], opt.batch_size)
        else:
            party = Async_Participant(opt.name, opt.id, opt.party_num, train_data_loader, test_data_loader, local_model,
                                      dataset_config['credit_card']['loss_function'], opt.batch_size, opt.mode)
        thread = threading.Thread(target=party.accept_connect)
        thread.setDaemon(True)
        thread.start()
        run_experiment(party, opt.mode)
