import random
import socket
import threading
import time

import numpy as np
import torch.nn
from experiments.conf import IP_DICT, epsilon, device
from torch.utils.data import DataLoader, TensorDataset
from utils.communication import save_result, calculate_result, request_handshake, accept_handshake, send_data, \
    recv_data


# 生成公私钥
# public_key, private_key = paillier.generate_paillier_keypair()


# def send_data(client, msg):
#     # msg: [[]]
#     # msg = grad.tolist()
#     # en_msg = [encryption(i, n) for i in msg]
#     length = len(str(msg).encode('utf8'))
#     client.send(str(length).encode('utf8'))
#     # client.recv(3)
#     while True:
#         try:
#             by = client.recv(3)
#             if len(by) == 0:
#                 continue
#             break
#         except socket.error:
#             time.sleep(0.1)
#     client.send(str(msg).encode('utf8'))
#
#
# def recv_data(client):
#     client.setblocking(0)
#     while True:
#         try:
#             by = client.recv(1024)
#             if len(by) == 0:
#                 continue
#             break
#         except socket.error:
#             time.sleep(0.1)
#     length = int(by.decode('utf8'))
#     client.send('ok\n'.encode('utf8'))
#     while True:
#         try:
#             by = client.recv(1024)
#             while len(by) < length:
#                 by += client.recv(1024)
#             break
#         except:
#             time.sleep(0.1)
#     msg = by.decode('utf8')[1:-1] + ','
#     # print('received data!')
#     data = []
#     for l in msg.split('],')[:-1]:
#         l = l.strip()[1:]
#         d = []
#         for i in l.split(','):
#             d.append(float(i))
#         data.append(d)
#     # print('return data')
#     return data
#
#
# def accept_connect(tcp_socket, message_handle, flag):
#     while flag:
#         client, _ = tcp_socket.accept()  # 阻塞，等待客户端连接
#         # 给每个客户端创建一个独立的线程进行管理
#         thread = threading.Thread(target=message_handle, args=(client,))
#         # 设置成守护线程
#         thread.setDaemon(True)
#         thread.start()


# 联邦学习参与者
class Participant:
    def __init__(self, name, id, party_num, train_data_loader, test_data_loader, local_model, loss, batch_size=100):
        self.name = name
        self.id = id
        self.batch_size = batch_size
        self.party_num = party_num
        self.host = IP_DICT[id][0]
        self.port = IP_DICT[id][1]
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_training = True
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if party_num > 1:
            self.tcp_socket.bind((self.host, self.port))
            self.tcp_socket.listen(5)
        self.local_model = local_model
        self.cost = loss
        self.party_list = [i for i in range(self.party_num)]
        self.party_list.remove(self.id)
        self.result_dict = dict()
        self.gradient_dict = dict()
        self.cnt = self.party_num - 1
        self.time_counter = 0
        self.thread_list = []
        self.epsilon = epsilon
        self.time_list = []
        self.tp_list = []
        self.tn_list = []
        self.fp_list = []
        self.fn_list = []
        self.mse_list = []
        self.var_list = []
        self.loss_list = []
        self.acc_list = []

    def set_model(self, local_model):
        self.local_model = local_model

    def save_result(self, path='result.csv'):
        save_result(path, self.tp_list, self.tn_list, self.fp_list, self.fn_list, self.time_list, self.mse_list,
                    self.var_list, self.loss_list, self.acc_list)

    def finish_training(self):
        self.is_training = False

    def message_handle(self, client):
        idx, _, _ = accept_handshake(client)
        if not self.result_dict.__contains__(idx):
            self.result_dict[idx] = []
            self.gradient_dict[idx] = []
        data = recv_data(client)
        # print('handshake done')
        if self.result_dict.__contains__(idx):
            self.result_dict[idx].append(torch.Tensor(data).to(device))
        else:
            self.result_dict[idx] = [torch.Tensor(data).to(device)]
        while len(self.gradient_dict[idx]) == 0:
            time.sleep(0.1)
        send_data(client, self.gradient_dict[idx][0])
        try:
            del self.gradient_dict[idx][0]
        except:
            pass

    def accept_connect(self):
        while self.is_training:
            client, _ = self.tcp_socket.accept()
            thread = threading.Thread(target=self.message_handle, args=(client,))
            thread.setDaemon(True)
            thread.start()
            self.thread_list.append(thread)

    def get_noise(self, shape):
        rand = torch.tensor(np.random.normal(loc=0, scale=1 / self.epsilon, size=shape)).to(device)
        return rand

    def train(self, epoch=0):
        sum_loss = 0.0
        train_correct = 0
        # 设置随机数，同步各个参与者的数据集
        torch.manual_seed(epoch)
        start = time.time()
        self.local_model.train()
        for i, data in enumerate(self.train_data_loader, 1):
            inputs, labels = data
            inputs.to(device)
            # 计算结果，并将结果保存
            outputs = self.local_model(inputs)
            # 判断是否有标签信息（id为0的参与者保存有标签信息），根据情况计算要发送给其他参与者的数据
            label_owner = i % self.party_num
            if self.id == label_owner:
                while True:
                    cnt = 1
                    for k in self.result_dict.keys():
                        if len(self.result_dict[k]) > 0:
                            cnt += 1
                    if cnt == self.party_num:
                        break
                    time.sleep(0.1)
                # add noise
                self.get_noise(outputs.shape)
                if self.party_num > 1:
                    outputs += self.get_noise(outputs.shape)
                for k in self.result_dict.keys():
                    outputs += self.result_dict[k][0]
                    del self.result_dict[k][0]
                # output = outputs.clone().detach().requires_grad_(True)
                # outputs = self.dense_model(outputs)
                # loss = self.cost(outputs, labels)
                # U = outputs
                outputs.to(device)
                labels = labels.clone().detach().to(device)
                class_loss = self.cost(outputs, labels)
                grads = torch.autograd.grad(outputs=class_loss, inputs=outputs)[0]
                for k in self.gradient_dict.keys():
                    self.gradient_dict[k].append(grads.tolist())
                self.local_model.backward(inputs, grads)
            else:
                sock = request_handshake(IP_DICT[label_owner][0], IP_DICT[label_owner][1], self.id, i)
                send_data(sock, (outputs + self.get_noise(outputs.shape)).tolist())
                # if self.party_num > 1:
                #     send_data(sock, (outputs + self.get_noise(outputs.shape)).tolist())
                # else:
                #     send_data(sock, outputs.tolist())
                grad = recv_data(sock)
                # msg = bytes.decode('utf8')
                grad = torch.Tensor(grad).float().to(device)
                x = inputs.clone().detach().float().to(device)
                self.local_model.backward(x, grad)
            # self.optimizer.step()
        self.time_counter += time.time() - start
        if self.id == 0:
            print(self.time_counter)
        # 测试，用明文
        self.local_model.eval()
        id_list, label_list = [], []
        batch_idx = 0
        for i, data in enumerate(self.test_data_loader, 1):
            inputs, labels = data
            inputs.to(device)
            outputs = self.local_model(inputs)
            # outputs = self.dense_model(outputs)
            label_owner = 0
            if self.id == label_owner:
                # 如果拥有标签信息，则接收其它参与者的数据
                while True:
                    cnt = 1
                    for k in self.result_dict.keys():
                        if len(self.result_dict[k]) > 0:
                            cnt += 1
                    if cnt == self.party_num:
                        break
                    time.sleep(0.1)
                for k in self.result_dict.keys():
                    outputs += self.result_dict[k][0]
                    del self.result_dict[k][0]
                outputs.to(device)
                labels = labels.clone().detach().to(device)
                class_loss = self.cost(outputs, labels)
                grads = torch.autograd.grad(outputs=class_loss, inputs=outputs)[0]
                for k in self.gradient_dict.keys():
                    self.gradient_dict[k].append(grads.tolist())
            # 如果没有标签信息，则将数据发送给拥有标签的参与者，用于计算梯度
            else:
                sock = request_handshake(IP_DICT[label_owner][0], IP_DICT[label_owner][1], self.id, i)
                send_data(sock, outputs.tolist())
                recv_data(sock)
            outputs.to(device)
            labels = labels.clone().detach().to(device)
            loss = self.cost(outputs, labels)
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
            id_list.extend(id.tolist())
            label_list.extend(labels.tolist())
            batch_idx += 1
        tp, tn, fp, fn, mse, var, acc = calculate_result(id_list, label_list)
        self.tp_list.append(tp)
        self.tn_list.append(tn)
        self.fp_list.append(fp)
        self.fn_list.append(fn)
        self.mse_list.append(mse)
        self.var_list.append(var)
        self.time_list.append(self.time_counter)
        self.loss_list.append(float(sum_loss) / len(self.test_data_loader.dataset))
        self.acc_list.append(float(100 * train_correct / len(self.test_data_loader.dataset)))
        print('epoch: %d loss: %.03f' % (epoch, sum_loss / len(self.test_data_loader.dataset)))
        print('correct:%.03f%%' % (100 * train_correct / len(self.test_data_loader.dataset)))


if __name__ == '__main__':
    p = Participant('1', 1, 3, [], [], [], None, )
    thread = threading.Thread(target=accept_connect, args=(p.tcp_socket, p.message_handle, p.is_training))
    thread.setDaemon(True)
    thread.start()
2
