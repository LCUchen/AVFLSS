import queue
import random
import socket
import threading
import time
import select
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
from experiments.secret_sharing import split_data_into_secret, recover_secret
from experiments.conf import IP_DICT, device, max_thread_count, epsilon, time_out
from utils.communication import request_handshake, accept_handshake
from utils.communication import save_result, calculate_result, send_data, recv_data
from experiments.dh import dh_exchange, get_key


def send_public_key(client, n):
    length = len(str(n).encode('utf8'))
    client.send(str(length).encode('utf8'))
    client.recv(1024)
    client.send(str(n).encode('utf8'))
    client.recv(1024)


def recv_public_key(client):
    by = client.recv(1024)
    while len(by) == 0:
        by = client.recv(1024)
    length = int(by.decode('utf8'))
    client.send('ok\n'.encode('utf8'))
    by = client.recv(1024)
    while len(by) < length:
        by += client.recv(1024)
    client.send('ok\n'.encode('utf8'))
    n = int(by.decode('utf8'))
    return n


# 联邦学习参与者（无中心服务器）
class Participant:
    def __init__(self, name, id, party_num, train_data_loader, test_data, local_model, loss, batch_size=100,
                 mode=2):
        self.thread_count = 0
        self.max_thread_count = max_thread_count
        self.name = name
        self.id = id
        self.batch_size = batch_size
        self.party_num = party_num
        self.host = IP_DICT[id][0]
        self.port = IP_DICT[id][1]
        self.train_data_loader = train_data_loader
        self.test_data = test_data
        self.pause = False
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置IP地址复用
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if party_num > 1:
            self.tcp_socket.bind((self.host, self.port))
            self.tcp_socket.listen(100)
            self.tcp_socket.setblocking(False)
        self.local_model = local_model
        self.cost = loss
        self.party_list = [i for i in range(self.party_num)]
        self.party_list.remove(self.id)
        self.result_dict = dict()
        self.gradient_dict = dict()
        self.noise_dict = dict()
        self.rdn_dict = dict()
        self.cnt = self.party_num - 1
        self.time_counter = 0
        self.public_key = 0
        self.thread_list = []
        self.mutex = threading.Lock()
        self.finished_num = 0
        self.loss_list = []
        self.time_list = []
        self.tp_list = []
        self.tn_list = []
        self.fp_list = []
        self.fn_list = []
        self.mse_list = []
        self.var_list = []
        self.acc_list = []
        self.mode = mode
        self.epoll = select.epoll()
        self.epoll.register(self.tcp_socket.fileno(), select.EPOLLIN)
        self.fd_to_socket = {self.tcp_socket.fileno(): self.tcp_socket, }
        self.message_queues = {}
        self.secret_dict = {}
        self.inputs_dict = {}

    def clear(self):
        self.noise_dict.clear()
        self.secret_dict.clear()
        self.rdn_dict.clear()
        self.result_dict.clear()
        self.gradient_dict.clear()
        self.inputs_dict.clear()

    def set_model(self, local_model):
        self.local_model = local_model

    def finish_training(self):
        self.pause = False

    def message_handle(self, client):
        # 接收来自其它参与者的请求，并将加密后的计算结果发送给请求方
        idx, index, cmd = accept_handshake(client)
        if cmd == 0:
            # aggregate result
            if not self.result_dict.__contains__(idx):
                self.result_dict[idx] = {}
                self.gradient_dict[idx] = {}
            # print('handshake', flush=True)
            data = recv_data(client)
            # print('handshake done', flush=True)
            if self.result_dict[idx].__contains__(index):
                self.result_dict[idx][index].append(torch.Tensor(data).to(device))
            else:
                self.result_dict[idx][index] = [torch.Tensor(data).to(device)]
            # print('wait gradient', flush=True)
            while not self.gradient_dict[idx].__contains__(index) or len(self.gradient_dict[idx][index]) == 0:
                time.sleep(0.01)
            # print('send grad', flush=True)
            send_data(client, self.gradient_dict[idx][index][0])
            # print('send grad done', flush=True)
            del self.gradient_dict[idx][index][0]
        elif cmd == 1:
            # get noise
            # print('received data from %d' %idx)
            if not self.noise_dict.__contains__(idx):
                self.noise_dict[idx] = {}
            rdn = random.randint(0, 10000)
            # print('send data')
            send_data(client, [[rdn]])
            # print('send data to %d done' %idx)
            data = int(recv_data(client)[0][0])
            # print('received data from %d' %idx)
            xa, ya, p = dh_exchange(seed=(data + rdn))
            yb = recv_data(client)[0][0]
            # print('received data from %d' %idx)
            send_data(client, [[xa]])
            # print('send data to %d done' %idx)
            # get_key(xa, yb, p)
            s = get_key(xa, yb, p)
            if self.noise_dict[idx].__contains__(index):
                self.noise_dict[idx][index].append(s)
            else:
                self.noise_dict[idx][index] = [s]

    def accept_connect(self):
        while True:
            client, _ = self.tcp_socket.accept()
            thread = threading.Thread(target=self.message_handle, args=(client,))
            thread.setDaemon(True)
            thread.start()
            self.thread_list.append(thread)

    def save_result(self, path='result.csv'):
        save_result(path, self.tp_list, self.tn_list, self.fp_list, self.fn_list, self.time_list, self.mse_list,
                    self.var_list, self.loss_list, self.acc_list)

    def get_noise(self, shape):
        rand = torch.tensor(np.random.normal(loc=0, scale=1 / epsilon, size=shape)).to(device)
        return rand

    def client_side(self, index, label_owner, inputs, outputs):
        for i in self.party_list:
            if self.id < i != label_owner:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((IP_DICT[i][0], IP_DICT[i][1]))
                    rdn = random.randint(0, 10000)
                    sock.sendall(json.dumps({'cmd': 'rdn', 'rdn': rdn, 'index': index, 'idx': self.id}).encode('utf8'))
                    rdx = int(json.loads(sock.recv(8192).decode('utf8'))['rdx'])
                    xa, ya, p = dh_exchange(seed=(rdx + rdn))
                    sock.sendall(json.dumps({'cmd': 'ya', 'ya': ya, 'seed': rdx + rdn, 'index': index, 'idx': self.id}).encode('utf8'))
                    yb = int(json.loads(sock.recv(8192).decode('utf8'))['yb'])
                    s = get_key(xa, yb, p)
                    sock.close()
                except:
                    s = self.id + index + i
                # add noise
                for i in range(outputs.shape[0]):
                    for j in range(outputs.shape[1]):
                        random.seed(s + i + j)
                        outputs[i][j] += random.uniform(0, 1)
        cnt = 0
        delay = {}
        while cnt > self.id:
            for i in self.party_list:
                if self.id > i != label_owner:
                    if i not in self.noise_dict or index not in self.noise_dict[i]:
                        time.sleep(0.01)
                        if i not in delay:
                            delay[i] = 1
                        else:
                            delay[i] += 1
                        if delay[i] < time_out:
                            continue
                        else:
                            cnt += 1
                            seed = self.id + index + i
                    else:
                        cnt += 1
                        seed = self.noise_dict[i][index]
                    for i in range(outputs.shape[0]):
                        for j in range(outputs.shape[1]):
                            random.seed(seed + i + j)
                            outputs[i][j] -= random.uniform(0, 1)
                elif i == label_owner:
                    cnt += 1
                elif self.id == i:
                    break
        # send result to server
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((IP_DICT[label_owner][0], IP_DICT[label_owner][1]))
            sock.sendall(json.dumps({'cmd': 'result', 'result': outputs.tolist(), 'index': index, 'idx': self.id}).encode('utf8'))
            # receive gradient
            grad = sock.recv(8192).decode('utf8')
            # msg = bytes.decode('utf8')
            grad = torch.Tensor(grad).float().to(device)
            x = inputs.clone().detach().float()
            # 更新模型
            self.mutex.acquire()
            self.local_model.backward(x, grad)
            self.finished_num += 1
            self.thread_count -= 1
            self.mutex.release()
            sock.close()
        except:
            # secret sharing
            secrets = split_data_into_secret(outputs.tolist(), self.party_num // 2, self.party_num - 2)
            for i in range(len(self.party_list)):
                if self.party_list[i] != label_owner:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        sock.connect((IP_DICT[self.party_list[i]][0], IP_DICT[self.party_list[i]][1]))
                        sock.sendall(json.dumps({'cmd': 'secret', 'secret': secrets[i], 'index': index, 'idx': self.id}).encode('utf8'))
                        sock.close()
                    except:
                        continue

    def server_side(self):
        while True:
            # print("等待活动连接......")
            # 轮询注册的事件集合，返回值为[(文件句柄，对应的事件)，(...),....]
            events = self.epoll.poll(time_out)
            if not events:
                # print("epoll超时无活动连接，重新轮询......")
                continue
            # print("有", len(events), "个新事件，开始处理......")

            for fd, event in events:
                socket = self.fd_to_socket[fd]
                # 如果活动socket为当前服务器socket，表示有新连接
                if socket == self.tcp_socket:
                    connection, address = self.tcp_socket.accept()
                    # print("新连接：", address)
                    # 新连接socket设置为非阻塞
                    connection.setblocking(False)
                    # 注册新连接fd到待读事件集合
                    self.epoll.register(connection.fileno(), select.EPOLLIN)
                    # 把新连接的文件句柄以及对象保存到字典
                    self.fd_to_socket[connection.fileno()] = connection
                    # 以新连接的对象为键值，值存储在队列中，保存每个连接的信息
                    self.message_queues[connection] = queue.Queue()
                # 关闭事件
                elif event & select.EPOLLHUP:
                    # print('client close')
                    # 在epoll中注销客户端的文件句柄
                    self.epoll.unregister(fd)
                    # 关闭客户端的文件句柄
                    self.fd_to_socket[fd].close()
                    # 在字典中删除与已关闭客户端相关的信息
                    del self.fd_to_socket[fd]
                # 可读事件
                elif event & select.EPOLLIN:
                    # 接收数据
                    data = socket.recv(16384).decode('utf8')
                    if data:
                        data = json.loads(data)
                        cmd = data['cmd']
                        if cmd == 'result':
                            print(data)
                            idx, index, result = data['idx'], data['index'], data['result']
                            if not self.result_dict.__contains__(idx):
                                self.result_dict[idx] = {}
                                self.gradient_dict[idx] = {}
                            if self.result_dict[idx].__contains__(index):
                                self.result_dict[idx][index].append(torch.Tensor(result).to(device))
                            else:
                                self.result_dict[idx][index] = [torch.Tensor(result).to(device)]
                        elif cmd == 'rdn':
                            rdx = random.randint(0, 10000)
                            self.message_queues[socket].put(json.dumps({'cmd': 'rdx', 'rdx': rdx}).encode('utf8'))
                            self.epoll.modify(fd, select.EPOLLOUT)
                        elif cmd == 'ya':
                            idx, index, seed, ya = data['idx'], data['index'], data['seed'], data['ya']
                            xb, yb, p = dh_exchange(seed=seed)
                            if idx not in self.noise_dict:
                                self.noise_dict[idx] = {}
                            s = get_key(xb, ya, p)
                            self.noise_dict[idx][index] = s
                            self.message_queues[socket].put(json.dumps({'cmd': 'yb', 'yb': yb}).encode('utf8'))
                            self.epoll.modify(fd, select.EPOLLOUT)
                        elif cmd == 'secret':
                            idx, index, secret = data['idx'], data['index'], data['secret']
                            if idx not in self.secret_dict:
                                self.secret_dict[idx] = {}
                            self.secret_dict[idx][index] = secret
                        elif cmd == 'gradient':
                            grad, idx, index = data['grad'], data['idx'], data['index']
                            grad = torch.Tensor(grad).float().to(device)
                            x = self.inputs_dict[index].clone().detach().float()
                            # 更新模型
                            self.mutex.acquire()
                            self.local_model.backward(x, grad)
                            self.finished_num += 1
                            self.thread_count -= 1
                            self.mutex.release()
                            del self.inputs_dict[index]
                            # print("收到数据：", data, "客户端：", socket.getpeername())
                        # 将数据放入对应客户端的字典
                        # self.message_queues[socket].put(data)
                        # 修改读取到消息的连接到等待写事件集合(即对应客户端收到消息后，再将其fd修改并加入写事件集合)
                        # self.epoll.modify(fd, select.EPOLLOUT)
                # 可写事件
                elif event & select.EPOLLOUT:
                    try:
                        # 从字典中获取对应客户端的信息
                        msg = self.message_queues[socket].get_nowait()
                    except queue.Empty:
                        print(socket.getpeername(), " queue empty")
                        # 修改文件句柄为读事件
                        self.epoll.modify(fd, select.EPOLLIN)
                    else:
                        # print("发送数据：", data, "客户端：", socket.getpeername())
                        # 发送数据
                        socket.send(msg)

    def calculate_grad(self, i, outputs, inputs, labels):
        # 如果拥有标签信息，则接收其它参与者的数据，并计算梯度
        while True:
            cnt = 1
            for k in self.result_dict.keys():
                if self.result_dict[k].__contains__(i) and len(self.result_dict[k][i]) > 0:
                    cnt += 1
            # if self.party_num > 0 and self.mode == 3:
            #     outputs += self.get_noise(outputs.shape)
            if cnt == self.party_num:
                for k in self.result_dict.keys():
                    outputs += self.result_dict[k][i][0]
                    del self.result_dict[k][i][0]
                outputs = outputs.to(device)
                labels = labels.clone().detach().to(device)
                class_loss = self.cost(outputs, labels)
                grads = torch.autograd.grad(outputs=class_loss, inputs=outputs)[0]
                for k in self.result_dict.keys():
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((IP_DICT[k][0], IP_DICT[k][1]))
                        for grd in self.gradient_dict[k]:
                            sock.sendall(json.dumps({'cmd': 'grad', 'grad': self.gradient_dict[k][grd], 'index': grd}).encode('utf8'))
                            del self.gradient_dict[k][grd]
                        sock.sendall(json.dumps({'cmd': 'grad', 'grad': grads.tolist(), 'index': i}).encode('utf8'))
                    except:
                        self.gradient_dict[k][i] = grads.tolist()
                # 更新模型
                self.mutex.acquire()
                self.local_model.backward(inputs, grads)
                self.finished_num += 1
                self.thread_count -= 1
                self.mutex.release()
                break
            else:
                time.sleep(0.01)

    def train(self, epoch=0):
        sum_loss = 0.0
        train_correct = 0
        # 设置随机数，同步各个参与者的数据集
        torch.manual_seed(epoch)
        time.sleep(5)
        # print('start', flush=True)
        start = time.time()
        self.finished_num = 1
        self.local_model.train()
        cnt = 0
        for i, data in enumerate(self.train_data_loader, 1):
            inputs, labels = data
            inputs.to(device)
            self.inputs_dict[i] = inputs
            # 计算结果，并将结果保存
            self.mutex.acquire()
            self.thread_count += 1
            self.mutex.release()
            # print('predict done', flush=True)
            while self.thread_count > self.max_thread_count:
                time.sleep(0.1)
            self.mutex.acquire()
            outputs = self.local_model(inputs)
            self.mutex.release()
            # TODO: 完善这一部分代码，加入秘密共享
            # secrets = split_data_into_secret(outputs.tolist(), self.party_num - 1, self.party_num)
            # 判断是否拥有标签
            label_owner = i % self.party_num
            if self.id == label_owner:
                t = threading.Thread(target=self.calculate_grad, args=(i, outputs, inputs, labels))
                t.start()
            # 如果没有标签信息，则将数据发送给拥有标签的参与者，用于计算梯度
            else:
                t = threading.Thread(target=self.client_side, args=(i, label_owner, inputs, outputs))
                t.start()
            cnt += 1
        while self.finished_num < len(self.train_data_loader):
            # print(self.finished_num, len(self.train_data_loader))
            time.sleep(0.1)
        self.time_counter += time.time() - start
        self.clear()
        print(cnt, self.time_counter, flush=True)
        # 测试，用明文
        self.local_model.eval()
        id_list, label_list = [], []
        batch_idx = 0
        for i in range(len(self.test_data['label'])):
            outputs = None
            for par in self.test_data['data']:
                inputs = torch.tensor(par[i]).float()
                inputs.to(device)
                if outputs is None:
                    outputs = self.local_model(inputs)
                else:
                    outputs += self.local_model(inputs)
            outputs.to(device)
            labels = self.test_data['label'].clone().detach().to(device)
            loss = self.cost(outputs, labels)
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
            id_list.extend(id.tolist())
            label_list.extend(labels.tolist())
            batch_idx += 1
        tp, tn, fp, fn, mse, var, acc = calculate_result(id_list, label_list)
        self.clear()
        self.tp_list.append(tp)
        self.tn_list.append(tn)
        self.fp_list.append(fp)
        self.fn_list.append(fn)
        self.mse_list.append(mse)
        self.var_list.append(var)
        self.time_list.append(self.time_counter)
        self.loss_list.append(sum_loss / len(self.test_data['label']))
        self.acc_list.append(acc)
        # print(len(self.test_data_loader.dataset))
        print('epoch: %d loss: %.03f' % (epoch, sum_loss / len(self.test_data['label'])))
        print('        correct:%.03f%%' % (100 * train_correct / len(self.test_data['label'])))
        # self.loss_list.append(sum_loss / len(self.test_data['label']))
        # if epoch == 100:
        #     plt.figure()
        #     plt.plot([i for i in range(len(self.loss_list))], self.loss_list)
        #     plt.show()


if __name__ == '__main__':
    p = Participant('1', 1, 3, [], [], [], [], None)
    thread = threading.Thread(target=p.accept_connect, args=(p.tcp_socket, p.message_handle, p.pause))
    thread.setDaemon(True)
    thread.start()
