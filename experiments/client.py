import socket
import threading
import time
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from experiments.conf import IP_DICT
from utils.communication import recv_data, send_data, accept_handshake, request_handshake, save_result, \
    calculate_result


# def send_grad(client, grad):
#     msg = grad.tolist()
#     length = len(str(msg).encode('utf8'))
#     client.send(str(length).encode('utf8'))
#     client.recv(1024)
#     client.send(str(msg).encode('utf8'))
#
#
# def recv_grad(client):
#     by = client.recv(1024)
#     while len(by) == 0:
#         by = client.recv(1024)
#     length = int(by.decode('utf8'))
#     client.send('ok'.encode('utf8'))
#     by = client.recv(1024)
#     while len(by) < length:
#         by += client.recv(1024)
#     msg = by.decode('utf8')[1:-1] + ','
#     data = []
#     for l in msg.split('],')[:-1]:
#         l = l.strip()[1:]
#         d = []
#         for i in l.split(','):
#             d.append(float(i))
#         data.append(d)
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


class Participant:
    def __init__(self, name, id, party_num, train_data_loader, test_data_loader, local_model, loss, batch_size=100):
        self.name = name
        self.id = id
        self.party_num = party_num
        self.host = IP_DICT[self.id][0]
        self.port = IP_DICT[self.id][1]
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_training = True
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.bind((self.host, self.port))
        self.tcp_socket.listen(5)
        self.connect_pool = []
        self.local_model = local_model
        self.cost = loss
        self.result_dict = dict()
        self.gradient_dict = dict()
        self.cnt = self.party_num - 1
        self.time_counter = 0
        self.batch_size = batch_size

        # self.thread = threading.Thread(self.accept_connect())
        # self.thread.setDaemon(True)
        # self.thread.start()

    def set_model(self, local_model):
        self.local_model = local_model

    def finish_training(self):
        self.is_training = False

    def message_handle(self, client):
        # client.sendall("连接服务器成功!".encode(encoding='utf8'))
        idx = int(client.recv(1024).decode('utf8'))
        client.send('ok'.encode('utf8'))
        self.result_dict[idx] = []
        self.gradient_dict[idx] = []
        while True:
            data = recv_data(client)
            self.result_dict[idx].append(torch.Tensor(data))
            # client.send(bytes)
            # self.result_list.append(msg)
            while len(self.gradient_dict[idx]) == 0:
                time.sleep(0.1)
            send_data(client, self.gradient_dict[idx][0])
            try:
                del self.gradient_dict[idx][0]
            except:
                continue
            # bytes = client.recv(1024)
            # print("客户端消息:", bytes.decode(encoding='utf8'))
            # if len(bytes) == 0:
            #     client.close()
            # 删除连接
            # self.connect_pool.remove(client)
            # break

    def accept_connect(self):
        while self.is_training:
            client, _ = self.tcp_socket.accept()  # 阻塞，等待客户端连接
            # 加入连接池
            # self.connect_pool.append(client)
            # 给每个客户端创建一个独立的线程进行管理
            thread = threading.Thread(target=self.message_handle, args=(client,))
            self.connect_pool.append(client)
            # 设置成守护线程
            thread.setDaemon(True)
            thread.start()

    def train(self, epoch=0):
        socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.id != 0:
            socket_send.connect((IP_DICT[0][0], IP_DICT[0][1]))
            socket_send.send(str(self.id).encode('utf8'))
            socket_send.recv(1024)
        sum_loss = 0.0
        train_correct = 0
        torch.manual_seed(epoch)
        start = time.time()
        for i, data in enumerate(self.train_data_loader, 1):
            inputs, labels = data
            # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # outputs = self.model(inputs)
            outputs = self.local_model(inputs)
            label_owner = i % self.party_num
            if self.id != label_owner:
                send_data(socket_send, outputs)
                grad = recv_data(socket_send)
                # msg = bytes.decode('utf8')
                grad = torch.Tensor(grad).float()
                x = inputs.clone().detach().float()
                self.local_model.backward(x, grad)
                # loss = self.dense_model.backward(outputs, grad)
                # self.local_model.backward(x, loss)
                # output = self.model(x)
                # output.backward(gradient=grad)
            else:
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
                # output = outputs.clone().detach().requires_grad_(True)
                # outputs = self.dense_model(outputs)
                # loss = self.cost(outputs, labels)
                # U = outputs
                class_loss = self.cost(outputs, labels)
                grads = torch.autograd.grad(outputs=class_loss, inputs=outputs)[0]
                for k in self.gradient_dict.keys():
                    self.gradient_dict[k].append(grads)
                # loss.backward()
                # outputs.backward(gradient=torch.Tensor(grads).float())
                # loss = self.dense_model.backward(output, torch.Tensor(grads).float())
                # self.local_model.backward(inputs, loss)
                self.local_model.backward(inputs, grads)
            # self.optimizer.step()
        self.time_counter += time.time() - start
        if self.id == 0:
            print(self.time_counter)
        for i, data in enumerate(self.test_data_loader, 1):
            inputs, labels = data
            outputs = self.local_model(inputs)
            # outputs = self.dense_model(outputs)
            if self.id != 0:
                send_data(socket_send, outputs)
                recv_data(socket_send)
            else:
                while True:
                    count = 1
                    for k in self.result_dict.keys():
                        if len(self.result_dict[k]) > 0:
                            count += 1
                    if count == self.party_num:
                        break
                    time.sleep(0.1)
                for k in self.result_dict.keys():
                    outputs += self.result_dict[k][0]
                    del self.result_dict[k][0]
                    self.gradient_dict[k].append(torch.Tensor([[0.0]]).float())
            loss = self.cost(outputs, labels)
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        print('epoch: %d loss: %.03f' % (epoch, sum_loss / len(self.test_data_loader.dataset)))
        print('        correct:%.03f%%' % (100 * train_correct / len(self.test_data_loader.dataset)))


if __name__ == '__main__':
    p = Participant('1', 1, [], 'localhost', 8888, [], [], [], None)
    thread = threading.Thread(target=accept_connect, args=(p.tcp_socket, p.message_handle, p.is_training))
    thread.setDaemon(True)
    thread.start()
