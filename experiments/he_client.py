import random
import socket
import threading
import time

import torch.nn
from experiments.conf import IP_DICT
from torch.utils.data import DataLoader, TensorDataset
from phe import paillier, PaillierPublicKey, EncryptedNumber
from utils.communication import save_result, calculate_result, send_data, recv_data


# 生成公私钥
# public_key, private_key = paillier.generate_paillier_keypair()


# 加密操作
def encryption(message_list, n):
    # g = n + 1
    public_key = PaillierPublicKey(n)
    encrypted_message_list = [public_key.encrypt(m, r_value=1).ciphertext() for m in message_list]
    return encrypted_message_list


def add_encrypted_value(list_a, list_b, n):
    # 对二维列表a和b中每个元素相加，a和b的长度应该相同
    public_key = PaillierPublicKey(n)
    result = []
    for i in range(len(list_a)):
        res = []
        for j in range(len(list_a[0])):
            res.append(
                (EncryptedNumber(public_key, list_a[i][j]) + EncryptedNumber(public_key, list_b[i][j])).ciphertext())
        result.append(res)
    return result


# 解密操作
def decryption(encrypted_message_list, private_key, n):
    public_key = PaillierPublicKey(n)
    decrypted_message_list = [private_key.decrypt(EncryptedNumber(public_key, c)) for c in encrypted_message_list]
    return decrypted_message_list


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


# 同态加密密钥分发服务器
class Server:
    def __init__(self, name, id, party_num):
        self.name = name
        self.id = id
        self.host = IP_DICT[self.id][0]
        self.port = IP_DICT[self.id][1]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.bind((self.host, self.port))
        self.tcp_socket.listen(5)
        self.connect_dict = {}
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.party_num = party_num
        self.status = [True] * self.party_num

    def message_handle(self, client):
        idx = int(client.recv(1024).decode('utf8'))
        client.send('ok\n'.encode('utf8'))
        self.connect_dict[idx] = client
        cmd = client.recv(1024).decode('utf8')
        print(idx, cmd)
        if cmd == 'public key':
            # 发送公钥
            print('send public key to %d ' % idx)
            send_public_key(client, self.public_key.n)
            print('done')
        elif cmd == 'decryption':
            # 同步
            while not self.status[idx]:
                time.sleep(0.1)
            # 接收加密后的数据
            encrypted_data = recv_data(client)
            # 解密数据
            decrypted_data = [decryption(m, self.private_key, self.public_key.n) for m in encrypted_data]
            # 发送解密后的数据
            send_data(client, decrypted_data)
            self.status[idx] = False
        self.connect_dict[idx] = None

    def accept_connect(self):
        while True:
            client, _ = self.tcp_socket.accept()
            thread = threading.Thread(target=self.message_handle, args=(client,))
            thread.setDaemon(True)
            thread.start()

    def train(self):
        flag = True
        while flag:
            flag = False
            # 同步
            for i in range(self.party_num):
                if self.status[i]:
                    flag = True
                    time.sleep(0.1)
                    break
        # 更新密钥
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.status = [True] * self.party_num


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
        self.public_key = 0
        self.thread_list = []
        self.time_list = []
        self.tp_list = []
        self.tn_list = []
        self.fp_list = []
        self.fn_list = []
        self.mse_list = []
        self.var_list = []
        self.loss_list = []
        self.acc_list = []

    def finish_training(self):
        self.is_training = False

    def save_result(self, path='result.csv'):
        save_result(path, self.tp_list, self.tn_list, self.fp_list, self.fn_list, self.time_list, self.mse_list,
                    self.var_list, self.loss_list, self.acc_list)

    def message_handle(self, client):
        # 接收来自其它参与者的请求，并将加密后的计算结果发送给请求方
        idx = int(client.recv(1024).decode('utf8'))
        print('par %d: ' % idx)
        client.send('ok\n'.encode('utf8'))
        if not self.result_dict.__contains__(idx):
            self.result_dict[idx] = []
        while len(self.result_dict[idx]) == 0:
            time.sleep(0.1)
        print('begin sending data')
        send_data(client, self.result_dict[idx])
        print('send data to par %d' % idx)
        self.result_dict[idx] = []

    def accept_connect(self):
        while self.is_training:
            client, _ = self.tcp_socket.accept()
            thread = threading.Thread(target=self.message_handle, args=(client,))
            thread.setDaemon(True)
            thread.start()
            self.thread_list.append(thread)

    def train(self, epoch=0):
        sum_loss = 0.0
        train_correct = 0
        # 设置随机数，同步各个参与者的数据集
        torch.manual_seed(epoch)
        start = time.time()
        for i, data in enumerate(self.train_data_loader, 1):
            inputs, labels = data
            # 计算结果，并将结果保存
            outputs = self.local_model(inputs)
            # 先获取同态加密公钥
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((IP_DICT[-1][0], IP_DICT[-1][1]))
            sock.send(str(self.id).encode('utf8'))
            sock.recv(1024)
            sock.send('public key'.encode('utf8'))
            # 这里实际上只保存了公钥的n
            public_key = recv_public_key(sock)
            sock.close()
            # 判断是否有标签信息（id为0的参与者保存有标签信息），根据情况计算要发送给其他参与者的数据
            label_owner = i % self.party_num
            if self.id == label_owner:
                encrypted_output = []
                label_list = labels.tolist()
                lb_idx = 0
                for out in outputs.tolist():
                    res = [.25 * o - label_list[lb_idx] + .5 for o in out]
                    encrypted_output.append(encryption(res, public_key))
                for par in self.party_list:
                    # 将计算结果保存到字典中，将会在另一个线程中发送数据
                    self.result_dict[par] = encrypted_output
            else:
                encrypted_output = []
                for out in outputs.tolist():
                    encrypted_output.append(encryption([o * .25 for o in out], public_key))
                for par in self.party_list:
                    self.result_dict[par] = encrypted_output
            # 请求获取其它参与者的数据（实现同态加密算法）
            # TODO: 收到的数据可能是inf（无穷大），此时无法进行同态运算，需要优化
            encrypted_data = []
            sock_list = []
            for par in self.party_list:
                print('client %d:' % par)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_list.append(sock)
                sock.connect((IP_DICT[par][0], IP_DICT[par][1]))
                sock.send(str(self.id).encode('utf8'))
                sock.recv(3)
                encrypted_data.append(recv_data(sock))
            print('aggregate data done!')
            for sock in sock_list:
                sock.close()
            # 计算加密结果，并向结果中添加噪声
            random_list = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(self.batch_size)]
            encrypted_random_list = []
            for rnd in random_list:
                encrypted_random_list.append(encryption(rnd, public_key))
            encrypted_result = add_encrypted_value(encrypted_output, encrypted_random_list, public_key)
            for enc in encrypted_data:
                encrypted_result = add_encrypted_value(encrypted_output, enc, public_key)
            # 请求服务器解密数据
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((IP_DICT[-1][0], IP_DICT[-1][1]))
            sock.send(str(self.id).encode('utf8'))
            sock.recv(1024)
            sock.send('decryption'.encode('utf8'))
            send_data(sock, encrypted_result)
            # print('decrypting data')
            decrypted_result = torch.tensor(recv_data(sock)).float()
            # print('decrypting data done!')
            # 去除添加的噪声，获得梯度信息
            # TODO: 利用获取到的解密数据计算梯度信息，这里未完成
            decrypted_result -= torch.tensor(random_list).float()
            # 更新模型
            # print('updating model')
            x = inputs.clone().detach().float()
            self.local_model.backward(x, decrypted_result)
            # print('updating model done!')
            # self.optimizer.step()
        self.time_counter += time.time() - start
        if self.id == 0:
            print(self.time_counter)
        # 测试，用明文
        for i, data in enumerate(self.test_data_loader, 1):
            inputs, labels = data
            outputs = self.local_model(inputs)
            # outputs = self.dense_model(outputs)
            if self.id != 0:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((IP_DICT[0][0], IP_DICT[0, 1]))
                send_data(sock, outputs)
                recv_data(sock)
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
            tp, tn, fp, fn, mse, var, acc = calculate_result(id.tolist(), labels.tolist())
            self.tp_list.append(tp)
            self.tn_list.append(tn)
            self.fp_list.append(fp)
            self.fn_list.append(fn)
            self.mse_list.append(mse)
            self.var_list.append(var)
            self.time_list.append(self.time_counter)
            self.loss_list.append(sum_loss / len(self.test_data_loader.dataset))
            self.acc_list.append(acc)
        print('epoch: %d loss: %.03f' % (epoch, sum_loss / len(self.test_data_loader.dataset)))
        print('        correct:%.03f%%' % (100 * train_correct / len(self.test_data_loader.dataset)))


if __name__ == '__main__':
    p = Participant('1', 1, 3, [], [], [], None, None)
    thread = threading.Thread(target=p.accept_connect, args=(p.tcp_socket, p.message_handle, p.is_training))
    thread.setDaemon(True)
    thread.start()
