import socket
import threading
import time
import numpy as np
from sklearn.metrics import mean_squared_error


def send_data(client, msg):
    # msg: [[]]
    # msg = grad.tolist()
    # en_msg = [encryption(i, n) for i in msg]
    length = len(str(msg).encode('utf8'))
    client.send(str(length).encode('utf8'))
    # client.recv(3)
    while True:
        try:
            by = client.recv(3)
            if len(by) == 0:
                continue
            break
        except socket.error:
            time.sleep(0.1)
    client.send(str(msg).encode('utf8'))


def recv_data(client):
    client.setblocking(0)
    while True:
        try:
            by = client.recv(1024)
            if len(by) == 0:
                continue
            break
        except socket.error:
            time.sleep(0.1)
    length = int(by.decode('utf8'))
    print('receive length')
    client.send('ok\n'.encode('utf8'))
    print('send ok')
    while True:
        try:
            by = client.recv(1024)
            while len(by) < length:
                by += client.recv(1024)
            break
        except:
            time.sleep(0.1)
    msg = by.decode('utf8')[1:-1] + ','
    print('received data!')
    data = []
    for l in msg.split('],')[:-1]:
        l = l.strip()[1:]
        d = []
        for i in l.split(','):
            d.append(int(i))
        data.append(d)
    print('return data')
    return data


def request_handshake(ip, port, client_id, index=0, cmd=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # print('done')
    print(ip, port, flush=True)
    sock.connect((ip, port))
    sock.send((str(client_id) + ',' + str(index) + ',' + str(cmd)).encode('utf8'))
    sock.setblocking(0)
    while True:
        try:
            by = sock.recv(1024)
            if len(by) == 0:
                continue
            break
        except socket.error:
            # time.sleep(0.1)
            continue
    sock.setblocking(1)
    return sock


def accept_handshake(sock):
    idx, index, cmd = sock.recv(1024).decode('utf8').split(',')
    idx = int(idx)
    index = int(index)
    cmd = int(cmd)
    sock.send('ok\n'.encode('utf8'))
    return idx, index, cmd


def calculate_result(y_pre, y_true):
    tp, tn, fp, fn, mse, var, acc = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(y_pre)):
        if y_true[i] == 1:
            if y_pre[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_pre[i] == 0:
                tn += 1
            else:
                fp += 1
        if y_true[i] == y_pre[i]:
            acc += 1
    mse = mean_squared_error(y_true, y_pre)
    var = np.var(y_true)
    acc /= len(y_true)
    return tp, tn, fp, fn, mse, var, acc


def save_result(path, TP_list, TN_list, FP_list, FN_list, time_list, mse_list, var_list, loss_list, acc_list):
    with open(path, 'w', encoding='utf8') as file:
        for i in range(len(time_list)):
            file.write(str(TP_list[i]) + ';' + str(TN_list[i]) + ';')
            file.write(str(FP_list[i]) + ';' + str(FN_list[i]) + ';')
            file.write(str(time_list[i]) + ';' + str(mse_list[i]) + ';')
            file.write(str(var_list[i]) + ';' + str(loss_list[i]) + ';')
            file.write(str(acc_list[i]) + '\n')


