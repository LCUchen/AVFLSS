import matplotlib.pyplot as plt
from experiments.conf import home


def draw_linear(x_list, y_list):
    plt.figure()
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i])
    plt.show()


def read_result(file_list):
    all_time, all_precision, all_recall, all_r_square, all_acc, all_f1, all_loss = [], [], [], [], [], [], []
    for file in file_list:
        time_list, precision_list, recall_list, r_square_list, acc_list, f1_list, loss_list = [], [], [], [], [], [], []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                if len(line) < 1:
                    break
                data = line.strip().split(';')
                tp, tn, fp, fn, time, mse, var, loss, acc = (float(i) for i in data)
                time_list.append(time)
                precision_list.append(tp / (tp + fp))
                recall_list.append(tp / (tp + fn))
                r_square_list.append(1 - mse / var)
                acc_list.append(acc)
                f1_list.append(2 * tp / (tp + tp + fp + fn))
                loss_list.append(loss)
        all_time.append(time_list)
        all_precision.append(precision_list)
        all_recall.append(recall_list)
        all_r_square.append(r_square_list)
        all_acc.append(acc_list)
        all_f1.append(f1_list)
        all_loss.append(loss_list)
    return all_time, all_precision, all_recall, all_r_square, all_acc, all_f1, all_loss


if __name__ == '__main__':
    all_time, all_precision, all_recall, all_r_square, all_acc, all_f1, all_loss = read_result([home + 'result/mnist-0-1'])
    draw_linear(all_time, all_acc)
