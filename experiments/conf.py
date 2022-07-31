import torch

# IP_DICT = {-1: ['127.0.0.1', 9999], 0: ['127.0.0.1', 8888], 1: ['127.0.0.1', 8889], 2: ['127.0.0.1', 8890], 3: ['127.0.0.1', 8891], 4: ['127.0.0.1', 8892]}
# IP_DICT = {-1: ['127.0.0.1', 9999], 0: ['127.0.0.1', 9888], 1: ['127.0.0.1', 9889], 2: ['127.0.0.1', 9890], 3: ['127.0.0.1', 9891], 4: ['127.0.0.1', 9892]}
# IP_DICT = {-1: ['127.0.0.1', 9999], 0: ['127.0.0.1', 6888], 1: ['127.0.0.1', 6889], 2: ['127.0.0.1', 6890], 3: ['127.0.0.1', 6891], 4: ['127.0.0.1', 6892]}
IP_DICT = {-1: ['127.0.0.1', 9999], 0: ['127.0.0.1', 5888], 1: ['127.0.0.1', 5889], 2: ['127.0.0.1', 5890], 3: ['127.0.0.1', 5891], 4: ['127.0.0.1', 5892]}
# differential privacy
home = '/home/chen/Code/vflexperiments/'
epsilon = .1
max_thread_count = 5
max_epoch = 100
shutdown_pro = 0
time_out = 300
# device = torch.device('cuda:0')
device = 'cpu'
dataset_config = {
    'boston_housing': {
        'train_ratio': .7,
        'loss_function': torch.nn.MSELoss(),
        'y_type': 'float'
    },
    'wine_quality': {
        'train_ratio': .7,
        'loss_function': torch.nn.MSELoss(),
        'y_type': 'float'
    },
    'MNIST': {
        'train_ratio': .7,
        'loss_function': torch.nn.CrossEntropyLoss(),
        'y_type': 'long'
    },
    'cifar10': {
        'loss_function': torch.nn.CrossEntropyLoss(),
        'y_type': 'long'
    },
    'credit_card': {
        'train_ratio': .7,
        'loss_function': torch.nn.CrossEntropyLoss(),
        'y_type': 'float'
    },
    'parkinsons': {
        'train_ratio': .7,
        'loss_function': torch.nn.MSELoss(),
        'y_type': 'float'
    }
}
