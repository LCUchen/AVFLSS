# VFLExperiments

#### 介绍
纵向联邦学习对比实验及测试框架代码

#### 软件架构
项目结构为  
```bash
.
├── dataset
│   ├── fashion
│   │   ├── load_mnist_data.py
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   └── MNIST
│       ├── load_mnist_data.py
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── train-images-idx3-ubyte.gz
│       └── train-labels-idx1-ubyte.gz
├── experiments
│   ├── asycn_client.py
│   ├── client.py
│   ├── conf.py
│   ├── fashion
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── model.py
│   ├── he_client.py
│   ├── __init__.py
│   └── MNIST
│       ├── __init__.py
│       ├── main.py
│       ├── model.py
│       └── pa.py
├── README.en.md
└── README.md
``` 
其中，**fashion**目录下的数据集为`Fashion_MNIST`数据集  

#### 示例
运行同态加密模式，进入experiments/MNIST目录下，运行如下指令启动服务器：  
```python main.py --mode 1 --id -1```  
默认参与者有三个，id为0, 1, 2：  
``python main.py --mode 1 --id 0``  
``python main.py --mode 1 --id 1``  
``python main.py --mode 1 --id 2``  
等待运行结果  
`mode`字段为**0**时表示同步模式（明文），为**1**时表示同态加密模式，为**2**时表示异步模式  



