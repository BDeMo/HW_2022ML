num_train = 2000
num_test = 3527 - num_train
learning_rate = 1e-4
batchsize = 20
num_epochs = 100
times = 10
gpu = 0
mandl = [
    ('linear', 'MSELoss')
    # ('mclr', 'NLLLoss'),
    # ('dnn', 'NLLLoss')
]
