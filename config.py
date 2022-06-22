num_train = 2000
num_test = 3527 - num_train
exp_name = 'hw_2022ml'
data_path = './data/data.npy'
label_path = './data/label.npy'
dec_data_path = './data/dec_data.npy'
batchsize = 20

NUM_CLASSES = 2
learning_rate = 1e-4
num_epochs = 100
times = 10
gpu = 0
dec = True
dec_dim = 1000

LLSM = True

mandl = [
    # ('lnr', 'MSELoss'),
    # ('mclr', 'NLLLoss')
    ('dnn', 'NLLLoss')
]
