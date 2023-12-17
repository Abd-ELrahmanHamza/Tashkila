import torch
from harakat_hot_encoding import harakat2idx
from characters_hot_encoding import char2idx

XTRAIN_PATH = f'../clean_out/X.csv'
YTRAIN_PATH = f'../clean_out/y.csv'


def read_data(input_path,output_path,verbose=False):
    X = []
    Y = []
    # read csv files
    with open(input_path, 'r', encoding="utf8") as f:
        X = f.readlines()
    with open(output_path, 'r', encoding="utf8") as f:
        Y = f.readlines()

    # remove the \n from the end of each line
    # make sure the number of lines in the two files are the same
    assert len(X) == len(Y)

    # split each line into a list of characters
    X = [x.strip().split('s') for x in X]
    Y = [y.strip().split('s') for y in Y]

    X = [item for sublist in X for item in sublist] + ['']*8
    Y = [item for sublist in Y for item in sublist] + ['$']*8

    if verbose:
        print('Data read successfully.')
        print(f'Number of lines: {len(X)}')
        print(f'X: {X[0]}')
        print(f'Y: {Y[0]}')

    return X, Y

X, Y = read_data(XTRAIN_PATH, YTRAIN_PATH, verbose=False)

x_new = [char2idx[x] for x in X]
y_new = [harakat2idx[y] for y in Y]

data_len = len(x_new)

# get the largest num that len of x_new can be divided by
# so that we can convert it to a tensor
num = 1
l = 1
r = 1001
for i in range(r, l, -1):
    if data_len % i == 0:
        num = i
        break

# convert data to 1000 sentences each and convert it to tensors
x_new = torch.tensor(x_new).view(-1, num)
y_new = torch.tensor(y_new).view(-1, num)

# convert to hot encoding
x_train = x_new

y_train = y_new

print(x_train.shape)
print(y_train.shape)