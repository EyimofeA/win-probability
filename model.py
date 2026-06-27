import pandas as pd
import numpy as np

folder_path = "."
df = pd.read_parquet(f"{folder_path}/wp_training_data_2025-26.parquet")

game_ids = df["gameId"].unique()
train_ids = np.random.choice(game_ids,size=int(0.8*len(game_ids)),replace=False)
df.set_index("gameId",inplace=True)
train = df.loc[train_ids]
test = df.loc[~df.index.isin(train.index)]

train = train[["margin","seconds_left","home_won"]].to_numpy()
test = test[["margin","seconds_left","home_won"]].to_numpy()

N_train,_ = train.shape
X_train = train[:,:2]
Y_train = train[:,2].reshape(-1,1)

x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)

x = (X_train - x_mean)/x_std
y = Y_train

w = np.zeros((x.shape[1],1)) 
b = 0.0

learning_rate = 0.1
epochs = 10000 #will try other methods, easier for now

def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)

def stable_sigmoid(x):
    positive = x >=0
    negative = ~positive

    result = np.empty_like(x)

    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result

def model(x):
    p =  p = stable_sigmoid(x @ w + b)
    p = np.clip(p,1e-15,1e15)
    return p

for e in range(epochs):
    # forward pass
    p = stable_sigmoid(x @ w + b)
    p = np.clip(p,1e-15,1-1e-15) #clipping prevents overflow

    loss = - (y * np.log(p) + (1-y)*np.log(1-p)).mean()
    

    # loss.backwards() smh i wish
    dldw = (x.T @ (p-y)) / N_train
    dldb = (p-y).mean()
    temp_w = w.copy()
    w = w - learning_rate * dldw
    b = b - learning_rate * dldb

    if e % (epochs/20) ==0:
        print(f"Epoch = {e}, loss = {loss}, dw = {(w-temp_w).sum()}")
    if np.abs(dldw).sum() + abs(dldb) < 1e-8: #early stopping using gradient norm.
        break

print(f"Final weights are {w} and {b}")

N_test,D = test.shape
X_test = test[:,:2]
Y_test = test[:,2].reshape(-1,1)
x_test = (X_test - x_mean)/x_std
y_test = Y_test

brier_score = np.square((model(x_test) - y_test)).mean()
print(brier_score)
