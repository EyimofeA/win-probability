import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys



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

def model(x,w,b):
    p = stable_sigmoid(x @ w + b)
    p = np.clip(p,1e-15,1-1e-15) #clipping prevents overflow
    return p

def plot_diags(preds,test):
    preds = preds.squeeze()
    bins = np.arange(0,1.1,0.1)
    inds = np.digitize(preds,bins)
    bin_count = np.bincount(inds)
    bin_val = np.bincount(inds,weights=preds)
    t_val = np.bincount(inds,weights=test.squeeze())
    bin_avg = bin_val/bin_count
    bin_tavg =t_val/bin_count

    fig,(ax1,ax2) = plt.subplots(2,1)

    ax1.plot(bin_avg,bin_tavg,color = "black")
    ax1.axline((0,0),slope = 1,color = "red")
    ax1.set(xlim=(0,1),ylim=(0,1),xlabel="Model Prediction",ylabel="Observed Result",title=f"{MODEL_NAME} Reliability Curve")

    ax2.hist(preds,bins=10)
    ax2.set(xlabel="Model Prediction",ylabel="Count",title="Freq of Preds")

    # # if "--keep" not in sys.argv:
    # #     timer = fig.canvas.new_timer(interval=15000)  # 1.5 seconds
    # #     timer.add_callback(plt.close)
    # #     timer.start()

    # plt.tight_layout()
    # plt.show()
def plot_reliability_curve(preds, test):
    preds = preds.squeeze()
    test = test.squeeze()
    
    # Binning logic
    bins = np.arange(0, 1.1, 0.1)
    inds = np.digitize(preds, bins)
    bin_count = np.bincount(inds)
    bin_val = np.bincount(inds, weights=preds)
    t_val = np.bincount(inds, weights=test)
    
    # Safety: Only calculate averages for bins that actually have data
    # This prevents division-by-zero warnings if a 10% bracket is empty
    valid_bins = bin_count > 0
    bin_avg = bin_val[valid_bins] / bin_count[valid_bins]
    bin_tavg = t_val[valid_bins] / bin_count[valid_bins]
    bin_count = bin_count[valid_bins]

    ece = np.average(np.abs(bin_avg - bin_tavg),weights=bin_count)

    # --- BEARS BLOG STYLING ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams.update({'font.size': 14})

    # Force a square figure so the 1:1 slope isn't distorted
    plt.figure(figsize=(8, 8))

    # 1. Perfect Calibration Line (Subtle)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Perfect Calibration')

    # 2. Model Reliability Line (Bold with markers)
    plt.plot(bin_avg, bin_tavg, marker='o', color='#000000', linewidth=2.5, markersize=8, label=f'{MODEL_NAME}')

    # Axes and Labels
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Model Prediction (Binned Probability)")
    plt.ylabel("Observed Result (Actual Win Rate)")
    plt.title(f"{MODEL_NAME} Reliability Diagram", pad=20, fontweight='bold')

    # Clean Legend (No bounding box)
    plt.legend(loc="upper left", frameon=False)

    # Add a very faint grid to help the eye measure deviation
    plt.grid(True, linestyle=':', alpha=0.4)

    # Remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Export
    plt.savefig(f"outputs/{MODEL_NAME}_reliability_curve.svg", format='svg',  bbox_inches='tight')
    plt.savefig(f"outputs/{MODEL_NAME}_reliability_curve.png", format='png',  bbox_inches='tight')
    plt.show()
    return ece
    
if __name__ == "__main__":
    MODEL_NAME = "2026-07-14-v2-stern-with-clip.npz"
    print("Starting training loop...")

    data_path = "./data"
    df = pd.read_parquet(f"{data_path}/wp_training_data_2025-26.parquet")

    # feature eng
    df["margin_time_interaction"] = df["margin"] * df["seconds_left"]
    df["time_frac_remaining"] = df["seconds_left"] / 2880 #normalized 1 -> 0
    df["margin_per_sqrt_time"] = df["margin"] / np.sqrt(df["seconds_left"]+1)
    df["sqrt_time_frac"] = np.sqrt(df["seconds_left"]/2880)

    game_ids = df["gameId"].unique()
    rng = np.random.default_rng(seed=42)
    train_ids = rng.choice(game_ids,size=int(0.8*len(game_ids)),replace=False)
    df.set_index("gameId",inplace=True)
    train = df.loc[train_ids]
    test = df.loc[~df.index.isin(train.index)]

    # clip based on train only
    lower = train["margin_per_sqrt_time"].quantile(0.01)
    upper = train["margin_per_sqrt_time"].quantile(0.99)
    train["margin_per_sqrt_time"] = train["margin_per_sqrt_time"].clip(lower,upper)
    test["margin_per_sqrt_time"] = test["margin_per_sqrt_time"].clip(lower,upper)
   
    features = ["margin","seconds_left","margin_time_interaction","margin_per_sqrt_time","sqrt_time_frac"]
    target = ["home_won"]

    train = train[features + target].to_numpy()
    test = test[features + target].to_numpy()

    N_train,_ = train.shape
    X_train = train[:,:-1]
    Y_train = train[:,-1].reshape(-1,1)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    x = (X_train - x_mean)/x_std
    y = Y_train

    w = np.zeros((x.shape[1],1)) 
    b = 0.0

    learning_rate = 0.1
    epochs = 10000 #will try other methods, easier for now
    losses = []
    for e in range(epochs):
        # forward pass
        p = model(x,w,b) 
        loss = - (y * np.log(p) + (1-y)*np.log(1-p)).mean()
        losses.append(loss)

        # loss.backwards() smh i wish
        dldw = (x.T @ (p-y)) / N_train
        dldb = (p-y).mean()
        temp_w = w.copy()
        w = w - learning_rate * dldw
        b = b - learning_rate * dldb

        if e % (epochs/20) ==0:
            learning_rate*=0.9
            print(f"Epoch = {e}, loss = {loss:.4f}, grad_norm = {(np.abs(dldw).sum() + abs(dldb)):.4f}, eta = {learning_rate:.4f}")
        # if np.abs(dldw).sum() + abs(dldb) < 1e-15: #early stopping using gradient norm.
        #     break

    print(f"Final weights are {w=}\n and {b}")

    N_test,_ = test.shape
    X_test = test[:,:-1]
    Y_test = test[:,-1].reshape(-1,1)
    x_test = (X_test - x_mean)/x_std
    y_test = Y_test

    brier_score = np.square((model(x_test,w,b) - y_test)).mean()
    print(f"Model brier_score is {brier_score}")

    test_preds = model(x_test,w,b)
    # plot_diags(test_preds,y_test)
    ece = plot_reliability_curve(test_preds,y_test)
    print(f"Model ECE is {ece}")

    np.savez(f"models/{MODEL_NAME}", weights = w,biases = b,scaler_mean = x_mean, scaler_std = x_std,feature_names= features,clipping_percentile = {lower,upper})
    print("Model saved")

    



    

