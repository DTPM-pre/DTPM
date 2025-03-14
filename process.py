from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def process_data(features, targets):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=features.columns)  # 

    scaler_target = StandardScaler()
    targets_scaled = scaler_target.fit_transform(targets.values.reshape(-1, 1)).ravel()

    return features_df, targets_scaled


def get_data(X_all, Y_all, all_sample_num, sample_size, seed):
    np.random.seed(seed)
    print('X_all shape',X_all.shape)
    print(Y_all.shape)
    permutation = np.random.permutation(all_sample_num)
    training_index = permutation[0:sample_size]
    X_train = X_all[training_index, :]
    Y_train = Y_all[training_index, :]

    return X_train, Y_train, seed

def seed_generator(sys_name, sample_size,N_train_all):
    """
    Generate the initial seed for each sample size (to match the seed of the results in the paper)
    This is just the initial seed, for each experiment, the seeds will be equal the initial seed + the number of the experiment

    Args:
        sys_name: [string] the name of the dataset
        sample_size: [int] the total number of samples
    """

    # N_train_all = system_samplesize(sys_name)
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)

    return seed_o
