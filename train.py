import threading
import pandas as pd
import numpy as np
import os
import joblib  # For saving and loading the model
import random
import time
import new_dataframe_module as ndm
import concurrent.futures


from correlations import calculate_combined_correlations
from select_top_feature import select_top_features
from process import process_data, get_data, seed_generator
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt


# python -u re_train_ours_multi-process_new_B_1.py > ./logs/re_train_ours_multi-process_new_B_1_1.txt &


def optimize_feature_combination(new_features, new_targets, top_feature_indices, sys_name, sample_size, seed, m):

    start_time_optimize = time.time()

    # Generate feature sets A and B
    selected_df_A, _ = divide_two_sets(new_features, top_feature_indices, new_targets)

    selected_features_df_A = selected_df_A.iloc[:, :-1]
    selected_targets_df_A = selected_df_A.iloc[:, -1]

    # Convert to NumPy arrays
    selected_features_df_A = selected_features_df_A.values
    selected_targets_df_A = selected_targets_df_A.values

    # feature space A 
    features = list(selected_df_A.columns[:-1])
    new_features_df = ndm.from_pandas(new_features)

    # 
    initial_performance, initial_xgb_model, initial_rf_model, initial_avg_w1, initial_avg_w2 = evaluate_model(
        selected_features_df_A, selected_targets_df_A, seed)

    best_performance = initial_performance
    best_feature_combination = features[:]

    best_avg_w1 = initial_avg_w1
    best_avg_w2 = initial_avg_w2
    all_features = set(new_features.columns)
    A_features = set(top_feature_indices)

    B_features = list(all_features - A_features)
    B_lock = threading.Lock()  # B_features

    improved = True
    print('Initial_features:{}'.format(best_feature_combination))
    while True:
       
        least_impactful_feature = None
        temp_sub_performance = 0
        weakest_performance = best_performance
        
        # 
        performance_lock = threading.Lock()

        def evaluate_feature_removal(feature, features_copy):
            temp_features_a = features_copy.copy()
            temp_features_a.remove(feature)
            temp_df_A = new_features_df.get_columns_as_numpy(temp_features_a)
            temp_performance, _, _, _, _ = evaluate_model(temp_df_A, selected_targets_df_A, seed)
            return feature, temp_performance

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_removal = [executor.submit(evaluate_feature_removal, feature, features[:]) for feature in features]
            for future_removal in concurrent.futures.as_completed(futures_removal):
                feature, temp_performance = future_removal.result()
                if temp_performance > weakest_performance:
                    continue
                # 
                with performance_lock:
                    if temp_sub_performance < (weakest_performance - temp_performance):
                        temp_sub_performance = weakest_performance - temp_performance
                        best_performance = temp_performance
                        least_impactful_feature = feature
                       

        print('remove_feature{}'.format(least_impactful_feature))

        print("second")
        improved = False


        
        def check_feature_combination(feature, features_copy):
            if '+' in feature:
                feature1, feature2 = feature.split('+')
                feature_set = set([f'{feature1}-{feature2}', f'{feature1}*{feature2}'])
            elif '-' in feature:
                feature1, feature2 = feature.split('-')
                feature_set = set([f'{feature1}-{feature2}', f'{feature1}*{feature2}'])
            elif '*' in feature:
                feature1, feature2 = feature.split('*')
                feature_set = set([f'{feature1}-{feature2}', f'{feature1}*{feature2}'])
            else:
                return False
            # 
            if not set(features_copy).isdisjoint(feature_set):
                # 
                return True
            return False
        
        def evaluate_feature_replacement(least_impactful_feature, features_copy):
            # nonlocal num_circles
            nonlocal B_features
            if least_impactful_feature is None:
                return None  # break 
            temp_features_b = features_copy.copy()
            temp_features_b.remove(least_impactful_feature)

            random_feature = random.choice(B_features)
            while check_feature_combination(random_feature, features_copy):
                random_feature = random.choice(B_features)

            temp_features_b.append(random_feature)
            temp_df_A = new_features_df.get_columns_as_numpy(temp_features_b)
            performance, temp_xgb_model, temp_rf_model, temp_avg_w1, temp_avg_w2 = evaluate_model(
                temp_df_A, selected_targets_df_A, seed)

            return random_feature, performance, temp_features_b, temp_xgb_model, temp_rf_model, temp_avg_w1, temp_avg_w2

        with concurrent.futures.ThreadPoolExecutor() as executor:
            features_replacement_futures = [executor.submit(evaluate_feature_replacement, least_impactful_feature, features[:]) for _ in range(50)]
            for future_replacement in concurrent.futures.as_completed(features_replacement_futures):
                result = future_replacement.result()
                if result is None:
                    continue
                random_feature, performance, temp_features_b, _, _, temp_avg_w1, temp_avg_w2 = result
                with performance_lock:         
                    if performance < best_performance:
                        best_performance = performance
                        best_feature_combination = temp_features_b[:]
                        best_avg_w1 = temp_avg_w1
                        best_avg_w2 = temp_avg_w2
                        B_features.remove(random_feature)
                        improved = True

        features = best_feature_combination

        if not improved:
            print("third. additional checks")
            weakest_performance = best_performance
            least_impactful_feature = None

            def evaluate_additional_check(feature, features_copy):
                temp_features_a = features_copy.copy()
                temp_features_a.remove(feature)
                temp_df_A = new_features_df.get_columns_as_numpy(temp_features_a)
                temp_performance, _, _, _, _ = evaluate_model(temp_df_A, selected_targets_df_A, seed)
                return feature, temp_performance

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_additional = [executor.submit(evaluate_additional_check, feature, features[:]) for feature in features]
                for future_additional in concurrent.futures.as_completed(futures_additional):
                    feature, temp_performance = future_additional.result()
                    with performance_lock:
                        if temp_performance < weakest_performance:
                            weakest_performance = temp_performance
                            least_impactful_feature = feature
                            print('The new minimum impact feature is as follows: {}'.format(least_impactful_feature))

            if least_impactful_feature is not None:
                if least_impactful_feature in features:
                    features.remove(least_impactful_feature)
                best_performance = weakest_performance
        else:
            print("There are improvements, no additional checks.")

        if not improved and least_impactful_feature is None:
            break

    end_time_optimize = time.time()

    # save 
    final_features = new_features[best_feature_combination].values
    final_xgb_model, final_rf_model = train_model(final_features, selected_targets_df_A, seed)

    final_train_time = time.time() - end_time_optimize

    model_dir_path = f'./model/ours/models/models_select_weights_new_{m}'
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    weighted_dir = f'./model/ours/weights/xgb_rf_models_weights_new_{m}'
    if not os.path.exists(weighted_dir):
        os.makedirs(weighted_dir)

    final_xgb_model_path = f'{model_dir_path}/{sys_name}_AllNumeric_train_{sample_size}_final_xgb_model.pkl'
    final_rf_model_path = f'{model_dir_path}/{sys_name}_AllNumeric_train_{sample_size}_final_rf_model.pkl'
    

    save_trained_model(final_xgb_model, final_xgb_model_path)
    save_trained_model(final_rf_model, final_rf_model_path)

    np.save(f'{weighted_dir}/{sys_name}_{sample_size}_best_weights.npy', np.array([best_avg_w1, best_avg_w2]))

    return best_performance, best_feature_combination, initial_performance, end_time_optimize - start_time_optimize, final_train_time



def optimize_weights(pred_xgb, pred_rf, y_val):
    """XGBoost RandomForest weights"""
    best_w1, best_w2 = 0, 0
    best_score = float('inf')

    # Search
    for w1 in np.arange(0.1, 1.1, 0.1):  # 
        w2 = 1 - w1
        final_pred = w1 * pred_xgb + w2 * pred_rf

        # MRE
        score = np.mean(np.abs((y_val - final_pred) / y_val)) * 100

        if score < best_score:
            best_score = score
            best_w1, best_w2 = w1, w2

    return best_w1, best_w2, best_score


def evaluate_model(X, y, seed, k=5):
    
    model_xgb = XGBRegressor(random_state=seed)
    model_rf = RandomForestRegressor(random_state=seed)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    try:
        predictions = np.zeros(len(y))  # 
        all_w1, all_w2 = [], []  # 

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # training
            model_xgb.fit(X_train, y_train)
            model_rf.fit(X_train, y_train)

            # prediction
            val_predictions_xgb = model_xgb.predict(X_val)
            val_predictions_rf = model_rf.predict(X_val)

            # weight
            best_w1, best_w2, best_score = optimize_weights(val_predictions_xgb, val_predictions_rf, y_val)
            
            all_w1.append(best_w1)
            all_w2.append(best_w2)

            # final_prediction
            final_pred = best_w1 * val_predictions_xgb + best_w2 * val_predictions_rf
            predictions[val_index] = final_pred

        # final_weight
        avg_w1 = np.mean(all_w1)
        avg_w2 = np.mean(all_w2)

        # 
        epsilon = 1e-10
        non_zero_y = np.where(y != 0, y, epsilon)
        predictions[predictions < 0] = 0  # 
        predictions = np.nan_to_num(predictions, nan=0.0)  # 

        #  MRE (Mean Relative Error)
        mre = np.mean(np.abs((non_zero_y - predictions) / non_zero_y)) * 100
        return mre, model_xgb, model_rf, avg_w1, avg_w2  # 

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return float('inf'), None, None, None, None


def train_model(X, y, seed):
    """K-Fold Cross-Validation to Evaluate Model Performanc"""
    model_xgb = XGBRegressor(random_state=seed)
    model_rf = RandomForestRegressor(random_state=seed)

    try:
        # model.fit(X, y)
        model_xgb.fit(X, y)
        model_rf.fit(X, y)
        return model_xgb, model_rf
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None



def divide_two_sets(df, indices, targets):
    # Dummy function to divide data into two sets based on indices
    set_A = df.iloc[:, indices]
    set_B = df.drop(df.columns[indices], axis=1)
    return pd.concat([set_A, targets], axis=1), set_B 



def save_trained_model(model, path):
    try:
        joblib.dump(model, path)
        # print(f"Model successfully saved to {path}")
    except Exception as e:
        print(f"Failed to save model: {e}")


# sample sizes
def system_samplesize(sys_name):
    if sys_name == 'x264':
        num_representative_points = np.multiply(16, [1, 2, 4, 6])
        num_features = 13
    elif sys_name == 'lrzip':
        num_representative_points = np.multiply(19, [1, 2, 4, 6])
        num_features = 19
    elif sys_name == 'vp9':
        num_representative_points = np.multiply(41, [1, 2, 4, 6])
        num_features = 41
    elif sys_name == 'polly':
        num_representative_points = np.multiply(39, [1, 2, 4, 6])
        num_features = 39
    elif sys_name == 'Dune':
        num_representative_points = np.asarray([49, 78, 384, 600])
        
        num_features = 11
    elif sys_name == 'hipacc':
        num_representative_points = np.asarray([261, 528, 736, 1281])
        num_features = 33
    elif sys_name == 'hsmgp':
        num_representative_points = np.asarray([77, 173, 384, 480])
        
        num_features = 14
    elif sys_name == 'javagc':
        num_representative_points = np.asarray([855, 2571, 3032, 5312])
        
        num_features = 35
    elif sys_name == 'sac':
        num_representative_points = np.asarray([2060, 2295, 2499, 3261])
        num_features = 59
    else:
        raise AssertionError("Unexpected value of 'sys_name'!")

    return num_representative_points, num_features

sys_names = ['x264', 'lrzip', 'vp9', 'polly', 'Dune', 'hipacc', 'hsmgp', 'javagc', 'sac']

n_exp = 30

for sys_name in sys_names:
    result_sys = []
    samplesizes, num_features = system_samplesize(sys_name)

    for sample_size in samplesizes:
        mre_list = []

        read_data_time_list = []
        generate_feature_time_list = []
        select_top_feature_time_list = []
        optimize_time_list = []
        total_time_list = []
        final_train_time_list = []
        for m in range(1, n_exp + 1):
            print("Experiment: {}".format(m))

            start_time_1 = time.time()
            seed_init = seed_generator(sys_name, sample_size, samplesizes)

            # Set seed and generate training data
            seed = seed_init * n_exp + m

            # 
            original_data_path = f'./datasets/select_represent_data/{sys_name}_AllNumeric_train_{sample_size}.csv'
            original_data_df = pd.read_csv(original_data_path)

            # 
            start_time_2 = time.time()

            # 
            save_dir = f'./datasets/ours/new_feature_{m}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = f'{save_dir}/{sys_name}_NewFeature_train_{sample_size}.csv'

            new_data_df = ndm.from_pandas(original_data_df)
            new_data_df.generate_and_save_new_features(save_path)

            # 
            start_time_3= time.time()
            new_data_path = f'{save_dir}/{sys_name}_NewFeature_train_{sample_size}.csv'

            new_data_df = pd.read_csv(save_path)
            new_features = new_data_df.iloc[:, :-1]

            new_targets = new_data_df.iloc[:, -1]

            # 
            Standard_features, Standard_targets = process_data(new_features, new_targets)
            # 
            combined_correlations_df = calculate_combined_correlations(Standard_features, Standard_targets)

            # 
            top_percentage = 1
            new_num_features = int(top_percentage * num_features)
            # 
            top_feature_indices = select_top_features(combined_correlations_df, new_num_features)
            
            # 
            start_time_4 = time.time()

            best_performance, best_features, initial_performance, optimize_time, final_train_time = optimize_feature_combination(
                new_features,
                new_targets,
                top_feature_indices,
                sys_name,
                sample_size,
                seed,
                m
            )

            read_data_time = start_time_2 - start_time_1
            generate_feature_time = start_time_3 - start_time_2
            select_top_feature_time = start_time_4 - start_time_3
            total_time = read_data_time + generate_feature_time + select_top_feature_time + optimize_time + final_train_time

            read_data_time_list.append(read_data_time)
            generate_feature_time_list.append(generate_feature_time)
            select_top_feature_time_list.append(select_top_feature_time)
            optimize_time_list.append(optimize_time)
            final_train_time_list.append(final_train_time)
            total_time_list.append(total_time)


            print("read_data_time:{}".format(read_data_time))
            print("generate_feature_time:{}".format(generate_feature_time))
            print("select_top_feature_time:{}".format(select_top_feature_time))
            print("optimize_time:{}".format(optimize_time))
            print("final_train_time:{}".format(final_train_time))
            print("total_time:{}".format(total_time))

            mre_list.append(best_performance)

            print("Best Performance:", best_performance)
            print("Best Feature Combination:", best_features)

            data_dict = {
                "sys_name": [sys_name],
                "sample_size": [sample_size],
                "Best Performance": [best_performance],
                "Best Feature Combination": [best_features],
                "Initial Performance": [initial_performance],

                "Read Data Time": [read_data_time],
                "Generate Feature Time": [generate_feature_time],
                "Select Top Feature Time": [select_top_feature_time],
                "Optimize Time": [optimize_time],
                "Final Train Time": [final_train_time],
                "Total Time": [total_time]
            }

            dir_path = f'./datasets/ours/results_select_new_data_weights_new_{m}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = f'{dir_path}/{sys_name}_AllNumeric_train_{sample_size}.csv'

            #  pandas DataFrame
            df = pd.DataFrame(data_dict)

            # 
            df.to_csv(file_path, index=False)
        result_dict = {
            "Type": ["MRE","Read Data Time", "Generate Feature Time",
                     "Select Top Feature Time", "Optimize Time", "Final Train Time","Total Time"],
            "Sample size": [sample_size] * 7,
            "Mean": [np.mean(mre_list), 
                     np.mean(read_data_time_list), 
                     np.mean(generate_feature_time_list),
                     np.mean(select_top_feature_time_list), 
                     np.mean(optimize_time_list),
                     np.mean(final_train_time_list),
                     np.mean(total_time_list)],
            "Margin":[
                1.96 * np.sqrt(np.var(mre_list, ddof=1)) / np.sqrt(len(mre_list)),
                1.96 * np.sqrt(np.var(read_data_time_list, ddof=1)) / np.sqrt(len(read_data_time_list)),
                1.96 * np.sqrt(np.var(generate_feature_time_list, ddof=1)) / np.sqrt(len(generate_feature_time_list)),
                1.96 * np.sqrt(np.var(select_top_feature_time_list, ddof=1)) / np.sqrt(len(select_top_feature_time_list)),
                1.96 * np.sqrt(np.var(optimize_time_list, ddof=1)) / np.sqrt(len(optimize_time_list)),
                1.96 * np.sqrt(np.var(final_train_time_list, ddof=1)) / np.sqrt(len(final_train_time_list)),
                1.96 * np.sqrt(np.var(total_time_list, ddof=1)) / np.sqrt(len(total_time_list))
            ]
        }
        result_df = pd.DataFrame(result_dict)
        result_sys.append(result_df)

        print(f'Finish experimenting for system {sys_name} with sample size {sample_size}.')
        print(f'Mean prediction relative error (%) is: MRE: {result_dict["Mean"][0]:.2f},Total Time: {result_dict["Mean"][5]:.2f}')
        final_result_path = f'./datasets/ours/final_results'
        if not os.path.exists(final_result_path):
            os.makedirs(final_result_path)

        filename = f'{final_result_path}/result_{sys_name}_{sample_size}.csv'
        result_df.to_csv(filename, index=False)
        print(f'Save the statistics to file {filename} ...')

result_sys_combined = pd.concat(result_sys, ignore_index=True)
save_path = './datasets/ours/final_results'

if not os.path.exists(save_path):
    os.makedirs(save_path)
filename =f'{save_path}/combined_result.csv'
result_sys_combined.to_csv(filename, index=False)        
