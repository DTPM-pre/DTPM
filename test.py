
import time
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib  # For saving and loading the model
import pandas as pd
import ast  # 
import os


# 
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


def load_trained_model(filepath='best_model.pkl'):
    """ """
    return joblib.load(filepath)

def save_trained_model(model, filepath='best_model.pkl'):
    """save"""
    joblib.dump(model, filepath)

def test_best_model(test_features, test_targets, xgb_model_path, rf_model_path, weighted_path):
    """model path"""
    xgb_model = load_trained_model(xgb_model_path)
    rf_model = load_trained_model(rf_model_path)

    
    # prediction
    xgb_predictions = xgb_model.predict(test_features.values)
    rf_predictions = rf_model.predict(test_features.values)  # 

    weights = np.load(weighted_path)
    w1, w2 = weights[0], weights[1]

    predictions = w1 * xgb_predictions + w2 * rf_predictions
    predictions[predictions < 0] = 0  # 

    non_zero_y = np.where(test_targets != 0, test_targets, np.nan)  #

    # MRE
    mre = np.mean(np.abs((non_zero_y - predictions) / non_zero_y)) * 100

    return mre, test_features, test_targets, predictions
sys_names = ['x264', 'lrzip', 'vp9', 'polly', 'Dune', 'hipacc', 'hsmgp', 'javagc', 'sac']

n_exp=30
result_sys = []
for sys_name in sys_names:
    samplesizes,_=system_samplesize(sys_name)
    for sample_size in samplesizes:
        final_mre_list = []
        test_time_list = []
        print("sys_name:",sys_name)
        print("sample_size:",sample_size)
        for m in range(1, n_exp+1):
            print(f"Experiment {m} ...")
            #
            new_test_file_path=f'./datasets/transformed_test/{sys_name}_AllNumeric_test.csv'
            new_test_data_df=pd.read_csv(new_test_file_path)
            new_test_features=new_test_data_df.iloc[:,:-1]
            new_test_targets=new_test_data_df.iloc[:,-1]

            # 
            best_feature_combination_path = f'./datasets/ours/results_select_new_data_weights_new_{m}/{sys_name}_AllNumeric_train_{sample_size}.csv'

            best_feature_combination_df=pd.read_csv(best_feature_combination_path)
            best_feature_combination = best_feature_combination_df.iloc[0, 3]  # 
            print('best_feature_combination:',best_feature_combination)

            try:
                best_feature_combination = ast.literal_eval(best_feature_combination)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing best_feature_combination: {e}")
                continue
            
            new_test_features=new_test_features[best_feature_combination]

            final_xgb_model_path = f'./model/ours/models/models_select_weights_new_{m}/{sys_name}_AllNumeric_train_{sample_size}_final_xgb_model.pkl'
            final_rf_model_path = f'./model/ours/models/models_select_weights_new_{m}/{sys_name}_AllNumeric_train_{sample_size}_final_rf_model.pkl'
            weighted_path = f'./model/ours/weights/xgb_rf_models_weights_new_{m}/{sys_name}_{sample_size}_best_weights.npy'

            # test
            start_time = time.time()
            final_mre, new_test_features, new_test_targets, new_predictions_final = test_best_model(new_test_features, new_test_targets, final_xgb_model_path, final_rf_model_path,weighted_path)
            test_time = time.time() - start_time
            print('final_mre:',final_mre)
            print('test_time:',test_time)

            data_dict = {
                "sys_name": [sys_name],
                "sample_size": [sample_size],
                "length of best_feature_combination": [len(best_feature_combination)],
                "final_mre": [final_mre],
                "test_time": [test_time]
            }

            dir_path=f'./test_result/ours/results_new_{m}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path=f'{dir_path}/{sys_name}_AllNumeric_train_{sample_size}.csv'
            
            df = pd.DataFrame(data_dict)
            df.to_csv(file_path, index=False)
            final_mre_list.append(final_mre)
            test_time_list.append(test_time)
        
        result_dict = {
            "System": [sys_name] * 2,
            "Sample size": [sample_size] * 2,
            "Type": ["Final_MRE", "Test Time"],
            "Mean": [
                np.mean(final_mre_list),
                np.mean(test_time_list)
            ],
            "Margin": [
                1.96 * np.sqrt(np.var(final_mre_list, ddof=1)) / np.sqrt(len(final_mre_list)),
                1.96 * np.sqrt(np.var(test_time_list, ddof=1)) / np.sqrt(len(test_time_list))
            ]
        }

        result_df = pd.DataFrame(result_dict)
        result_sys.append(result_df)

        print(f'Finish experimenting for system {sys_name} with sample size {sample_size}.')
        print(f'Mean prediction relative error (%) is: Final_MRE: {result_dict["Mean"][0]:.2f}, Test Time: {result_dict["Mean"][1]:.2f}')
        
        final_results_path = f'./test_result/ours/final_result'

        if not os.path.exists(final_results_path):
            os.makedirs(final_results_path)

        filename = f'{final_results_path}/result_{sys_name}_{sample_size}.csv'
        result_df.to_csv(filename, index=False)
        print(f'Save the statistics to file {filename} ...')
        
result_sys_combined = pd.concat(result_sys, ignore_index=True)
save_path = './test_result/ours/final_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
filename =f'{save_path}/combined_result.csv'
result_sys_combined.to_csv(filename, index=False)        
