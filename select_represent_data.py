# -*- coding: utf-8 -*-
import time
import numpy as np
import represent_data  
import pandas as pd
import os

from multiprocessing import cpu_count

cpu_num = cpu_count()
cpu_use = 8
cur_pid = os.getpid()
os.sched_setaffinity(cur_pid, list(range(cpu_num))[:cpu_use])
print(f"set the max number of cpu used to {cpu_use}") 

def system_samplesize(sys_name):
    if sys_name == 'x264':
        num_representative_points = np.multiply(16, [1, 2, 4, 6])  
    elif sys_name == 'lrzip':
        num_representative_points = np.multiply(19, [1, 2, 4, 6]) 
    elif sys_name == 'vp9':
        num_representative_points = np.multiply(41, [1, 2, 4, 6])  
    elif sys_name == 'polly':
        num_representative_points = np.multiply(39, [1, 2, 4, 6])  
    elif sys_name == 'Dune':
        num_representative_points = np.asarray([49, 78, 384, 600])
    elif sys_name == 'hipacc':
        num_representative_points = np.asarray([261, 528, 736, 1281]) 
    elif sys_name == 'hsmgp':
        num_representative_points = np.asarray([77, 173, 384, 480])  
    elif sys_name == 'javagc':
        num_representative_points = np.asarray([855, 2571, 3032, 5312])  
    elif sys_name == 'sac':
        num_representative_points = np.asarray([2060, 2295, 2499, 3261]) 
    else:
        raise AssertionError("Unexpected value of 'sys_name'!")

    return num_representative_points

np.random.seed(42)

sys_names = ['x264', 'lrzip', 'vp9', 'polly', 'Dune', 'hipacc', 'hsmgp', 'javagc', 'sac']

type_list = []
nums_points = []
time_list = []

for sys_name in sys_names:
    file_path = f'./datasets/Raw data/{sys_name}_AllNumeric_train.csv'
    
    num_representative_points = system_samplesize(sys_name)

    print(f"Process the dataset:{file_path}")

    for num_representative_point in num_representative_points:
        try:
            start_time = time.time()

            save_dir = './datasets/select_represent_data/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = save_dir+f'{sys_name}_AllNumeric_train_{num_representative_point}.csv'
            
            print(f"save_path: {save_path}")

            represent_data.process_and_select_samples(file_path, int(num_representative_point), save_path)
            end_time = time.time()
            print(f" processing {sys_name} representative points {num_representative_point} time: {end_time-start_time:.2f}s")

            end_time = time.time()
            type_list.append(sys_name)
            nums_points.append(num_representative_point)
            time_list.append(end_time-start_time)
        except Exception as e:
            print(f" {sys_name} error: {e}")
        
result_df = pd.DataFrame({'sys_name': type_list, 'num_representative_point': nums_points, 'time': time_list})
save_dir = './datasets/select_represent_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = f'{save_dir}/re_select_represent_data_result.csv'
result_df.to_csv(save_path, index=False)