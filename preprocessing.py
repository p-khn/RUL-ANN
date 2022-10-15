import os
import sys
from typing import List, Tuple
import pandas as pd
import re

# Get datasets path list
def get_datasets_path(path="./CMAPSSData", dataset_id="FD004") -> List[str]:
    path_list = list()
    for root, dirs, files in os.walk("./CMAPSSData"):
        for file in files:
            try:
                if file.endswith(f'{dataset_id}.txt'):
                   path_list.append(os.path.join(root, file))
            except FileNotFoundError:
                sys.exit('File or directory does not exist!!!')

    return path_list

# Get datasets
def get_data(path_list: List[str]) -> Tuple:
    columns_name = [
    'unit_number', 'time_in_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i+1}' for i in range(0,21)]
    
    for path in path_list:
        if re.search(r"RUL",path):
            y_test = pd.read_csv(path, sep='\s+', header=None).unstack()
        elif re.search(r"test", path):
            X_test = pd.read_csv(path, sep='\s+', names=columns_name, header=None)
        elif re.search(r"train", path):
            X_train = pd.read_csv(path, sep='\s+', names=columns_name, header=None)
        else:
            print("Not valid, please check the dataset path!!!")
    
    return X_train, X_test, y_test

# Calculate RUL for train data
def calculate_rul(data:pd.DataFrame, index_names: List[str]) -> pd.Series:
    max_cycle_of_units_df = data.groupby(by=index_names[0])[index_names[1]].max().to_frame('max_of_cycles')
    merged_df = data.merge(max_cycle_of_units_df, right_index= True, left_on=index_names[0])
    rul = merged_df['max_of_cycles'] - merged_df[index_names[1]]

    return rul

# Get the last instance grouped by unit number for test dataset
def get_last(data:pd.DataFrame, index_name: str) -> pd.DataFrame:
    data = data.groupby(by=index_name).last().reset_index()

    return data

