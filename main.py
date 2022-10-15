
import preprocessing as prp
from ann import Ann

# Preprocessing
from sklearn.preprocessing import MinMaxScaler


# Get datasets path_list selected by dataset_id
path_list = prp.get_datasets_path(path="./CMAPSSData", dataset_id="FD004")

# Get X_train, X_test, y_test
X_train, X_test, y_test = prp.get_data(path_list)

# Calculate the y_train (RUL) for train data
y_train = prp.calculate_rul(X_train, ['unit_number','time_in_cycles'])

# Get the last instance grouped by unit number for test dataset
X_test = prp.get_last(X_test, 'unit_number')

# Drop extra columns
columns_to_drop = [
    'unit_number', 'time_in_cycles',
    'op_setting_1', 'op_setting_2',
    'op_setting_3']

X_train.drop(columns_to_drop, axis=1, inplace=True)
X_test.drop(columns_to_drop, axis=1, inplace=True)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate the ANN model
ann = Ann()
_, score = ann.ann_model(X_train, X_test, y_train, y_test, epochs=3000, batch_size=50)


print(f'\nThe MAE score for ANN model: {score}')