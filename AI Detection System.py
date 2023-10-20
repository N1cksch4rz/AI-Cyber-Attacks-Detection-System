import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import get_file

def download_and_read_data():
    try:
        path = get_file('kddcup.data.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
    except Exception as e:
        print('Error downloading the dataset:', e)
        raise

    print('=' * 40)
    print(path)
    print('=' * 40)

    dataset_KDDCUP99 = pd.read_csv(path, header=None)

    print('=' * 40)
    print("Read {} rows.".format(len(dataset_KDDCUP99)))
    print('=' * 40)

    dataset_KDDCUP99.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                                "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                                "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
                                "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                                "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                                "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    return dataset_KDDCUP99

def preprocess_data(dataset):
    input_columns = dataset.columns[1:-1]
    target_column = 'label'
    numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()[:-1]

    scaler = MinMaxScaler()
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

    le = LabelEncoder()
    dataset['label'] = le.fit_transform(dataset['label'])
    dataset['protocol_type'] = le.fit_transform(dataset['protocol_type'])
    dataset['service'] = le.fit_transform(dataset['service'])
    dataset['flag'] = le.fit_transform(dataset['flag'])

    return dataset, input_columns, target_column

def split_data(dataset, test_size=0.3):
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)
    return train_dataset, test_dataset

def select_features(train_inputs, train_targets):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_inputs, train_targets)
    selected_features = train_inputs.columns[model.feature_importances_ > 0.01]
    return selected_features

def train_and_evaluate_models(train_inputs, train_targets, test_inputs, test_targets, selected_features):
    # Random Forest classifier
    random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
    random_forest_model.fit(train_inputs[selected_features], train_targets)
    predictions_random_forest = random_forest_model.predict(test_inputs[selected_features])
    score_random_forest = accuracy_score(test_targets, predictions_random_forest)

    print('-' * 30)
    print("The result of accuracy of Random Forests is: ", score_random_forest)
    print('-' * 30)

    # Decision tree classifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(train_inputs[selected_features], train_targets)
    predictions_decision_tree = decision_tree_model.predict(test_inputs[selected_features])
    score_decision_tree = accuracy_score(test_targets, predictions_decision_tree)

    print('-' * 30)
    print("The result of accuracy of Decision Tree is: ", score_decision_tree)
    print('-' * 30)

if __name__ == '__main__':
    dataset = download_and_read_data()
    dataset, input_columns, target_column = preprocess_data(dataset)
    train_dataset, test_dataset = split_data(dataset)
    selected_features = select_features(train_dataset[input_columns], train_dataset[target_column])
    train_and_evaluate_models(train_dataset[input_columns], train_dataset[target_column],
                             test_dataset[input_columns], test_dataset[target_column], selected_features)
