import os
import pandas as pd


def get_processed_data():
    final_path = get_proc_data_path()
    return pd.read_hdf(final_path)


def get_proc_data_path():
    # Get current absolute path
    absolute_path = os.path.abspath(__file__)

    # Get parent path
    project_root = os.path.dirname(os.path.dirname(absolute_path))

    # Get final path (absolute path for '../../data/raw/'
    final_path = os.path.join(project_root, 'data/processed/train.hdf5')

    return final_path


def get_test_data():
    final_path = get_test_data_path()
    return pd.read_csv(final_path)


def get_test_data_path():
    # Get current absolute path
    absolute_path = os.path.abspath(__file__)

    # Get parent path
    project_root = os.path.dirname(os.path.dirname(absolute_path))

    # Get final path (absolute path for '../../data/raw/'
    final_path = os.path.join(project_root, 'data/raw/test.csv')

    return final_path
