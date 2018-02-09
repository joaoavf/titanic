import os
import pandas as pd


def get_processed_train_set():
    """
    Returns processed train data.

    :return: Processed train data
    :rtype: pandas DataFrame
    """
    final_path = get_proc_data_path('train.hdf5')
    return pd.read_hdf(final_path)


def get_processed_test_set():
    """
    Returns processed train data.

    :return: Processed train data
    :rtype: pandas DataFrame
    """
    final_path = get_proc_data_path('test.hdf5')
    return pd.read_hdf(final_path)


def get_proc_data_path(filename):
    """
    Returns processed train data path.

    :return: Processed train data path
    :rtype: String
    """

    # Get current absolute path
    absolute_path = os.path.abspath(__file__)

    # Get parent path
    project_root = os.path.dirname(os.path.dirname(absolute_path))

    # Get final path (absolute path for '../../data/raw/'
    final_path = os.path.join(project_root, 'data/processed/' + filename)

    return final_path


def get_test_data():
    """
    Get Test Data.

    :return: Test data
    :rtype: pandas DataFrame
    """
    final_path = get_test_data_path()
    return pd.read_csv(final_path)


def get_test_data_path():
    """
    Get Test Data Path.

    :return: Test data path
    :rtype: String
    """

    # Get current absolute path
    absolute_path = os.path.abspath(__file__)

    # Get parent path
    project_root = os.path.dirname(os.path.dirname(absolute_path))

    # Get final path (absolute path for '../../data/raw/'
    final_path = os.path.join(project_root, 'data/raw/test.csv')

    return final_path
