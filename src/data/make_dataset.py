# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
# TODO learn to use click
def main(input_filepath='x', output_filepath='y'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def load_df():
    """
    Loads train_df
    :return: Train DataFrame
    :rtype: pandas DataFrame
    """
    final_path = os.path.join(get_data_path(), 'raw/train.csv')
    return pd.read_csv(final_path)


def save_df(df):
    """
    Saves DataFrame in hdf5 with name 'train.hdf5'

    :param df: DataFrame to be Saved
    """
    final_path = os.path.join(get_data_path(), 'processed')
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    final_path = os.path.join(final_path, 'train.hdf5')

    df.to_hdf(final_path, 'processed_data')


def get_data_path():
    # Get current absolute path
    absolute_path = os.path.abspath(__file__)

    # Get parent path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(absolute_path)))

    # Get final path (absolute path for '../../data/raw/'
    final_path = os.path.join(project_root, 'data/')

    return final_path


def sex_to_int(df):
    """
    Converts 'Sex' string to int.

    :param df: Input DF

    :return: Transformed DF
    :rtype: pandas DataFrame
    """
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    return df


def add_title(df, merge_small_sample=True):
    """
    Add Title to DataFrame (Mr., Miss., etc)

    :param df: Input Titanic DataFrame (with Name column)

    :return: df with ['Title'] column
    :rtype: pandas DataFrame
    """

    # Creates 'names' pd.Series
    names = df['Name']

    # Split 'names' by ', ' and get the latter value
    proc_names = names.str.split(', ').str[-1]

    # Sets ['Title'] by splitting 'proc_names' by ' ' and getting first value
    df['Title'] = proc_names.str.split(' ').str[0]

    if merge_small_sample:
        small_sample_series = df['Title'].value_counts()[df['Title'].value_counts() < 3]
        df['Title'] = df.apply(merge_title_small_sample, axis=1, args=(small_sample_series,))

    # Returns df
    return df


def merge_title_small_sample(df_line, small_sample_series):
    """
    Function to be used by DataFrame.apply(axis=1), that goes line by line checking if value for 'Title' column is part
    of a list of values that has a small sample. This list is given by param 'small_sample_series'.

    :param df_line: DataFrame Line
    :param small_sample_series: Series that contains values that have low count.

    :return: Title
    """
    if df_line['Title'] in small_sample_series:
        return 'Other'
    else:
        return df_line['Title']


def one_hot_encoder_title(df):
    """
    Transform title columns in a  one hot encoder column.

    :param df: Input DataFrame

    :return: Transformed DataFrame
    :rtype: pandas DataFrame
    """
    encoder = LabelEncoder()
    title_cat = df["Title"]
    df['Title'] = encoder.fit_transform(title_cat)

    return df

    """
    onehotencoder = OneHotEncoder()
    housing_cat_onehot = onehotencoder.fit_transform(title_cat_encoded.reshape(-1, 1))
    # TODO check if column names is right
    onehot_df = pd.DataFrame(housing_cat_onehot, index=df.index, columns=df["Title"].value_counts().index)

    df.drop(columns=['Title'])
    ndf = pd.concat([df, onehot_df])
    

    return ndf"""


def clean_df(df):
    """
    Cleans DataFrame

    :param df: Input DataFrame

    :return: Cleaned DataFrame
    :rytpe: pandas DataFrame
    """

    # FillsNA if any (better to fill before with more robust functions)
    # TODO make better fillNA functions
    df = df.fillna(df.mean())

    # Remove String dtypes
    df = df.select_dtypes(exclude=['object'])

    # Returns clean DF
    return df


def routine(df):
    """
    Routine to transform DataFrames.

    :param df: input

    :return: Transformed DataFrame
    :rtype: pandas DataFrame
    """
    df = sex_to_int(df)
    df = add_title(df)
    df = one_hot_encoder_title(df)
    df = clean_df(df)
    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    # TODO understand 2 lines below
    # load_dotenv(find_dotenv())
    # main()

    # Load DataFrame
    df = load_df()

    # Transforms DataFrame
    df = routine(df)

    # Saves Transformed (Processed) DataFrame to disk
    save_df(df)
