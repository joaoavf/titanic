# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


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
    final_path = os.path.join(get_data_path(), 'raw/train.csv')
    return pd.read_csv(final_path)


def save_df(df):
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
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    return df


def add_title(df, merge_small_sample=True):
    """
    Add Title to DataFrame (Mr., Miss., etc)

    :param df: Input Titanic DataFrame (with Name column)
    :return: df with ['Title'] column
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
    if df_line['Title'] in small_sample_series:
        return 'Other'
    else:
        return df_line['Title']


def one_hot_encoder_title(df):
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


def add_infant_status(df):
    """
        Add 'Male Infant' and 'Female Infant' categories

        :param df: Input Titanic DataFrame (with Name column)
        :return: df with [['Male Infant','Female Infant']]
        """
    if 'Survived' not in df.columns:
        df['Male Infant'] = df['Age'] < 12
        df['Female Infant'] = df['Age'] < 63
        return df

    # Get Maximum age to max out for loop
    max_age = df['Age'].max()

    # Get Survival rates by gender
    male_survival_mean = df[df['Sex'] == 'male']['Survived'].mean()
    female_survival_mean = df[df['Sex'] == 'female']['Survived'].mean()

    # Instantiates variables
    global_mean_male, global_mean_female, optimal_male_age, optimal_female_age = None, None, 0, 0

    # Test all possible age divisions for infants to check out which is the threshold that is better than average
    for age in range(0, int(max_age) + 1):

        # Instantiates 'temp_df'
        temp_df = df[df['Age'] < age]

        # TODO checkout warning issued here
        # Instantiates 'male_df'
        male_df = temp_df[(temp_df['Sex'] == 'male') & (df['Age'] > optimal_male_age)]
        male_count = male_df.count().max()

        # Runs code only if sample is not zero
        if male_count > 0:
            infant_mean_male = male_df['Survived'].mean()

            # If average for this age bracket is better, change optimal age
            if infant_mean_male > male_survival_mean:
                optimal_male_age = age

        # Instantiates 'female_df'
        female_df = temp_df[(temp_df['Sex'] == 'female') & (df['Age'] > optimal_female_age)]
        female_count = female_df.count().max()

        # Runs code only if sample is not zero
        if female_count > 0:
            infant_mean_female = female_df['Survived'].mean()

            # If average for this age bracket is better, change optimal age
            if infant_mean_female > female_survival_mean:
                print(age, infant_mean_female)
                optimal_female_age = age

    # Sets new columns to df
    df['Male Infant'] = df['Age'] < optimal_male_age
    df['Female Infant'] = df['Age'] < optimal_female_age

    # Returns df
    return df


def clean_df(df):
    df = df.fillna(df.mean())
    df = df.select_dtypes(exclude=['object'])
    return df

def routine(df):
    df = add_infant_status(df)
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

    df = load_df()

    df = routine(df)

    save_df(df)

    print(df.dtypes)
