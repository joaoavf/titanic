# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
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


def load_df(filename):
    """
    Loads train_df
    :return: Train DataFrame
    :rtype: pandas DataFrame
    """
    final_path = os.path.join(get_data_path(), 'raw/' + filename)
    return pd.read_csv(final_path)


def save_df(df, filename):
    """
    Saves DataFrame in hdf5 with name 'train.hdf5'

    :param df: DataFrame to be Saved
    """
    final_path = os.path.join(get_data_path(), 'processed')
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    final_path = os.path.join(final_path, filename)

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


def embarked_onehot(df):
    """
    Transforms embarked into OneHotEncoder columns.

    :param df: Input DataFrame that contains 'Embarked' column

    :return: df
    :rtype: pandas DataFrame
    """

    # TODO make it foolproof if number of categories in train is different from test and etc
    # TODO better treat NaNs

    # Does basically the same thing as OneHotEncoder
    embarked_df = pd.get_dummies(df.Embarked)

    # Merges DataFrames (OneHotEncoder and DataFrame being processed)
    ndf = pd.concat([df, embarked_df], axis=1)

    fa = FactorAnalysis(n_components=1)
    y = fa.fit_transform(embarked_df.values)
    ndf['embarked_fa'] = y

    # Returns transformed DataFrame
    return ndf


def add_name_length(df):
    df['name_len'] = df['Name'].str.len()
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
        valid_titles = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.']
        df['Title'] = df.apply(merge_title_small_sample, axis=1, args=(valid_titles,))

    # Returns df
    return df


def merge_title_small_sample(df_line, valid_titles):
    """
    Function to be used by DataFrame.apply(axis=1), that goes line by line checking if value for 'Title' column is part
    of a list of values that has a small sample. This list is given by param 'small_sample_series'.

    :param df_line: DataFrame Line
    :param small_sample_series: Series that contains values that have low count.

    :return: Title
    """
    if df_line['Title'] not in valid_titles:
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
    # Gets Title Category encoded
    title_cat_encoded = pd.Categorical(df.Title, ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.', 'Other']).codes

    # Instantiates OneHotEncoder
    onehotencoder = OneHotEncoder()

    # TODO fix issue with mismatching categories between test and train, use modified version of test dataset
    # Transform 1d np.array into 2d np.array
    title_cat_encoded = np.array([title_cat_encoded])

    # Gets OneHot np.array. '.fit_transform' outputs scipy sparse matrix, needs '.toarray' to get np.array
    title_cat_onehot = onehotencoder.fit_transform(title_cat_encoded.reshape(-1, 1)).toarray()

    # Creates OneHot DataFrame
    onehot_df = pd.DataFrame(title_cat_onehot, index=df.index, columns=df["Title"].value_counts().index)

    # Drops Title columns
    df.drop(columns=['Title'], inplace=True)

    # Concatenates DataFrames laterally
    ndf = pd.concat([df, onehot_df], axis=1)

    # Creates FactorAnalysis columns
    fa = FactorAnalysis(n_components=2)
    y = fa.fit_transform(onehot_df.values)
    ndf[['title_fa1', 'title_fa2']] = pd.DataFrame(y)

    # Returns Transformed DataFrame
    return ndf


def avg_fare(df):
    df['avg_fare'] = df.groupby('Ticket')['Fare'].transform('mean')
    return df


def cabin_first_letter(df):
    # Does basically the same thing as OneHotEncoder
    cabin_df = pd.get_dummies(df['Cabin'].str[0], prefix='cabin_first_letter', dummy_na=True)
    cabin_df['cabin_first_letter_other'] = 0

    columns = ['cabin_first_letter_A', 'cabin_first_letter_B', 'cabin_first_letter_C', 'cabin_first_letter_D',
               'cabin_first_letter_E', 'cabin_first_letter_F', 'cabin_first_letter_G', 'cabin_first_letter_T',
               'cabin_first_letter_nan', 'cabin_first_letter_other']

    # Code below harmonizes OneHotEncoder between train and test sets
    for column in cabin_df.columns:
        if column not in columns:
            cabin_df['cabin_first_letter_other'] = cabin_df['cabin_first_letter_other'] + cabin_df[column]
            cabin_df.drop(column=column)

    for column in columns:
        if column not in cabin_df.columns:
            cabin_df[column] = 0

    cabin_df = cabin_df[columns]
    fa = FactorAnalysis(n_components=1)
    y = fa.fit_transform(cabin_df.values)

    # Merges DataFrames (OneHotEncoder and DataFrame being processed)
    ndf = pd.concat([df, cabin_df], axis=1)
    ndf['cabin_fa'] = y

    return ndf


def has_cabin(df):
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    return df


def age_fillna(df):
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    df['Age'] = df['Age'].astype(int)
    return df


def cat_age(df):
    df['CategoricalAge'] = pd.to_numeric(pd.cut(df['Age'], 5, labels=(0, 1, 2, 3, 4)))
    return df


def remove_age_na(df):
    df['Age_na'] = np.where(df['Age'].isnull(), True, False)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    return df


def sibsparch_fa(df):
    sibsparch_df = df[['SibSp', 'Parch']]
    fa = FactorAnalysis(n_components=1)
    y = fa.fit_transform(sibsparch_df.values)

    df['sibsparch_fa'] = y

    return df


def clean_df(df):
    """
    Cleans DataFrame

    :param df: Input DataFrame

    :return: Cleaned DataFrame
    :rtype: pandas DataFrame
    """

    # FillsNA if any (better to fill before with more robust functions)
    # TODO make better fillNA functions
    df = df.fillna(df.mean())

    # Remove String dtypes
    df = df.select_dtypes(exclude=['object'])

    # Passenger Id doesn't seem like noise, I tried to remove it, but it actually worsened my prediction
    # df.drop(columns='PassengerId', inplace=True)

    # Returns clean DF
    return df


def one_hot_encoding_fixed_columns(pandas_series, fixed_columns):
    # Creates complete fixed columns list (with nan and 'other')
    fixed_columns = list(fixed_columns)
    fixed_columns.extend([np.nan, 'other'])

    # Get dummies dataset
    ohe_df = pd.get_dummies(pandas_series, dummy_na=True)

    # Create blank 'other' column
    ohe_df['other'] = 0

    # Check if columns created by get_dummies() are in 'fixed_columns' list.
    for column in ohe_df.columns:

        if column not in fixed_columns:
            # If not in 'fixed_columns', transforms exceeding column into 'other'.
            ohe_df['other'] = ohe_df['other'] + ohe_df[column]
            ohe_df.drop(columns=[column])

    # Check if elements in 'fixed_columns' are in the df generated by get_dummies()
    for column in fixed_columns:

        if column not in ohe_df.columns:
            # If the element is not present, create a new column with all values set to 0.
            ohe_df['column'] = 0

    # Reorders columns according to fixed columns
    ohe_df = ohe_df[fixed_columns]

    return ohe_df


def routine(df):
    """
    Routine to transform DataFrames.

    :param df: input

    :return: Transformed DataFrame
    :rtype: pandas DataFrame
    """
    df = remove_age_na(df)
    df = sex_to_int(df)
    df = add_name_length(df)
    df = add_title(df)
    df = one_hot_encoder_title(df)
    df = embarked_onehot(df)
    df = avg_fare(df)
    df = cabin_first_letter(df)
    df = has_cabin(df)
    df = age_fillna(df)
    df = cat_age(df)
    df = sibsparch_fa(df)

    # Clean DF, remove object columns and FillNA
    df = clean_df(df)

    return df


def open_run_save(name):
    # Load DataFrame
    df = load_df(name + '.csv')

    # Transforms DataFrame
    df = routine(df)

    # Saves Transformed (Processed) DataFrame to disk
    save_df(df, name + '.hdf5')


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

    open_run_save('train')
    open_run_save('test')

# Creates a ´columns´ list with all valid categories.
