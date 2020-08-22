import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ It converts messages and categories .csv files into DataFrames, merged them together
    and returns the merged DataFrame.

    INPUT:
    messages_filepath -  messages.csv filepath (str)
    categories_filepath - categories.csv filepath (str)

    OUTPUT:
    df - Pandas DataFrame containing messages DataFrame merged with categories DataFrame on 'id' column.
    """

    # read .csv files and convert into DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # drop duplicates
    messages = messages.drop_duplicates('id')
    categories = categories.drop_duplicates('id')

    # merge DataFrames on 'id' column
    df = messages.merge(categories, left_on='id', right_on='id', how='right')

    return df


def clean_data(df):
    """ It takes df returned by load_data, split the split the 'categories' column into separate category column
    assigning 1's if message is relevant to the category and clean the data removing duplicates.

    INPUT:
    df - Pandas DataFrame containing messages DataFrame merged with categories DataFrame on 'id' column.

    OUTPUT:
    df_cleaned - Pandas DataFrame containing cleaned data and the 'categories' columns split into separate category columns with
    numerical values representing the occurrence.
    """

    # create a DataFrame of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories DataFrame
    row = categories[:1]
    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')

    # drop 'categories' from original df
    df = df.drop(['categories'], axis=1)

    # concatenate the original DataFrame with the new `categories` DataFrame
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df_cleaned = df.drop_duplicates('id', keep='first')

    return df_cleaned


def save_data(df, database_filename):
    """ It saves df_cleaned (the clean dataset) into an sqlite database.

    INPUT:
    df - Pandas DataFrame containing clean data.
    database_filename - Name of the database (String)

    OUTPUT:
    None
    """

    # create engine
    engine = create_engine('sqlite:///{}.db'.format(database_filename))

    # insert df into table
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
