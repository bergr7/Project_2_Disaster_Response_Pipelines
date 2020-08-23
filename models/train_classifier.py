import sys

import re
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """ It loads X, y and df from an existing database.

    INPUT:
    database_filepath: Name of the sqlite database. e.g. 'sqlite:///InsertDatabaseName.db'

    OUTPUT:
    X: Messages feature stored in df (pandas series)
    y: Categories or classes stored in df (pandas dataframe)
    category_names: List with the names of all the categories
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = list(df.columns[4:])
    return X, y, category_names


def tokenize(text):
    """ It takes an array with the messages and returns an array with the tokenized and lemmatized text.

    INPUT:
    text: Messages in X (array-like object, string)

    OUTPUT:
    lemmed_text: Tokenized and lemmed messages in X (array-like object, string)
    """

    # Normalize text to all lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    lower_case_text = text.lower()

    # Tokenize text
    tokens = word_tokenize(lower_case_text)

    # Remove stopwords
    words = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmed_text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    return lemmed_text


def build_model():
    """It performs a Grid Search of the best hyper-parameters for KNeighborsClassifier
    based on cross-validations scores.
    INPUT:
    None

    OUTPUT:
    cv: Grid search based on cross-validation scores
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier(random_state=42, class_weight='balanced', n_jobs=-1)))
    ])

    parameters = {
        'tfidf__norm': ('l1', 'l2'),
        'tfidf__smooth_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'clf__estimator__criterion': ('gini', 'entropy'),
        'clf__estimator__splitter': ('best', 'random'),
        'clf__estimator__max_depth': [5, 7, 10],
        'clf__estimator__max_leaf_nodes': [1, 3, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """It takes the trained model and test its performance. It displays the Precision, Recall and F1 score for
    each category.

    INPUT:
    model: Trained model
    X_test: Messages stored in the test set
    Y_test: Relevant categories stored in the test set
    category_names: List of string with the category names

    OUTPUT:
    None
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Display classification report
    for i, name in zip(range(y_pred.shape[1]), category_names):
        print('Category:', name, '\n', classification_report(y_test[:, i], y_pred[:, i],
                                                             target_names=['Negative', 'Positive'], labels=[0, 1]))


def save_model(model, model_filepath):
    """ It saves the trained model as a pickle in the specified directory.

    INPUT:
    model: Trained model
    model_filepath: Full file path, including name of the pickle file. e.g. .../models/KNeighborsClassifier.pkl

    OUTPUT:
    None
    """
    # Save the model to a folder
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
