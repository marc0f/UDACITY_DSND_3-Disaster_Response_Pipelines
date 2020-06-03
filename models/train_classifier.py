import sys
import os
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from joblib import dump


from utils.extra import MyTfidfTransformer, clean_one_class_category

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

num_cpus = os.cpu_count() - 1
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """ load data from database """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():

    # compose the processing pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        # ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('tfidf', MyTfidfTransformer(smooth_idf=False)),
        # ('cls', MultiOutputClassifier(RandomForestClassifier(), n_jobs=num_cpus))
        ('cls', MultiOutputClassifier(SVC(), n_jobs=num_cpus))
    ])

    # full params
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'cls__estimator__n_estimators': [10, 50, 100, 200],
        'cls__estimator__min_samples_split': [2, 3, 4]
    }

    # reduced params
    # best_parameters = {
    #     'cls__estimator__kernel': ['linear', 'rgf'],
    #     'tfidf__use_idf': (True, False),
    #     'vect__max_df': 0.5,
    #     'vect__max_features': 10000,
    #     'vect__ngram_range': (1, 2)}

    # instantiate search grid
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv
    # return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    # use model to predict output given the test data
    Y_pred = model.predict(X_test)

    # convert prediction and expected outputs into dataframes
    y_pred_df = pd.DataFrame(Y_pred)
    y_pred_df.columns = category_names
    y_test_df = pd.DataFrame(Y_test)
    y_test_df.columns = category_names

    # get reports of the performance (accuracy, f1-score, precision, recall) for each category
    # reports = dict()
    for col in category_names:
        print(col)
        print(classification_report(y_test_df[col], y_pred_df[col]))
        # reports[col] = classification_report(y_test_df[col], y_pred_df[col], output_dict=True)

    # print best params when search grid is performed
    if isinstance(model, GridSearchCV):
        print("Best params:")
        print(model.best_params_)


def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # add step to remove non used classes. Avoid error in Classificators that do not support unique classes (e.g., SVC)
        Y, category_names = clean_one_class_category(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
