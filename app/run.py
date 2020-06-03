import json
import plotly
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

from utils.extra import MyTfidfTransformer, clean_one_class_category


app = Flask(__name__)


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)
_, category_names = clean_one_class_category(df.drop(columns=['id', 'message', 'original', 'genre']))

# load model
model = load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    avg_message_tokens = df['message'].iloc[0:10].apply(tokenize).str.len().mean()
    df_categories = df.drop(columns=['id', 'message', 'original', 'genre'])
    avg_categories_per_message = df_categories.sum(axis=1).mean()

    sort_categories_series = df_categories.sum(axis=0).sort_values(ascending=False)
    # less_common_categories_series = df_categories.sum(axis=0).sort_values()[:3]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=['Avg. tokens', 'Avg. categories'],
                    y=[avg_message_tokens, avg_categories_per_message]
                )
            ],

            'layout': {
                'title': 'Average tokens per message and average categories per message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=sort_categories_series.index,
                    y=sort_categories_series.values / df.shape[0]
                )
            ],

            'layout': {
                'title': 'Distribution of categories (percentage of usage)',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        # {
        #     'data': [
        #         Bar(
        #             x=less_common_categories_series.index,
        #             y=less_common_categories_series.values
        #         )
        #     ],
        #
        #     'layout': {
        #         'title': 'Less common categories (bottom 3)',
        #         'yaxis': {
        #             'title': "Count"
        #         },
        #         'xaxis': {
        #             'title': "Categories"
        #         }
        #     }
        # }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(category_names, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()