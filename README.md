# Disaster Response Pipeline Project

### Preparation
Install the required packages:

    pip install -r requirements.txt

### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`  
        (alternatively run the script: `run_etl_pipeline.sh`)
        
    - To run ML pipeline that trains classifier and saves (before run check **Execution time** section)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`  
        (lternatively run the script: `run_ml_pipeline.sh` )

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**NOTE:** DisasterResponse.db and classifier.pkl are provided in the repository as references. To avoid errors or to 
overwriting, rename them.

## Details

### Data cleaning
The data, messages and categories, are loaded from CSVs and merged into a dataframe. Before the merging, the data cleaning routine performs these main steps:

- Split the values in the categories column on the ";" character so that each value becomes a separate column, and 
create a dataframe of the 36 individual category columns
- Use the first row of categories dataframe to create column names for the categories data.
- Convert category values to just numbers 0 or 1
- Concatenate the original dataframe with the new `categories` dataframe
- Remove duplicates

#### Categories inconsistency

In the __clean_data__ function (__process_data.py__ script) a further check over categories values is performed to spot inconsistent values. Specifically, each row that contains non-boolean values (thus not 0 or 1) is removed. 

    In the given dataset, some values in **related** category as values 2, those rows are removed.

#### One-class categories

Some classification methods, like SVC, do not support a label that assumes only one value (one-class), e.g., a category in which all values in column are zeros or ones. In order to avoid the issue, the routine **clean_one_class_category** is invoked in the 
__train_classification.py__ script, to remove the labels with all values zeros or ones.

    In the given dataset, the **child_alone** category is composed of only zeros, thus removed to allow the usage of SVC.

Routine available in the __extra.py__ script.

### Model

#### Pipeline
At first, a tokenize function is defined, which performs: 

- Test normalization: remove non-alphabetical and non-numerical characters
- Word tokens: convert message into tokens
- Remove stop words
- Lemmatizing

The ML pipeline performs the following operations: 

- CountVectorizer (using tokenize function)
- MyTfidfTransformer (standard TfidfTransformer with fix to support multiprocessing, see below for details)
- MultiOutputClassifier applied to selected estimator
- Estimator: Support Vector Classification

The pipeline is then feeded into search grid, that search the best combination of the following parameters: 

```python
'vect__ngram_range': ((1, 1), (1, 2)),
'vect__max_df': (0.5, 0.75, 1.0),
'vect__max_features': (None, 5000, 10000),
'tfidf__use_idf': (True, False),
'cls__estimator__kernel': ['linear', 'rbf']
```

##### Execution time
The "full" GridSearchCV takes approximately 13.5 hours to be performed on a i9-9980HK (using the MultiOutputClassifier with 15 processes). In order to get a rapid overview of the GridSearchCV, the following reduced version of the parameters can be used (commented in the code - line 78 in train_classifier.py):

```python
'vect__ngram_range': ((1, 2)),
'vect__max_df': (0.5),
'vect__max_features': (10000),
'tfidf__use_idf': (True, False),
'cls__estimator__kernel': ['linear', 'rgf']
```

#### Custom Tfidf

A [workaround](https://github.com/scikit-learn/scikit-learn/issues/6614) for a bugs that doesn't allow to SVC to benefit of multiprocessing in MultiOutputClassifier, has been implemented. Specifically, the following class is added to replace the default TfidfTransformer estimator.

```python
class MyTfidfTransformer(TfidfTransformer):
    def fit_transform(self, X, y):
        result = super(MyTfidfTransformer, self).fit_transform(X, y)
        result.sort_indices()
        return result
```

Class available in the __extra.py__ script.


### App

#### Model results
The Falsk app allows to insert a custom message, and displays the categories for the given message.

#### Graphs
The app displays 3 graphs about training data:

- Distribution of Message Genres
- Average tokens per message and average categories per message
- Distribution of categories (percentage of usage)

## Best model
The best model presents the following parameters:

```python
'vect__ngram_range': (1, 2),
'vect__max_df': 0.5,
'vect__max_features': 10000,
'tfidf__use_idf': False,
'cls__estimator__kernel': 'linear'
```

The test results achieved are:

| Category             | Label | Precision | Recall | f1-score | Accuracy |
| :------------------- | :---: | :-------- | :----- | :------- | -------- |
| related              |   0   | 0.7       | 0.54   | 0.61     | 0.83     |
|                      |   1   | 0.87      | 0.93   | 0.89     |          |
| request              |   0   | 0.91      | 0.97   | 0.94     | 0.90     |
|                      |   1   | 0.8       | 0.55   | 0.65     |          |
| offer                |   0   | 1         | 1      | 1        | 1.00     |
|                      |   1   | 0         | 0      | 0        |          |
| aid_related          |   0   | 0.8       | 0.86   | 0.82     | 0.79     |
|                      |   1   | 0.77      | 0.69   | 0.73     |          |
| medical_help         |   0   | 0.94      | 0.99   | 0.96     | 0.93     |
|                      |   1   | 0.69      | 0.24   | 0.36     |          |
| medical_products     |   0   | 0.96      | 0.99   | 0.98     | 0.96     |
|                      |   1   | 0.75      | 0.29   | 0.42     |          |
| search_and_rescue    |   0   | 0.98      | 1      | 0.99     | 0.97     |
|                      |   1   | 0.69      | 0.17   | 0.27     |          |
| security             |   0   | 0.98      | 1      | 0.99     | 0.98     |
|                      |   1   | 0         | 0      | 0        |          |
| military             |   0   | 0.98      | 1      | 0.99     | 0.97     |
|                      |   1   | 0.66      | 0.25   | 0.36     |          |
| water                |   0   | 0.98      | 0.99   | 0.98     | 0.97     |
|                      |   1   | 0.77      | 0.63   | 0.69     |          |
| food                 |   0   | 0.97      | 0.98   | 0.97     | 0.95     |
|                      |   1   | 0.81      | 0.75   | 0.78     |          |
| shelter              |   0   | 0.96      | 0.98   | 0.97     | 0.95     |
|                      |   1   | 0.77      | 0.56   | 0.65     |          |
| clothing             |   0   | 0.99      | 1      | 0.99     | 0.99     |
|                      |   1   | 0.76      | 0.49   | 0.6      |          |
| money                |   0   | 0.98      | 1      | 0.99     | 0.97     |
|                      |   1   | 0.72      | 0.09   | 0.16     |          |
| missing_people       |   0   | 0.99      | 1      | 0.99     | 0.99     |
|                      |   1   | 0.64      | 0.12   | 0.2      |          |
| refugees             |   0   | 0.97      | 1      | 0.98     | 0.97     |
|                      |   1   | 0.71      | 0.16   | 0.27     |          |
| death                |   0   | 0.97      | 0.99   | 0.98     | 0.97     |
|                      |   1   | 0.72      | 0.43   | 0.54     |          |
| other_aid            |   0   | 0.87      | 0.99   | 0.93     | 0.87     |
|                      |   1   | 0.56      | 0.07   | 0.13     |          |
| infrastructure       |   0   | 0.94      | 1      | 0.97     | 0.94     |
|                      |   1   | 1         | 0      | 0.01     |          |
| transport            |   0   | 0.96      | 1      | 0.98     | 0.96     |
|                      |   1   | 0.83      | 0.15   | 0.25     |          |
| buildings            |   0   | 0.96      | 1      | 0.98     | 0.96     |
|                      |   1   | 0.8       | 0.27   | 0.41     |          |
| electricity          |   0   | 0.98      | 1      | 0.99     | 0.98     |
|                      |   1   | 0.65      | 0.25   | 0.37     |          |
| tools                |   0   | 0.99      | 1      | 1        | 0.99     |
|                      |   1   | 0         | 0      | 0        |          |
| hospitals            |   0   | 0.99      | 1      | 1        | 0.99     |
|                      |   1   | 0         | 0      | 0        |          |
| shops                |   0   | 1         | 1      | 1        | 1.00     |
|                      |   1   | 0         | 0      | 0        |          |
| aid_centers          |   0   | 0.99      | 1      | 0.99     | 0.99     |
|                      |   1   | 0         | 0      | 0        |          |
| other_infrastructure |   0   | 0.96      | 1      | 0.98     | 0.96     |
|                      |   1   | 0         | 0      | 0        |          |
| weather_related      |   0   | 0.9       | 0.96   | 0.93     | 0.90     |
|                      |   1   | 0.87      | 0.74   | 0.8      |          |
| floods               |   0   | 0.96      | 1      | 0.98     | 0.95     |
|                      |   1   | 0.92      | 0.51   | 0.66     |          |
| storm                |   0   | 0.97      | 0.98   | 0.97     | 0.95     |
|                      |   1   | 0.78      | 0.66   | 0.72     |          |
| fire                 |   0   | 0.99      | 1      | 1        | 0.99     |
|                      |   1   | 0.83      | 0.26   | 0.39     |          |
| earthquake           |   0   | 0.98      | 0.99   | 0.99     | 0.98     |
|                      |   1   | 0.9       | 0.83   | 0.86     |          |
| cold                 |   0   | 0.98      | 1      | 0.99     | 0.98     |
|                      |   1   | 0.76      | 0.27   | 0.4      |          |
| other_weather        |   0   | 0.96      | 0.99   | 0.98     | 0.95     |
|                      |   1   | 0.5       | 0.11   | 0.17     |          |
| direct_report        |   0   | 0.88      | 0.96   | 0.92     | 0.86     |
|                      |   1   | 0.74      | 0.44   | 0.56     |          |


