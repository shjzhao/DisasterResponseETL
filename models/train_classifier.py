# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    load data from database
    :param database_filepath: (str)
    :return: (pd.Series)features, (pd.DataFrame)labels, (pd.columns)category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Split text into tokens.For each token: lemmatize, normalize case, and strip leading and trailing white space.
    Return the tokens in a list!
    :param text:(str)
    :return:(list)
    """
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Build pipeline and use grid search to find better parameters.
    :return: (GridSearchCV)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Show the accuracy, precision, and recall of the tuned model.
    :param model:(sklearn.pipeline.Pipeline)
    :param X_test: (pd.Series)test features
    :param Y_test:(pd.DataFrame)test labels
    :param category_names:(pd.columns)category names
    :return:None
    """
    Y_pred = model.predict(X_test)
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy * 100))
    y_pred_pd = pd.DataFrame(Y_pred, columns=category_names)
    for column in category_names:
        print('------------------------------------------------------\n')
        print('category: {}\n'.format(column))
        print(classification_report(Y_test[column], y_pred_pd[column]))
    return


def save_model(model, model_filepath):
    """
    Export the model as a pickle file
    :param model:(sklearn.pipeline.Pipeline)
    :param model_filepath:(str)
    :return:None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
