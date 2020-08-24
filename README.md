# Disaster Response Pipeline Project

Udacity Data Scientist Nanodegree - Project 2

## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Instructions
5. Results
6. Licensing, Authors, Acknowledgements
7. MIT License

## Installation

- Libraries included in Anaconda distribution of Python 3.8.
- Plotly package
- Packages versions:

    - pandas Version: 1.0.1
    - nltk Version: 3.4.5
    - SQLAlchemy Version: 1.3.13
    - scikit-learn Version: 0.22.1
    - Flask Version: 1.1.1
    - plotly Version: 4.9.0


## Project Motivation

The aim of this project is to analyse disaster data provided and pre-proccessed
by Figure Eight (currently acquired by Appen) to build a Machine Learning model
for an API that classifies disaster messages. This is a multi-label multi-output
classification task.

The project includes a web app where an emergency worker can input a new message
and get classification results in several categories, including visualisations.

By classifying messages into relevant categories using an ML model as opposed
to filtering messages manually, professionals will save a lot of time an
therefore respond much faster to the demands of a disaster event.

## File Descriptions

The data provided and pre-processed by Figure Eight is saved in two .csv files:

  - messages.csv
  - categories.csv

The main code is contained in two python scripts:

  - process_data.py - This script contains an ETL(Extract, Transform, Load)
  pipeline that reads then datasets, cleans the data, and then stores it in a
  SQLite database.
  - train_classifier.py - This script contains code that splits the
  data into a training set and a test set, creates a ML pipeline that uses NLTK,
  as well as scikit-learn's Pipeline and GridSearchCV to output a final model
  that uses the messages to predict classifications for 36 categories. Finally,
  the trained model is saved as a pickle file.

The folder called 'app' contains the code to run the Flask web app.

## Instructions

1. Run the following commands in the project's root directory to set up your
database and model.

    - To run ETL pipeline that cleans data and stores in database, with the terminal
    in the data directory:
        `python process_data.py messages.csv categories.csv DisasterResponse`
    - To run ML pipeline that trains classifier and saves, with the terminal in
    the models directory:
        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results

The Disaster Response Project web app allows the user to enter a message and
it classifies the message into categories. It also shows some visual insights
about the data such as the distribution of the categories in the datasets with
which the model has been trained.

All the code is in a public repository at the link below:

https://github.com/bergr7/Project_2_Disaster_Response_Pipelines

Note: The emphasis of this learning project is not on model performance but more
on data engineering and clean code. For that reason and due to limited time,
the performance of the model is not acceptable for a real world problem at this
stage of the development. The author plans to further try out different ML
algorithm and data processing steps, such as sampling, in order to make the model
pay more attention to categories with lots of negatives. (Typical issue of an
Imbalanced dataset).

## Licensing, Authors and Acknowledgements

I would like to give credit to:

- Figure Eight for providing and pre-processing the datasets
- Stackoverflow community for making code questions and answers publicly available
- Jason Brownlee from Machine Learning Mastery for the post '8 Tactics to Combat
  Imbalanced Classes in Your Machine Learning Dataset'

## MIT License

Copyright 2020 Bernardo Garcia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
