#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import operator
import math
from functools import reduce
from sklearn.model_selection import train_test_split

df = pd.read_csv('toxic_data_mid.csv')


print(df.head())


stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
             'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
maxDictionaryLength = 8000


def tokenize(sentence, isCreateDict=False):
    tmpTokens = sentence.lower().split()
    tokens = [token for token in tmpTokens if (
        (token not in stopwords) and (len(token) > 0))]
    # tokens = tmpTokens.filter((token) => !stopwords.includes(token) && token.length > 0);

    if isCreateDict:
        for token in tokens:
            if token in dictionary_dict:
                dictionary_dict[token] += 1
            else:
                dictionary_dict[token] = 1
    documentTokens.append(tokens)
    return tokens


def getInverseDocumentFrequency(documentTokens, dictionary):
    return list(map(lambda word: 1 + math.log(len(documentTokens) / reduce(lambda acc, curr: (1 if (word in curr) else 0) + acc, documentTokens, 0)), dictionary))


def encoder(sentence, dictionary, idfs):
    tokens = tokenize(sentence)
    tfs = getTermFrequency(tokens, dictionary)
    tfidfs = getTfIdf(tfs, idfs)
    return tfidfs


def getTermFrequency(tokens, dictionary):
    return list(map(lambda token: reduce(lambda acc, curr: (acc + 1 if (curr == token) else acc), tokens, 0), dictionary))


def getTfIdf(tfs, idfs):
    return [tf * idf for (tf, idf) in zip(tfs, idfs)]


# Sample Test Code used in the slides ( Module : preparing data for machine learning model )
dictionary_dict = {}
documentTokens = []
testComments = ['i loved the movie', 'movie was boring']

for comment in testComments:
    documentTokens.append(tokenize(comment, True))


dictionary = sorted(dictionary_dict, key=dictionary_dict.get, reverse=True)
idfs = getInverseDocumentFrequency(documentTokens, dictionary)

tfidfs = []

for comment in testComments:
    tfidfs.append(encoder(comment, dictionary, idfs))

print(dictionary_dict)
print(dictionary)
print(idfs)
print(tfidfs)


dictionary_dict = {}
documentTokens = []
df['tokens'] = df['comment_text'].apply(lambda x: tokenize(x, True))


print(df.head())


dictionary = sorted(dictionary_dict, key=dictionary_dict.get, reverse=True)
dictionary = dictionary[:maxDictionaryLength]
print('Length of dictionary : {0}'.format(len(dictionary)))
print(dictionary[:10])


idfs = getInverseDocumentFrequency(documentTokens, dictionary)
print(len(idfs))


df['features'] = df['comment_text'].apply(
    lambda x: encoder(x, dictionary, idfs))
df['features'].head()


df_new = df['features'].apply(lambda x: pd.Series(x))
df_new['toxic'] = df['toxic']


# ### Train Test Split


train, test = train_test_split(df_new, test_size=0.3)
train, val = train_test_split(train, test_size=0.1)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


train.shape, test.shape, val.shape


def df_to_dataset(dataframe, shuffle=True, batch_size=16):
    dataframe = dataframe.copy()
    labels = dataframe.pop('toxic')
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 16
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


numOfFeatures = len(dictionary)


# ### Build Model


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation='relu',
                              input_shape=(numOfFeatures,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.06),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=['accuracy'])
    return model


model = get_compiled_model()
model.summary()
model.fit(train_ds, epochs=20, validation_data=val_ds)


# ### Evaluate Model


model.evaluate(test_ds)


# ### Make Predictions


# make predictions
testComments = ['you suck', 'you are a great person']
tfidfs = []
for comment in testComments:
    tfidfs.append(encoder(comment, dictionary, idfs))
print(f'predicted probabliities : {model.predict(tfidfs)}')
print(f'predicted classes : {tf.round(model.predict(tfidfs))}')


# ### Export Model

model.save('toxicity_python.h5')

# run 'tensorflowjs_converter --input_format=keras toxicity_python.h5 tfjs_python_toxicity'

# write dictionary and IDFs


with open('tfjs_python_toxicity/dictionary.json', 'w') as outfile:
    json.dump(dictionary, outfile)

with open('tfjs_python_toxicity/idfs.json', 'w') as outfile:
    json.dump(idfs, outfile)
