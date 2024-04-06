import tensorflow as tf
import keras
from keras import layers
from keras import models

import json
import random
import os
import numpy as np
import re

from keras_preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import tensorflow_text


class Tokenizer:
    def __init__(self):
        self.wordMap = {}
        self.inverseMap = {}
        
    def fit(self, textData):
        tokenized_texts = [self.tokenizeSentence(sentence) for sentence in textData] 
        wordMap = {'<EMP>': 0, '<STR>': 1, '<END>': 2}
        inverseMap = {0: '<EMP>', 1: '<STR>', 2: '<END>'}
        for text in tokenized_texts:
            for word in text:
                if word not in wordMap:
                    wordMap[word] = len(wordMap)
                    inverseMap[len(wordMap) - 1] = word
            
        self.wordMap = wordMap
        self.inverseMap = inverseMap
        
    def tokenize(self, text):        
        if len(self.wordMap) == 0:
            self.fit(text)
        wordTokens = [self.tokenizeSentence(sentence) for sentence in text]
            
        sequences = self.translateTokens(wordTokens)
        return sequences

    def tokenizeSentence(self, text):
        tokens = text.split()
        tokens = [re.sub(r'([^\w\s])', r' \1 ', token) for token in tokens]
        return tokens

    def translateTokens(self, tokenized_texts):
        sequences = [[self.wordMap[word] for word in text] for text in tokenized_texts]
        return sequences

    def deTranslate(self, tokenized_texts):
        sequences = [[self.inverseMap[word] for word in text] for text in tokenized_texts]
        return sequences        
    

def loadData(path, singular=False):
    prompts, responses = [], []
    tweets = 0
    
    files = os.listdir(path)
    for fileName in files:
        if not fileName.endswith('.json'):
            continue
        file = open(path + fileName)
        fileData = json.load(file)

        for key, tweet in fileData.items():
            tweetText = tweet['text']
            replies = tweet['replies']
            if singular and replies:
                prompts.extend([tweetText])
                responses.append(random.choice(replies))
            else:
                prompts.extend([tweetText] * len(replies))
                responses.extend(replies)
            tweets += 1
            print("Tweets:", tweets, "Replies:", len(responses), end='\r')

    zipped = list(zip(prompts, responses))
    random.shuffle(zipped)
    prompts, responses = zip(*zipped)

    both = np.array(prompts + responses)

    prompts = np.array(prompts)
    responses = np.array(responses)

    tokenizer = Tokenizer()
    tokenizer.fit(both)
    
    x_sequences = tokenizer.tokenize(prompts)
    y_sequences = tokenizer.tokenize(responses)

    maxX = max(len(seq) for seq in x_sequences)
    maxY = max(len(seq) for seq in y_sequences)
    maxLength = max(maxX, maxY)

    x_train = pad_sequences(x_sequences, maxlen=maxLength)
    y_train = pad_sequences(y_sequences, maxlen=maxLength, padding='post')

    print()

    return x_train, y_train, tokenizer


def cycleData(x_train, y_train):
    x_cycled, y_cycled = [], []
    for i in range(len(x_train)):
        x, y = x_train[i], y_train[i]
        data = x
        while len(y) > 0 and y[0] != 0:
            x_cycled.append(data)
            data = np.roll(data, -1)
            data[-1] = y[0]
            y = y[1:]
            y_cycled.append(data)

    x_cycled = np.array(x_cycled)
    y_cycled = np.array(y_cycled)
    
    return x_cycled, y_cycled


def loadTransformerData(path, singular=False):
    prompts, responses = [], []
    tweets = 0

    seenIds = []
    
    files = os.listdir(path)
    for fileName in files:
        if not fileName.endswith('.json'):
            continue
        file = open(path + fileName)
        fileData = json.load(file)

        for key, tweet in fileData.items():
            if key in seenIds:
                continue
            seenIds.append(key)
            tweetText = tweet['text']
            replies = tweet['replies']
            if singular and replies:
                prompts.extend([tweetText])
                responses.extend([random.choice(replies)])
            else:
                prompts.extend([tweetText] * len(replies))
                responses.extend(replies)
            tweets += 1
            print("Tweets:", tweets, "Replies:", len(responses), end='\r')

    both = np.array(prompts + responses)

    prompts = np.array(prompts)
    responses = np.array(responses)

    tokenizer = Tokenizer()
    tokenizer.fit(both)
    
    x_sequences = tokenizer.tokenize(prompts)
    y_sequences = tokenizer.tokenize(responses)

    x_data = [[1] + sequence + [2] for sequence in x_sequences]
    y_data = [[1] + sequence for sequence in y_sequences]
    z_data = [sequence + [2] for sequence in y_sequences]

    maxX = max(len(seq) for seq in x_sequences)
    maxY = max(len(seq) for seq in y_sequences)
    maxLength = max(maxX, maxY)

    x = pad_sequences(x_data, maxlen=maxLength, padding='post')
    context = pad_sequences(y_data, maxlen=maxLength, padding='post')
    labels = pad_sequences(z_data, maxlen=maxLength, padding='post')

    print()

    x = tf.data.Dataset.from_tensor_slices(x)
    context = tf.data.Dataset.from_tensor_slices(context)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((x, context))
    dataset = tf.data.Dataset.zip((dataset, labels))

    batch_size = 32
    dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)

    return dataset, tokenizer


tokenizers = None

MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64
def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))


def loadStableData(percent=100):
    global tokenizers
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True,
                                  split='train[:' + str(percent) + '%]')
    train_examples = examples
    
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load(model_name)

    return make_batches(train_examples), tokenizers

    


def LSTMLLM(vocabSize, embeddingDim, maxLength, lstmDepth):
    model = models.Sequential([
        layers.Input(shape=(maxLength,)),
        layers.Embedding(vocabSize, embeddingDim),
        layers.LSTM(lstmDepth, return_sequences=True),
        layers.LSTM(lstmDepth, return_sequences=True),
        layers.Dense(512, activation='relu'),
        layers.Dense(vocabSize, activation='softmax')
    ])
    return model






    
