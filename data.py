import tensorflow as tf
import keras
from keras import layers
from keras import models

import json
import random
import os
import numpy as np

import tensorflow_datasets as tfds
import tensorflow_text

from process import Tokenizer, process


def loadTwitterReplyData(path, singular=False):
    context, responses = [], []
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
                context.extend([tweetText])
                responses.extend([random.choice(replies)])
            else:
                context.extend([tweetText] * len(replies))
                responses.extend(replies)
            tweets += 1
            print("Tweets:", tweets, "Replies:", len(responses), end='\r')

    context = np.array(context)
    responses = np.array(responses)

    return process(context, responses)


def splitConversation(line):
    pass


def loadTwitterThreadData(location):
    paths = [location + path for path in os.listdir(location)]


def loadConversationData():
    file = open("DailyDialog.txt")

    context, responses = [], []

    for conversation in file.readlines():
        lines = conversation.split("__eou__")
        context.append("__eou__".join(lines[:-1]))
        responses.append(lines[-1])

    return process(context, responses)


# tokenizers = None

# MAX_TOKENS = 128
#
#
# def prepare_batch(pt, en):
#     pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
#     pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
#     pt = pt.to_tensor()  # Convert to 0-padded dense Tensor
#
#     en = tokenizers.en.tokenize(en)
#     en = en[:, :(MAX_TOKENS+1)]
#     en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
#     en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens
#
#     return (pt, en_inputs), en_labels
#
#
# BUFFER_SIZE = 20000
# BATCH_SIZE = 64
#
#
# def make_batches(ds):
#     return (
#       ds
#       .shuffle(BUFFER_SIZE)
#       .batch(BATCH_SIZE)
#       .map(prepare_batch, tf.data.AUTOTUNE)
#       .prefetch(buffer_size=tf.data.AUTOTUNE))
#
#
# def loadStableData(percent=100):
#     global tokenizers
#     examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True,
#                                   split='train[:' + str(percent) + '%]')
#     train_examples = examples
#
#     model_name = 'ted_hrlr_translate_pt_en_converter'
#     tf.keras.utils.get_file(
#         f'{model_name}.zip',
#         f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
#         cache_dir='.', cache_subdir='', extract=True
#     )
#
#     tokenizers = tf.saved_model.load(model_name)
#
#     return make_batches(train_examples), tokenizers


# def LSTMLLM(vocabSize, embeddingDim, maxLength, lstmDepth):
#     model = models.Sequential([
#         layers.Input(shape=(maxLength,)),
#         layers.Embedding(vocabSize, embeddingDim),
#         layers.LSTM(lstmDepth, return_sequences=True),
#         layers.LSTM(lstmDepth, return_sequences=True),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(vocabSize, activation='softmax')
#     ])
#     return model






    
