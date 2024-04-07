import re
import numpy as np
from keras_preprocessing.sequence import pad_sequences

import tensorflow as tf


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


def process(prompts, responses):
    prompts = np.array(prompts)
    responses = np.array(responses)

    inTokenizer = Tokenizer()
    inTokenizer.fit(prompts)

    outTokenizer = Tokenizer()
    outTokenizer.fit(responses)

    x_sequences = inTokenizer.tokenize(prompts)
    y_sequences = outTokenizer.tokenize(responses)

    x_data = [[1] + sequence + [2] for sequence in x_sequences]
    y_data = [[1] + sequence for sequence in y_sequences]
    z_data = [sequence + [2] for sequence in y_sequences]

    maxX = max(len(seq) for seq in x_sequences)
    maxY = max(len(seq) for seq in y_sequences)
    maxLength = max(maxX, maxY)

    x = pad_sequences(x_data, maxlen=maxLength, padding='post')
    context = pad_sequences(y_data, maxlen=maxLength, padding='post')
    labels = pad_sequences(z_data, maxlen=maxLength, padding='post')

    x = tf.data.Dataset.from_tensor_slices(x)
    context = tf.data.Dataset.from_tensor_slices(context)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((x, context))
    dataset = tf.data.Dataset.zip((dataset, labels))

    batch_size = 32
    dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)

    return dataset, (inTokenizer, outTokenizer)


