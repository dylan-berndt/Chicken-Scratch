import tensorflow as tf
import keras
from keras import layers

from keras import Sequential

import numpy as np


def positionalEncoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocabSize, embeddingDepth):
        super().__init__()
        self.embeddingDepth = embeddingDepth
        self.embedding = layers.Embedding(vocabSize, embeddingDepth, mask_zero=True)
        self.encoding = positionalEncoding(2048, embeddingDepth)

    def computeMask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embeddingDepth, tf.float32))
        x = x + self.encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.multiAttention = layers.MultiHeadAttention(**kwargs)
        self.normalization = layers.LayerNormalization()
        self.add = layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attentionOutput, attentionScores = self.multiAttention(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        self.lastAttention = attentionScores
        x = self.add([x, attentionOutput])
        x = self.normalization(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attentionOutput = self.multiAttention(
            query=x,
            value=x,
            key=x
        )

        x = self.add([x, attentionOutput])
        x = self.normalization(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attentionOutput = self.multiAttention(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )

        x = self.add([x, attentionOutput])
        x = self.normalization(x)

        return x


class FeedForward(layers.Layer):
    def __init__(self, embeddingDepth, feedDepth, dropout=0.1):
        super().__init__()
        self.sequence = Sequential([
            layers.Dense(feedDepth, activation='relu'),
            layers.Dense(embeddingDepth),
            layers.Dropout(dropout)
        ])
        self.add = layers.Add()
        self.normalization = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.sequence(x)])
        x = self.normalization(x)
        return x


class EncoderLayer(layers.Layer):
    def __init__(self, *, embeddingDepth, numHeads, feedDepth, dropout=0.1):
        super().__init__()
        self.attention = GlobalSelfAttention(
            num_heads=numHeads,
            key_dim=embeddingDepth,
            dropout=dropout
        )

        self.feed = FeedForward(embeddingDepth, feedDepth)

    def call(self, x):
        x = self.attention(x)
        x = self.feed(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, *, numLayers, embeddingDepth, numHeads, feedDepth, vocabSize, dropout=0.1):
        super().__init__()
        self.embeddingDepth = embeddingDepth
        self.numLayers = numLayers

        self.embedding = PositionalEmbedding(vocabSize=vocabSize, embeddingDepth=embeddingDepth)
        self.encoders = [
            EncoderLayer(embeddingDepth=embeddingDepth,
                         numHeads=numHeads,
                         feedDepth=feedDepth,
                         dropout=dropout
            )
            for _ in range(numLayers)
        ]
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        
        for i in range(self.numLayers):
            x = self.encoders[i](x)

        return x


class DecoderLayer(layers.Layer):
    def __init__(self, *, embeddingDepth, numHeads, feedDepth, dropout=0.1):
        super().__init__()
        self.selfAttention = CausalSelfAttention(
            num_heads=numHeads,
            key_dim=embeddingDepth,
            dropout=dropout
        )
        self.crossAttention = CrossAttention(
            num_heads=numHeads,
            key_dim=embeddingDepth,
            dropout=dropout
        )
        self.feed = FeedForward(embeddingDepth, feedDepth)

    def call(self, x, context):
        x = self.selfAttention(x)
        x = self.crossAttention(x, context)

        self.lastAttention = self.crossAttention.lastAttention

        x = self.feed(x)

        return x


class Decoder(layers.Layer):
    def __init__(self, *, numLayers, embeddingDepth, numHeads, feedDepth, vocabSize, dropout=0.1):
        super(Decoder, self).__init__()
        self.embeddingDepth = embeddingDepth
        self.numLayers = numLayers

        self.embedding = PositionalEmbedding(vocabSize=vocabSize, embeddingDepth=embeddingDepth)
        self.dropout = layers.Dropout(dropout)
        self.decoders = [
            DecoderLayer(
                embeddingDepth=embeddingDepth,
                numHeads=numHeads,
                feedDepth=feedDepth,
                dropout=dropout
            )
            for _ in range(numLayers)
        ]
        self.lastAttention = None

    def call(self, x, context):
        x = self.embedding(x)
        x = self.dropout(x)
        
        for i in range(self.numLayers):
            x = self.decoders[i](x, context)

        self.lastAttention = self.decoders[-1].lastAttention

        return x


def masked_loss(label, pred):
    mask = label != 0

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    
    mask = label != 0
    
    match = match & mask
    
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


class Transformer(keras.Model):
    def __init__(self, *, numLayers, embeddingDepth, numHeads, feedDepth, vocabSize, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(numLayers=numLayers, embeddingDepth=embeddingDepth,
                               numHeads=numHeads, feedDepth=feedDepth,
                               vocabSize=vocabSize, dropout=dropout)

        self.decoder = Decoder(numLayers=numLayers, embeddingDepth=embeddingDepth,
                               numHeads=numHeads, feedDepth=feedDepth,
                               vocabSize=vocabSize, dropout=dropout)

        self.probabilities = layers.Dense(vocabSize)

    def call(self, inputs):
        context, x = inputs
        
        context = self.encoder(context)
        x = self.decoder(x, context)
        
        logits = self.probabilities(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


class Translator(tf.Module):
    def __init__(self, tokenizer, transformer):
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, sentence: str, maxLength: int = 256):
        tokenized = self.tokenizer.tokenize([sentence])
        print(tokenized)
        sentence = tf.data.Dataset.from_tensor_slices(tokenized)

        encoderInput = sentence

        start = 1
        end = 0

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in range(maxLength):
            output = tf.transpose(output_array.stack())

            predictions = self.transformer([encoderInput, output], training=False)
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        return output_array.stack()

















