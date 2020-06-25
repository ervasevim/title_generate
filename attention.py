from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import os

# BASE_DIR = 'C:/Users/casper/Desktop/klasörler/tez/glove'
# GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
# TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000  # max_review_length   maxlen
MAX_NUM_WORDS = 33000  # the number of possible tokens,  (1 + maximum word index),    ##max_features
EMBEDDING_DIM = 100  # the dimensionality of the embeddings  embed_size =300
VALIDATION_SPLIT = 0.2
train_title = []
train_content = []
test_title = []
test_content = []


class attention(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        print('{}_W'.format(self.name))

        self.W = self.add_weight(name="attention_1_W",
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name="attention_1_W",
                                     shape=(input_shape[1]),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
