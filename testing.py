# load and evaluate a saved model
from scipy.spatial import distance as dis

from keras.models import load_model
from database.database import database
from tokenizer import tokenizer
import numpy as np
import sys

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# sys.exit()
tk = tokenizer()
# load model
model = load_model('model4_new_tokenizer2.h5')
# summarize model.
model.summary()
# load dataset
test_titles = []
test_contents = []

tokenized_content, tokenized_title, word_index, my_tokenizer = tk.tokenizer()

for content in tokenized_content[20001:]:
    text = []
    encoded_arr = []
    cumle = model.predict(content)
    # break
    for kelime_vektor in cumle:
        kelime_sayi = np.argmax(kelime_vektor, axis=0)
        encoded_arr.append(kelime_sayi)

    text = my_tokenizer.sequences_to_texts([encoded_arr])
    print( text )