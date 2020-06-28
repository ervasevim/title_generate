# load and evaluate a saved model
from scipy.spatial import distance as dis

from keras.models import load_model
from database.database import database
from tokenizer import tokenizer
import numpy as np
import sys

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

my_arr = [3.90820112e-24, 1.71026426e-20, 1.15613995e-23, 7.50746205e-02, 1.56222810e-14, 4.19488298e-11, 9.24925387e-01, 5.02750064e-10, 1.57507929e-09]
my_arr2 = softmax(my_arr).ravel()
print(my_arr2)

# sys.exit()
tk = tokenizer()
# load model
model = load_model('model3_only_lstm.h5')
# summarize model.
model.summary()
# load dataset
test_titles = []
test_contents = []


xtr, xte, y, word_index, my_tokenizer_t, my_tokenizer_c = tk.tokenizer()

file_c = open(r"C:/PythonProjects/title_generate/results/result_c.txt","w+")
file_t = open(r"C:/PythonProjects/title_generate/results/result_t.txt","w+")

for content in xte:
    text = []
    encoded_arr = []
    cumle = model.predict( content )
    for kelime_vektor in cumle:
        kelime_sayi = np.argmax( kelime_vektor, axis=0)
        encoded_arr.append( kelime_sayi )

    text_t = my_tokenizer_t.sequences_to_texts( [encoded_arr] )
    text_c = my_tokenizer_c.sequences_to_texts( [encoded_arr] )

    file_t.write(' '.join( text_t ))
    file_t.write( '\r\n' )

    file_c.write( ' '.join( text_c ) )
    file_c.write( '\r\n' )

file_t.close()
file_c.close()

