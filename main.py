import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tokenizer import tokenizer
import sys


np.random.seed(7)
tk = tokenizer()

if __name__ == "__main__":
    max_features = 100000
    maxlen = 150
    embed_size = 300
    xtr, xte, y, word_index = tk.tokenizer()
    embedding_vector = tk.make_glovevec(word_index)
    # print(word_index)

    predicted = [ [1.7147006, 3.8925774],
                  [  0, 4.9815335 ],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335 * 2.78],
                  [ 3.8925774, 4.9815335 * 2.78],
                  [ 3.8925774, 4.9815335 * 2.78],
                  [ 3.8925774, 4.9815335 * 2.78],
                  [ 3.8925774, 4.9815335],
                  [ 3.8925774, 4.9815335]
                  ] #,  4.6426992,  4.9815335,  7.558293, 5.3626356,  9.016965,   8.049676,   8.867272,   8.9429865,  7.102303, 9.823166,  10.860782,  10.14977,    8.756805 ] ]
    classes = np.argmax(predicted, axis=1)
    print(classes)

    #print(embedding_vector)
    model = tk.BidLstm(embedding_vector)

    print("model compile started")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #file_path = "./model.hdf5"
    #ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
    #                       save_best_only=True, mode='min')
    #early = EarlyStopping(monitor="val_loss", mode="min", patience=1)

    print("model fit started")#
    print(y.shape)
    #model.fit(xtr, y, batch_size=64, epochs=1, validation_split=0.1, callbacks=[ckpt, early])
    model.fit(xtr, y, batch_size=64, epochs=1, validation_split=0.1)

    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")


    #model.load_weights(file_path)
    y_test = model.predict(xte[1])

    print('test results')
    print(y_test)
    print(y_test.shape)

    for i in range(len(y_test)):
        print("X=%s, Predicted=%s" % (xte[i], y_test[i]))
    '''
    print("X=%s, Predicted=%s" % (xte[1], y_test[1]))
    print("X=%s, Predicted=%s" % (xte[2], y_test[2]))
    print("X=%s, Predicted=%s" % (xte[3], y_test[3]))
    print("X=%s, Predicted=%s" % (xte[4], y_test[4]))
    '''




