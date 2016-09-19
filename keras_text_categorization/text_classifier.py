# coding:utf-8
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from sklearn import metrics
import os,sys
import process_data

class keras_text_classifier(object):
    """
    keras text classifier with convolution neural network.
    """
    def __init__(self,
                 train_path=None,
                 test_path=None,
                 nb_classes=2,
                 dictionary_path="./dict/char.list",
                 weights_path=None,
                 w2v_path=None,
                 embedding_binary=False,
                 ischar=True,
                 maxlen=200,
                 batch_size=128,
                 nb_epoch=5,
                 nb_filter=256,
                 filter_length=5,
                 hidden_dims=256,
                 embedding_length=100):
        """
        Initialization parameters
        :param train_path: train sets path, format : <label><tab><text>
        :param test_path: test sets path, format : <label><tab><text>
        :param nb_classes: categorization numbers
        :param dictionary_path: dictionary path
        :param weights_path: weights path of model
        :param w2v_path: pre-trained word2vec path <default None>
        :param embedding_binary: word2vec is text or binary, False-text, True-binary, TODO: just when w2v!=None could be used
        :param ischar: True-char, False-words
        :param maxlen: max size of sentence, padding with it <default 200>
        :param batch_size: batch size <default 128>
        :param nb_filter: Number of convolution kernels to use (dimensionality of the output).
        :param filter_length: The extension (spatial or temporal) of each filter
        :param hidden_dims: hidden dims ( fully connected NN layer)
        :param embedding_length: embedding length
        """
        self.train_path = train_path
        self.test_path = test_path
        self.nb_classes = nb_classes
        self.dictionary_path = dictionary_path
        self.weights_path = weights_path
        self.w2v_path = w2v_path
        self.embedding_binary = embedding_binary
        self.ischar = ischar
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.hidden_dims = hidden_dims
        self.embedding_length = embedding_length

        # if use character level, default use "char.list" dictionary
        if self.ischar == True:
            self.dictionary = './dict/char.list'

    def processing(self):
        """
        processing data for keras format
        """
        print("Load data...")
        # sentence to id
        if self.ischar:
            self.words = process_data.loaddict(self.dictionary_path)
            x_train, y_train, x_test, y_test = process_data.kerasformat(self.train_path, self.test_path, self.words, True)
        else:
            self.words = process_data.gendict(self.train_path, self.test_path)
            process_data.savedict(self.words, self.dictionary_path)
            x_train, y_train, x_test, y_test = process_data.kerasformat(self.train_path, self.test_path, self.words, False)

        # load pre-trained word2vec weights
        # print(self.w2v_path)
        if self.w2v_path:
            W, layer1_size = self.loadword2vec()
            data = (x_train, y_train, x_test, y_test, W, layer1_size)
        else:
            data = (x_train, y_train, x_test, y_test, None, None)

        self.x_train, self.y_train, self.x_test, self.y_test, self.W, self.layer1_size = data

        print("train size:", len(self.x_train))
        print("test size:", len(self.x_test))
        print("max_features:", max(self.words.values()) + 1)

    def loadword2vec(self):
        """
        load pre-train word2vec
        :return: W, layer1_size
        """
        if self.embedding_binary:
            word_vecs, layer1_size = process_data.load_bin_vec(self.w2v_path, self.words)
        else:
            word_vecs, layer1_size = process_data.load_txt_vec(self.w2v_path, self.words)
        if layer1_size > 0:
            process_data.add_unknown_words(word_vecs, self.words, layer1_size, 1)
        W = process_data.get_W(word_vecs, layer1_size, self.words)
        return W, layer1_size

    def createmodel(self):
        """
        create cnn model structure
        :return: model structure
        """
        max_features = max(self.words.values()) + 1 # input dims
        model = Sequential()
        if self.W is None:
            model.add(Embedding(max_features, self.embedding_length, input_length=self.maxlen, dropout=0.2))
        else:
            model.add(Embedding(max_features, self.layer1_size, weights=[self.W], input_length=self.maxlen, dropout=0.2))
        
        model.add(Convolution1D(nb_filter=self.nb_filter,
                                filter_length=self.filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))

        model.add(MaxPooling1D(pool_length=model.output_shape[1]))
        model.add(Flatten())
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
        return model

    def train(self):
        """
        train model
        """
        if not os.path.exists(self.train_path):
            print("Not found exist train file:%s" % self.train_path)
            sys.exit(0)
        self.processing()
        model = self.createmodel()

        # padding
        x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)

        y_train = np_utils.to_categorical(np.asarray(self.y_train))
        y_test = np_utils.to_categorical(np.asarray(self.y_test))

        print("Train...")
        model.fit(x_train,
                  y_train,
                  batch_size=self.batch_size,
                  nb_epoch=self.nb_epoch,
                  validation_data=(x_test, y_test),
                  shuffle=1)
        model.save_weights(self.weights_path, overwrite=True)

        # evaluate
        self.evaluate(model, x_test, y_test)

    def evaluate(self, model, x_test, y_test):
        pred_y = model.predict_classes(x_test)
        confusion = metrics.confusion_matrix(y_test[:,1], pred_y)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print('Accuracy:(TP+TN) / float(TP+TN+FN+FP)=', (TP + TN) / float(TP + TN + FN + FP))
        print('Error:(FP+FN) / float(TP+TN+FN+FP)=', (FP + FN) / float(TP + TN + FN + FP))
        print('Recall:TP / float(TP+FN)=', TP / float(TP + FN))
        print('Specificity:TN / float(TN+FP)=', TN / float(TN + FP))
        print('False Positive Rate:FP / float(TN+FP)=', FP / float(TN + FP))
        print('Precision:TP / float(TP+FP)=', TP / float(TP + FP))
        print('AUC=', metrics.roc_auc_score(y_test[:,1], pred_y))
       
    def loadmodel(self):
        self.words = process_data.loaddict(self.dictionary_path)
        if self.w2v_path:
            self.W, self.layer1_size = self.loadword2vec()
        else:
            self.W = None
            self.layer1_size = None
        model = self.createmodel()
        model.load_weights(self.weights_path)
        return model

    def testbatch(self):
        """
        batch test
        """
        if not os.path.exists(self.test_path):
            print("Not found exist test file:%s" % self.train_path)
            sys.exit(0)
        model = self.loadmodel()
        x_test, y_test = process_data.test_kerasformat(self.test_path, self.words, self.ischar)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        y_test = np_utils.to_categorical(np.asarray(y_test))
        self.evaluate(model, x_test, y_test)

    def test(self, sentence, model, words):
        """
        test only a sentence
        :param sentence: a sentence, if ischar==False, the sentence should be segmented
        :param model: cnn model
        :param words: words list
        :return:
        """
        if self.ischar is True:
            sentence = list(sentence)
        else:
            sentence = sentence.split()
        x_test = [[words[w] for w in sentence if words.has_key(w)]]
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        pred_y = model.predict(x_test)
        return pred_y

if __name__ == "__main__":
    ftrain = "data/train.seg"#"data/cat9/trainclass9.txt"
    ftest  = "data/test.seg"#"data/cat9/testclass9.txt"
    nb_classes = 2
    dictionary_path = "dict/words.list"
    weights_path = "model/cnn_words_w2v.h5"
    w2v_path = "data/bbs100dword.vec"

    #tc = keras_text_classifier(ftrain, ftest, nb_classes, dictionary_path, weights_path, ischar=False, maxlen=70)
    tc = keras_text_classifier(ftrain, ftest, nb_classes, dictionary_path, weights_path, w2v_path, ischar=False, maxlen=70)

    # 1. training model
    tc.train()

    # 2. test batch
    tc.testbatch()

    # 3. test sentence, need for segment
    sentence = u'油耗 稍微 有点 高'
    model = tc.loadmodel()
    words = process_data.loaddict(dictionary_path)
    print(tc.test(sentence, model, words))

