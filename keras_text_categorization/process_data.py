# coding:utf-8
import codecs
import random
import os
import numpy as np

def kerasformat(ftrain, ftest, words, ischar=True):
    """
    转换为keras输入格式
    :param ftrain: train sets path
    :param ftest:  test sets path
    :param words: dictionary
    :param ischar: True-char, False-words
    :return: keras format of train and test
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    for f in [ftrain, ftest]:
        with codecs.open(f, "r", "utf-8") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                line = strQ2B(line.strip())
                line = line.strip().lower().split('\t', 2)
                label = int(line[0])
                if ischar:
                    content = list(line[1])
                else:
                    content = line[1].split()
                if f == ftrain:
                    sent_id = [words[w] for w in content if words.has_key(w)]
                    x_train.append(sent_id)
                    y_train.append(label)
                else:
                    sent_id = [words[w] for w in content if words.has_key(w)]
                    x_test.append(sent_id)
                    y_test.append(label)
    return x_train, y_train, x_test, y_test

def test_kerasformat(ftest, words, ischar=True):
    """
    转换为keras输入格式，仅转换测试集
    :param ftest:
    :param words:
    :param ischar:
    :return:
    """
    x_test, y_test = [], []
    with codecs.open(ftest, "r", "utf-8") as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = strQ2B(line.strip())
            line = line.strip().lower().split('\t', 2)
            label = int(line[0])
            if ischar:
                content = list(line[1])
            else:
                content = line[1].split()
            sent_id = [words[w] for w in content if words.has_key(w)]
            x_test.append(sent_id)
            y_test.append(label)
    return x_test, y_test

def gendict(ftrain, ftest):
    """
    根据语料生成词典，仅基于词时使用
    :param ftrain:
    :param ftest:
    :return:
    """
    wordscount = dict()
    words = dict()
    maxfeature = 1
    for f in [ftrain, ftest]:
        with codecs.open(f, "r", "utf-8") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                line = strQ2B(line.strip())
                line = line.strip().lower().split('\t', 2)
                content = line[1].split()
                for w in content:
                    if not wordscount.has_key(w):
                        wordscount[w] = 1
                    else:
                        wordscount[w] += 1
    for w in wordscount:
        if wordscount[w] > 20:
            words[w] = maxfeature
            maxfeature += 1
    return words

# 读取二进制格式的词向量
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    layer1_size = 0
    if not os.path.exists(fname):
        print "[load_bin_vec] error file not exists." + fname
        return word_vecs, layer1_size

    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs, layer1_size

# 读取txt格式的词向量
def load_txt_vec(fname, vocab):
    """
    Loads nx1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    layer1_size = 0
    if not os.path.exists(fname):
        print "[load_txt_vec] error file not exists." + fname
        return word_vecs, layer1_size

    with codecs.open(fname, "rb", "utf-8") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        word = []
        while True:
            lines = f.readlines(100000)
            if not lines:
                 break
            for line in lines:
                 sp = line.split()
                 word = sp[0]
                 if word in vocab:
                     word_vecs[word] = np.array(map(float, sp[1:]),dtype='float32')
                     if len(word_vecs[word]) != layer1_size:
                         print "[load_txt_vec] word vector error."
                     else:
                         continue
    return word_vecs, layer1_size

def get_W(word_vecs, k, words):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    for word in word_vecs:
        if words.has_key(word):
            W[words[word]] = word_vecs[word]
        else:
            print word
    return W

def add_unknown_words(word_vecs, vocab, k=300, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    i = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            i += 1
            if i % 100 ==0:
                print i,' unknown word is added such as ', word,'.'

# 保存词典
def savedict(words, fwords):
    sort = sorted(words.items(), key=lambda e:e[1], reverse=False)
    with codecs.open(fwords,"w","utf-8") as fw:
        for i in sort:
            fw.write(i[0]+'\t'+str(i[1])+'\n')
# 读取字典
def loaddict(fwords):
    wordsdict = dict()
    with codecs.open(fwords, "r", "utf-8") as fr:
        for line in fr.readlines():
            line = line.strip().split()
            if len(line) < 2:
                wordsdict[' '] = int(line[0]) 
            else:
                wordsdict[line[0]] = int(line[1])
    return wordsdict

# 全角符号转半角
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        elif (inside_code >= 0xff01 and inside_code <= 0xff5e):
            inside_code -= 0xfee0
        rstring += unichr(inside_code)
    return rstring

if __name__ == "__main__":
    vec = "data/bbs50dchar.vec"
    ftrain = "data/train"
    ftest = "data/test"
    k = 50
    #_, _, _, _, vocab = kerasformat(ftrain, ftest, True)
    #word_vecs, layer1_size = load_txt_vec(vec, vocab)
    #add_unknown_words(word_vecs, vocab, k, min_df=1)
    #W = get_W(word_vecs, k, vocab)
    #print len(vocab)

    s = u'发动机ｄｋＦＬ０２８＼｜国，。？！，'
    print s
    print strQ2B(s)
