# coding:utf-8
from text_classifier import keras_text_classifier
import process_data

class Classifier(object):
    def __init__(self,
                 ftrain=None,
                 ftest=None,
                 nb_classes=2,
                 dictionary_path="keras_text_categorization/dict/char.list",
                 weights_path=None,
                 w2v_path=None,
                 ischar=True,
                 maxlen=200):
        """
        参数初始化
        a. 默认使用基于字的cnn模型(ischar=True)，如果基于词需设置ischar=False
        b. 如果不使用预训练word2vec，需设置w2v_path=None
        c. 如果使用预训练word2vec，且为文本格式需设置embedding_binary=False
        :param ftrain: 训练数据路径，格式为： <label><tab><content>，预测时不需要指定该参数，设置为None
                       label：从0开始的连续数字
                       tab：tab分割
                       content: 文本数据，如果使用基于词的分类，文本需要分词
        :param ftest:  测试数据路径，格式同训练数据，单句预测时不需要指定该参数，设置为None
        :param nb_classes: 分类类别，默认为2
        :param dictionary_path: 词典，如果基于词的分类会根据训练集自动生成词典，如果基于字的分类，会默认使用char.list此时不用更改
        :param weights_path: 模型保存位置，如果已经存在同名的模型，会将其覆盖
        :param w2v_path: 预训练word2vec位置，注意区分是基于词还是基于字的分类，默认不使用(w2v_path=None)
        :param maxlen: 单个样本长度，超过该长度会截断，不足会进行补齐，默认基于字为200，基于词为70
        """

        self.tc = keras_text_classifier(ftrain,
                                        ftest,
                                        nb_classes,
                                        dictionary_path,
                                        weights_path,
                                        w2v_path=w2v_path,
                                        embedding_binary=False,
                                        ischar=ischar,
                                        maxlen=maxlen)
        self.dictionary_path = dictionary_path

    def train(self):
        self.tc.train()
    def predict_batch(self):
        self.tc.testbatch()
    def loadmodel(self):
        return self.tc.loadmodel()
    def loaddict(self):
        return process_data.loaddict(self.dictionary_path)
    def predict(self, sentence, model, words):
        return self.tc.test(sentence, model, words)

