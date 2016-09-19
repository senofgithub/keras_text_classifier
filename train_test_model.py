# coding:utf-8
"""
模型训练与测试
"""
import keras_text_categorization as ktc

"""
1. 初始化参数
"""
# 训练数据路径，格式为： <label><tab><content>，预测时不需要指定该参数，设置为None
# label：从0开始的连续数字
# tab：tab分割
# content: 文本数据，如果使用基于词的分类，文本需要分词
ftrain = "data/train"
# 测试数据路径，格式同训练数据，单句预测时不需要指定该参数，设置为None
ftest = "data/test"
# 分类类别
nb_classes = 2
# 词典，如果基于词的分类会根据训练集自动生成词典，如果基于字的分类，会默认使用char.list此时不用更改
dictionary_path = "dict/char.list"

# 模型保存位置，如果已经存在同名的模型，会将其覆盖
weights_path = "model/cnn_sent_char.h5"

# 预训练word2vec位置，注意区分是基于词还是基于字的分类，默认不使用(w2v_path=None)
w2v_path = None#"data/bbs100.vec"

# 参数初始化
"""
a. 默认使用基于字的cnn模型(ischar=True)，如果基于词需设置ischar=False
b. 如果不使用预训练word2vec，需设置w2v_path=None
c. 如果使用预训练word2vec，且为文本格式需设置embedding_binary=False
d. maxlen参数为单个样本长度，超过该长度会截断，不足会进行补齐，默认基于字为200，基于词为70
"""
tc = ktc.keras_text_classifier(train_path=ftrain,
                               test_path=ftest,
                               nb_classes=nb_classes,
                               dictionary_path=dictionary_path,
                               weights_path=weights_path,
                               w2v_path=w2v_path,
                               embedding_binary=False,
                               ischar=True,
                               maxlen=200)

"""
2. 模型训练，目前的模型结构为1层卷积+1层池化+2层softmax
"""
tc.train()

# TODO：预测时maxlen需要与训练集保持一致
"""
3. 批量测试，即读取已经训练好的模型对测试集进行预测
"""
tc.testbatch()

"""
4. 测试单个句子，输出每个类别的概率
"""
# 如果基于词，需要分词
#sentence = u'内饰看起来不是很协调'     # 基于字
sentence = u'发动机 异响'  # 基于词
# 读取模型
model = tc.loadmodel()
# 读取词典
words = tc.loaddict()
# 输出概率
print(tc.test(sentence, model, words))
