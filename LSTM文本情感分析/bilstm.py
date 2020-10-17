# _*_ coding:utf-8 _*_

'''
新增功能：读入有标签但在一个文件夹里的数据，按正、中、负分好类
@Author: Ruan Yang
@Date: 2018.12.9
@Purpose: 文本情感分析(positive,negative,neutral)
@Reference: https://github.com/Edward1Chou/SentimentAnalysis
@算法：Bi_LSTM
@需要有事先标准好的数据集
@positive: [1,0,0]
@neutral: [0,1,0]
@negative:[0,0,1]
'''

import warnings
warnings.filterwarnings("ignore")

import codecs
import jieba
import datetime
start = datetime.datetime.now()
datapaths="G:\\研究生\\研一下课程\\CCF竞赛\\data\\"

positive_data=[]
y_positive=[]
neutral_data=[]
y_neutral=[]
negative_data=[]
y_negative=[]

print("#------------------------------------------------------#")
print("加载数据集")
import numpy as np
from pandas import Series
from pandas import DataFrame
import pandas as pd
with open(datapaths+"example.csv",encoding="utf-8") as f:
    df = pd.read_csv(f)
df1 = df['微博中文内容']
df2 = df['情感倾向']
row = len(df1)

for i in range(row):
    if df2[i]==1:
        line = str(df1[i])
        positive_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_positive.append([1,0,0])
    elif df2[i]==0:
        line = str(df1[i])
        neutral_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_neutral.append([0,1,0])
    elif df2[i]==-1:
        line = str(df1[i])
        negative_data.append(" ".join(i for i in jieba.lcut(line.strip(),cut_all=False)))
        y_neutral.append([0,0,1])
        
print("positive data:{}".format(len(positive_data)))
print("neutral data:{}".format(len(neutral_data)))
print("negative data:{}".format(len(negative_data)))

x_text=positive_data+neutral_data+negative_data
y_label=y_positive+y_neutral+y_negative
print("#------------------------------------------------------#")
print("\n")

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import collections

max_document_length=200     #文档最大长度
min_frequency=1             #词频最小值

# 创建词汇表
# 根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
# 例子https://blog.csdn.net/The_lastest/article/details/81771723
vocab = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency, tokenizer_fn=list)
x = np.array(list(vocab.fit_transform(x_text)))

# 根据放入元素的先后顺序进行排序,以dict形式输出
#                                                      以dict的形式输出 这是一个 词--->词表id（索引）的映射
vocab_dict = collections.OrderedDict(vocab.vocabulary_._mapping)

with codecs.open(r"G:\研究生\研一下课程\bilstm\cnn_lstm_cnnlstm_textcnn_bilstm\vocabulary.txt","w","utf-8") as f:
    for key,value in vocab_dict.items():
        f.write("{} {}\n".format(key,value))

print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("数据混洗")
np.random.seed(10)
y=np.array(y_label)
# 随机排列
shuffle_indices = np.random.permutation(np.arange(len(y)))  # 打乱顺序
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

test_sample_percentage=0.2
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

train_positive_label=0
train_neutral_label=0
train_negative_label=0
test_positive_label=0
test_neutral_label=0
test_negative_label=0

for i in range(len(y_train)):
    if y_train[i,0] == 1:
        train_positive_label += 1
    elif y_train[i,1] == 1:
        train_neutral_label += 1
    else:
        train_negative_label += 1

for i in range(len(y_test)):
    if y_test[i,0] == 1:
        test_positive_label += 1
    elif y_test[i,1] == 1:
        test_neutral_label += 1
    else:
        test_negative_label += 1

print("训练集中 positive 样本个数：{}".format(train_positive_label))
print("训练集中 neutral 样本个数：{}".format(train_neutral_label))
print("训练集中 negative 样本个数：{}".format(train_negative_label))
print("测试集中 positive 样本个数：{}".format(test_positive_label))
print("测试集中 neutral 样本个数：{}".format(test_neutral_label))
print("测试集中 negative 样本个数：{}".format(test_negative_label))

print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("读取預训练词向量矩阵")

pretrainpath=r"G:\研究生\研一下课程\bilstm\cnn_lstm_cnnlstm_textcnn_bilstm\\"

embedding_index={}  # 预训练词向量的字典 词->向量

'''
    本资源中的预训练词向量文件以文本格式存储。每一行包含一个单词及其词向量。
    每个值由空格分开。第一行记录元信息：第一个数字表示该单词在文件中的排序，
    第二个数字表示维度大小
'''
with codecs.open(pretrainpath+"sgns.wiki.bigram","r","utf-8") as f:
    #line=f.readlines()
    line=f.readline()
    nwords=int(line.strip().split(" ")[0])  #是否是总词数？
    ndims=int(line.strip().split(" ")[1])
    for line in f:
        values=line.split()
        words=values[0]
        coefs=np.asarray(values[1:],dtype="float32")
        embedding_index[words]=coefs

print("預训练模型中Token总数：{} = {}".format(nwords,len(embedding_index)))
print("預训练模型的维度：{}".format(ndims))
print("#----------------------------------------------------------#")
print("\n")

print("#----------------------------------------------------------#")
print("将vocabulary中的 index-word 对应关系映射到 index-word vector形式")

'''
    预训练的词向量中有的词加入embedding_matrix
    没有的词随机从-1 到 +1见生成三百个随机数作为词向量
'''
embedding_matrix=[]
notfoundword=0

for word in vocab_dict.keys():
    if word in embedding_index.keys():
        embedding_matrix.append(embedding_index[word])
    else:
        notfoundword += 1
        embedding_matrix.append(np.random.uniform(-1,1,size=ndims))     #从-1到+1随机采样

embedding_matrix=np.array(embedding_matrix,dtype=np.float32) # 必须使用 np.float32
print("词汇表中未找到单词个数：{}".format(notfoundword))
print("#----------------------------------------------------------#")
print("\n")

print("#---------------------------------------------------#")
print("Build model .................")
print("NN structure .......")
print("Embedding layer --- Bi_LSTM layer --- Dense layer")
print("#---------------------------------------------------#")
print("\n")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import LSTM

batch_size=256
max_sentence_length=200
embedding_dims=ndims
dropout=0.2
recurrent_dropout=0.2
num_classes=3
epochs=2

# 定义网络结构,此处或许可以加上正则化
model=Sequential()
model.add(Embedding(len(vocab_dict),
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=max_sentence_length,
                    trainable=False))
model.add(Dropout(dropout))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(dropout))
model.add(Dense(num_classes,activation="sigmoid"))  #输出结果的维度和激活函数

# 模型编译

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

print("#---------------------------------------------------#")
print("Train ....................")
print("#---------------------------------------------------#")
print("\n")

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))
# 画出train集的loss和epoch图（选用）
# from matplotlib import pyplot
# history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2)
# pyplot.plot(history.history['loss'])
# pyplot.plot(history.history['val_loss'])
# pyplot.title('model train vs validation loss')
# pyplot.ylabel('loss')
# pyplot.xlabel('epoch')
# pyplot.legend(['train', 'validation'], loc='upper right')
# pyplot.savefig('zzzz3.pdf')
# print('画图完成')


# 训练得分和准确度

score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测
# 返回值是数值，表示样本属于每一个类别的概率
predictions=model.predict(x_test)

print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
print(predictions)
print("#---------------------------------------------------#")
print("\n")

# 模型预测类别
# 返回的是类别的索引，即该样本所属的类别标签
predict_class=model.predict_classes(x_test)

print("#---------------------------------------------------#")
print("测试集的预测类别")
print(predict_class)
print("#---------------------------------------------------#")
print("\n")


print('开始计时')
end = datetime.datetime.now()
time = end-start
print('运行时间：{}'.format(time))

# 模型保存

# model.save(r"C:\Users\Administrator\Desktop\研一下课程\bilstm\cnn_lstm_cnnlstm_textcnn_bilstm\lstm.h5")
#
# print("#---------------------------------------------------#")
# print("保存模型")
# print("#---------------------------------------------------#")
# print("\n")
#
# 模型总结
# 打印出模型概况，它实际调用的是keras.utils.print_summary
print("#---------------------------------------------------#")
print("输出模型总结")
print(model.summary())
print("#---------------------------------------------------#")
print("\n")

# 模型的配置文件
# 返回包含模型配置信息的Python字典
config=model.get_config()

print("#---------------------------------------------------#")
print("输出模型配置信息")
print(config)
print("#---------------------------------------------------#")
print("\n")
