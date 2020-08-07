import sys
from collections import Counter
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from datetime import timedelta
import random
#根据训练即构建词汇表

def read_file(filename):
    train_data = pd.read_csv(filename, names=['title', 'content'], sep='\t', engine='python',
                             encoding='UTF-8')
    data_train = train_data['content']
    _ = train_data['title']
    return data_train,_

def build_vocab(train_dir,vocab_dir,vocab_size=5000):
    """
    构建词汇表存储
    :param train_dir: 训练集的路径文件
    :param vocab_dir: 词汇表的路径文件
    :param vocab_size: 词汇表的大小
    :return:
    """
    train_data = pd.read_csv(train_dir + '/cnews.train.txt', names=['title', 'content'], sep='\t', engine='python',
                             encoding='UTF-8')
    data_train = train_data['content']
    _ = train_data['title']
    all_data = []
    for content in  data_train:
        all_data.extend(content)#在列表末尾一次性追加多个值
    counter  =Counter(all_data)#counter使用的时计数器操作
    count_pairs = counter.most_common(vocab_size-1)#统计最长出现的词
    words,_ = list(zip(*count_pairs))
    #添加一个<PAD>来将所有的文本pad为同一长度
    words = ['<PAD>']+list(words)
    open(vocab_dir,'w').write('\n'.join(words)+'\n')


def read_vocab(vocab_dir):
    """
    读取词汇表
    :param vocab_dir:
    :return:
    """
    with open(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]#移除空格或者换行符
    word_to_id = dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    读取分类目录，固定的十个类别
    :return:
    """
    categories = ['体育','财经','房产','家居','教育','科技','时尚','时政','游戏','娱乐']
    cat_to_id = dict(zip(categories,range(len(categories))))
    return categories,cat_to_id


def to_words(content,words):
    """
    将id表示的内容转换为文字
    :param content:
    :param words:
    :return:
    """
    return ''.join(words[x] for x in content)

def file_to_id(filename,word_to_id,cat_to_id,max_length=400):
    """
    将文件转换为id表示
    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param maxlength:
    :return:
    """

    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    print(len(max(data_id)))
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = tf.keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def getdata(traindir,vocabdir):
    """
    生成数据
    :param x:
    :param y:
    :return:
    """
    word, word_to_id = read_vocab(vocabdir)
    cat, cat_to_id = read_category()
    x_pad ,y_pad = file_to_id(traindir,word_to_id,cat_to_id)
    return x_pad,y_pad

def lower_data(x,size):
    if(size<=5000):
        x0 = x[0:size]
        x1 = x[5000:5000+size]
        x2 = x[10000:10000+size]
        x3 = x[15000:15000+size]
        x4 = x[20000:20000+size]
        x5 = x[25000:25000+size]
        x6 = x[30000:30000+size]
        x7 = x[35000:35000+size]
        x8 = x[40000:40000+size]
        x9 = x[45000:45000+size]
        rel = np.vstack((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9))
    else:
        print("error data overflow")
    return rel

def random_getdata(x,size):#随机抽取矩阵中的某size行
    row = np.arange(x.shape[0])
    np.random.shuffle(row)
    row_rand  = x[row[0:size]]
    return row_rand

def onehot_transfer(y):
    """
    将one-hot矩阵转换为1维矩阵标签
    :param y:
    :return:
    """
    new_y = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if(y[i][j]==1):
                new_y[i]=j
    return new_y


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

def get_random_data(test_dir):
    #该函数为从测试集随机抽取一条数据
    #首先获取该文本数据
    content,label = read_file(test_dir)
    index = random.randint(0,content.shape[0])
    return content[index]

#将单一文本内容转换为测试集能输入的词向量
def transfer_toVector(input,max_length=400,vocabdir=None):
    #print(len(input))
    contents = []
    contents.append(input)
    #print(len(contents))
    data_id = []
    word, word_to_id = read_vocab(vocabdir)
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad
"""测试函数"""
