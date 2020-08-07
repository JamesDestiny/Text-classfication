# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#基于SVM分类器的文本分类的sklearn实现
import pandas as pd#数据读取文件
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
from dataset.cnew_loader import *
from sklearn import metrics
from sklearn.model_selection import *

#CountVectorizer 类实现文本的词频统计以及向量化
"""
测试CountVectorizer类
"""
def test_CountVectorizer():
    vectorizer = CountVectorizer()
    test = ["I come to China to travel",
        "This is a car polupar in China",
        "I love tea and Apple ",
        "The work is to write some papers in science"]
    test2 = [
        "I am apple car in is",
        "How are you",
        "Gnls"
    ]
    print(len(test))
    print(vectorizer.fit_transform(test))#查看词频统计数据
    #print(vectorizer.fit_transform(test).toarray().shape)
    print(vectorizer.get_feature_names())
    print(vectorizer.fit_transform(test2))
    print(vectorizer.get_feature_names())
    return vectorizer.fit_transform(test),vectorizer.fit_transform(test2)
"""
测试该类
"""

#由于词频不一定能反映该词的重要性，例如to这个词，所以需要进一步的预处理来反映这个文本特征，预处理就是TF-IDF称为词频=逆文本频率
#如果一个词在很多的文本中出现，那么它的IDF值应该低
"""
测试TfidfVectorizer类
"""
def test_Tfid():
    transformer = TfidfTransformer()
    dat,dat2 = test_CountVectorizer()
    tfidf = transformer.fit_transform(dat)
    tfidf2 = transformer.fit_transform(dat2)
    #print(tfidf[0])
    print(tfidf2[0])
    print(tfidf2.shape)
    #print(tfidf.shape)
    tfidf = transformer.fit_transform(dat).toarray()
    #print(tfidf[0])
    #print(tfidf.shape)

"""
测试该类
"""
#定义数据获取函数
def get_data(path,mode=None):
    train_data = pd.read_csv(path+'/cnews.train.txt',names= ['title','content'],sep = '\t',engine= 'python',encoding='UTF-8')
    test_data = pd.read_csv(path + '/cnews.test.txt',names=['title', 'content'], sep='\t', engine='python',
                             encoding='UTF-8')
    val_data = pd.read_csv(path + '/cnews.val.txt', names=['title', 'content'], sep='\t', engine='python',
                             encoding='UTF-8')
    #print(train_data.tail(10))
    #print(val_data.tail(10))
    x_train = train_data['content']
    y_train = train_data['title']
    x_test= test_data['content']
    y_test = test_data['title']
    x_val = val_data['content']
    y_val = val_data['title']
    #设定不同的获取模式，通过调用不同mode函数获取不同类型集合
    if mode==0:
        return x_test,y_test
    elif mode==1:
        return x_train,y_train
    else :
        return x_val,y_val


#进行训练样本的处理分别有去除停用词的向量化和不去除停用词的向量化
def Verctorizer(mode = None,x = None):
    if mode == 0:
        vec = CountVectorizer()
        return vec.fit_transform(x)
    elif mode == 1:
        vec = CountVectorizer(analyzer='word',stop_words='english')
        return vec.fit_transform(x)
    else:
        print("向量化模式选择错误")

#向量化后进行TF-TDF处理
def tf_idf(x = None,mode=None):
    vector = Verctorizer(mode=mode,x=x)
    tfid = TfidfTransformer()
    return tfid.fit_transform(vector)

#构建SVM模型进行训练
def Svm_model(x = None,y=None):
    model = SVC(cache_size=12000)
    model.fit(x,y)
    return model

#通过构建的SVM模型进行预测,获得预测值
def Svm_pre(x=None,model = None):
    #预测模型特征选取与之同步选取n个特征
    pre_y = model.predict(x)
    return pre_y

#定义保存模型,使用pickle模块
def save_mode(model = None,path = None):
    with open(path,'wb') as f:
        pickle.dump(model,f)


#定义读取模型
def load_model(path = None):
    with open(path,'rb') as f:
        load = pickle.load(f)
    return load

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    time = time.time()
    #test_Tfid()
    train_path = "dataset/cnews.train.txt"
    vocab_path = "dataset/cnews.vocab.txt"
    x_train,y_train = getdata(train_path,vocabdir=vocab_path)
    #print(x_train.shape)
    #print(y_train.shape)
    test_path = "dataset/cnews.test.txt"
    x_test,y_test = getdata(test_path,vocabdir=vocab_path)
    #进行模型训练
    #print(x_train)
    print(y_train.shape)
    train_size = 5000
    x_train = lower_data(x_train,train_size)
    y_train =lower_data(y_train,train_size)
    #将onehot矩阵转换为一维矩阵
    y_train = onehot_transfer(y_train)
    y_test = onehot_transfer(y_test)
    print(x_train.shape,y_train.shape)
    #model = Svm_model(x = x_train,y=y_train)
    print("模型训练成功")
    #保存模型
    save_path = 'SVM.txt'
    #save_mode(model=model,path=save_path)
    print("模型保存成功")
    #读取模型进行预测
    new_model = load_model(path=save_path)
    print("模型读取成功")
    #进行测试集上的预测
    predict_y = Svm_pre(model=new_model,x=x_test)
    #predict_y = 0
    print(predict_y)
    print(y_test)
    #计算得分
    print(metrics.classification_report(y_test,predict_y))
    print("test accuracy_score ：{:.2f}%".format(accuracy_score(y_test,predict_y)*100))
    print(get_time_dif(time))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
