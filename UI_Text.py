from UI.text import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize
import tensorflow as tf
from dataset.cnew_loader import *
from SVM import *

categories = ['体育','财经','房产','家居','教育','科技','时尚','时政','游戏','娱乐']
MODE_SELECT = 1#设置两种不同的输入模式
MODE_INPUT = 2
vdir = "dataset/cnews.vocab.txt"#设置词汇表路径
class window(QMainWindow,Ui_Form):
    def __init__(self):
        super(window, self).__init__()
        self.setupUi(self)
        self.center()
        self.cbBox_Mode_Callback()
        self.cbBox_Mode_2_Cakllback()
        self.mode = 1
        self.model = 0
        self.label_4.setText("包含分类类别："+str(categories))
        self.new_model_0 = tf.keras.models.load_model("Text-CNN/Cnn_model")
        self.new_model_1 = load_model("SVM.txt")
        self.new_model_2 = tf.keras.models.load_model("Text-RNN/Rnn_model")

    # 窗口居中
    def center(self):
         # 获得窗口
        framePos = self.frameGeometry()
            # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
            # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())


# 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def clearDataArea(self):
        self.textBrowser.clear()
        self.textBrowser_2.clear()
        self.lineEdit.clear()
        self.pushButton_3.setEnabled(False)
# 模式下拉列表回调
    def cbBox_Mode_Callback(self):
        text = self.cbBox_Mode.currentText()
        if text == '<1>：测试集随机抽取':
            self.pushButton_5.setEnabled(True)
            self.mode = MODE_SELECT
            self.clearDataArea()
            self.pushButton_2.setEnabled(False)
            self.lineEdit.setEnabled(False)
            self.mode=1

        elif text == '<2>：输入内容识别':
            self.mode = MODE_INPUT
            self.clearDataArea()
            self.pushButton_2.setEnabled(True)
            self.lineEdit.setEnabled(True)
            self.pushButton_5.setEnabled(False)
            self.mode =2
#模型下拉回调
    def cbBox_Mode_2_Cakllback(self):
        text = self.cbBox_Mode_2.currentText()
        if text=="<1>：神经网络模型":
            self.textBrowser.clear()
            self.model =0
        elif text=="<2>：SVM模型":
            self.textBrowser.clear()
            self.model =1
        elif text=="<3>：简单RNN模型":
            self.textBrowser.clear()
            self.model =2
#清除文本内容
    def pB_Callback(self):
        if self.textBrowser_2.toPlainText()=="":
            QMessageBox.information(self, "提示", "Text is empty")
        else:
            self.textBrowser_2.clear()
            self.textBrowser.clear()
#清空输入框
    def pB_6_Callback(self):
        if self.lineEdit.text()=="":
            QMessageBox.information(self,"提示","Text is empty")
        else :
            self.lineEdit.clear()
#输入按钮
    def pB_2_Callback(self):
        if(self.lineEdit.text()!=""):
            self.textBrowser.clear()
            #获得输入文本框的输入
            input = self.lineEdit.text()
            #将文本框内容输入到文本内容中
            self.textBrowser_2.setText("输入的文本为：\n"+input)
        else:
            QMessageBox.information(self, "提示", "Input is empty")

#分类按钮，进行分类
    def pB_3_Callback(self):
        #首先获取文本内容,先要判断文本内容是否为空
        global eff2, eff
        if self.textBrowser_2.toPlainText()=="":
            QMessageBox.information(self, "提示", "Text is empty")
        else:
            input_x = self.textBrowser_2.toPlainText()
            #将该内容转换为词向量
            #用模型训练获取输出
            #判断该输出代表的类别
            #输出分类结果到文本中
            if self.mode ==1:
                sentence = input_x.strip("随机抽取的文本为：\n")
            else :
                sentence = input_x.strip("输入的文本为：\n")
            x = transfer_toVector(sentence,vocabdir=vdir)
                #print(x.shape)
            #进行模型选择
            if self.model ==0:#选定神经网络模型
                classes = self.new_model_0.predict(x)
                result = categories[np.argmax(classes)]
                self.textBrowser.setText("文本分类结果："+result+"\n模型：{}".format(self.cbBox_Mode_2.currentText().replace("<1>：","")))
            elif self.model ==1:  #选定SVM模型
                classes = self.new_model_1.predict(x)
                #print(classes[0])
                result = categories[int(classes[0])]
                self.textBrowser.setText("文本分类结果："+result+"\n模型：{}".format(self.cbBox_Mode_2.currentText().replace("<2>：","")))
            elif self.model == 2:#选定简单RNN模型
                classes = self.new_model_2.predict(x)
                result = categories[np.argmax(classes)]
                self.textBrowser.setText("文本分类结果："+result+"\n模型：{}".format(self.cbBox_Mode_2.currentText().replace("<3>：","")))



#分类结果清除按钮
    def pB_4_Callback(self):
        if self.textBrowser.toPlainText()!='':
            self.textBrowser.clear()
        else:
            QMessageBox.information(self,"提示","Text is empty")

#定义抽取按钮
    def pB_5_Callback(self):
        self.textBrowser.clear()
        #从测试集中随机抽取一个文本内容
        R_select_text = get_random_data("dataset/cnews.test.txt")
        self.textBrowser_2.clear()
        #将该文本内容添加到其中
        self.textBrowser_2.setText("随机抽取的文本为：\n"+R_select_text)
#定义文本内容改变回调函数
    def tB_2_Callbaack(self):
        if self.textBrowser_2.toPlainText()=="":
            self.pushButton_3.setEnabled(False)
        else:
            self.pushButton_3.setEnabled(True)


if __name__ == "__main__":
    gui =QApplication(sys.argv)
    app = window()
    app.show()
    sys.exit(gui.exec_())
