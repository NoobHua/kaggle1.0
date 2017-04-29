#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/29 16:46
# @Author  : Gene Lee
# @Site    : https://github.com/NoobHua/machinelearning
# @File    : testste.py
# @Software: PyCharm
# knn


from numpy import *
import operator
import csv
import os


# 将字符串转换为整数(从文件中读取的都是字符串，这里需要转换)
def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray


# 由于是对数字的识别，不需要知道像素值的具体的大小，所以直接使用1-0值即可
def nomalizing(array):
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


# 加载训练数据
def loadTrainData():
    l = []

    # 此段代码为判断文件是否存在的代码，可以删除
    # filename = 'train.csv'
    # if os.path.exists(filename):
    #     message = 'OK, the "%s" file exists.'
    # else:
    #     message = "Sorry, I cannot find the file "
    # print(message)

    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    # 将表格的头去掉
    l.remove(l[0])
    l = array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toInt(data)), toInt(label)  # label 1*42000  data 42000*784
    # return data,label


# 加载测试数据集
def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
            # 28001*784
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))  # data 28000*784


# 如果已经有结果了，可以使用这段代码进行导入，用以计算准确率
# def loadTestResult():
#     l = []
#     with open('knn_benchmark.csv') as file:
#         lines = csv.reader(file)
#         for line in lines:
#             l.append(line)
#             # 28001*2
#     l.remove(l[0])
#     label = array(l)
#     return toInt(label[:, 1])  # label 28000*1


# 进行分类
# dataSet:m*n   labels:m*1  inX:1*n
# inX: 待测数据
# dataSet: 训练集数据
# labels: 训练集的标签
# k: k近邻算法中的k值，意味着从训练集中寻找k个与测试样本距离最近的样本，这k个样本中出现频率最高的类别即作为测试样本的类别。
def classify(inX, dataSet, labels, k):
    inX = mat(inX)
    dataSet = mat(dataSet)
    labels = mat(labels)

    # 返回矩阵的行数m，参数为1的话返回矩阵的列数n
    dataSetSize = dataSet.shape[0]

    # 将数组inX作为元素构造出dataSetSize行，1列的矩阵，由于数组inX本身为1*n，所以最终形成一个和训练数据集完全等大的矩阵。
    # 做差之后，并平方求和，得出距离。
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = array(diffMat) ** 2

    # 将数组进行按行累加，参数为0的话按列累加
    # 得到最终待测样本和每个点的距离
    sqDistances = sqDiffMat.sum(1)
    distances = sqDistances ** 0.5

    # argsort得到矩阵中每个元素的排序序号
    # A=array.argsort()。 A[0]表示排序后，排在第一个的那个数在原来数组中的下标
    sortedDistIndicies = distances.argsort()

    # classCount为字典类型
    classCount = {}
    for i in range(k):
        # 得到训练集中和待测样本最近的数据的标记
        voteIlabel = labels[sortedDistIndicies[i], 0]
        if voteIlabel in classCount.keys():
            classCount[voteIlabel] = classCount.get(voteIlabel) + 1
        else:
            classCount[voteIlabel] = 1
    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1])
    return int(sortedClassCount[len(sortedClassCount) - 1][0])


# 保存运算结果
def saveResult(result):
    with open('result.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

# TEST
def handwritingClassTest():
    print('运行了handwritingClassTest......')
    trainData, trainLabel = loadTrainData()
    print('训练集读取成功........')
    testData = loadTestData()
    print('测试集读取成功........')
    # testLabel = loadTestResult()
    m, n = shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)
        print('对第',i,'个数据进行预测：',classifierResult)
        resultList.append(classifierResult)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0, i]))
        # if (classifierResult != testLabel[0, i]):
        #     errorCount += 1.0
    # print("\nthe total number of errors is: %d" % errorCount)
    # print("\nthe total error rate is: %f" % (errorCount / float(m)))
    saveResult(resultList)


'''
trainData[0:20000], trainLabel.transpose()[0:20000]
get 20000 of the 42000 samples to train
'''

handwritingClassTest();