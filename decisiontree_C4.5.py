#-*- coding:utf-8 -*-
from math import log
import matplotlib.pyplot as plt
import operator
import csv  # 读CSV文

def createDataSet():# 创建数据集

    csvfile = file('tree_data_dellackvalue.csv', 'rb')
    reader = csv.reader(csvfile)
    dataSet = []
    testSet = []
    for line in reader:
        if reader.line_num == 1:  ###忽略第一行，第一行的行号是1，不是0
            del line[0] #删除第一个格，对于整个文件来说就是删除第一列
            del line[-1]#删除最后一个格，对于整个feature name来说就是删除类标签
            labels = []
            labels.extend(line)#形成标签list
        else:
            del line[0]
            if reader.line_num>100 and reader.line_num<701:
                dataSet.append(line)  #形成训练集
            else:
                testSet.append(line)#形成测试集,自带标签
    return dataSet, testSet,labels

############计算香农熵###############
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 计算实例的总数
    labelCounts = {}  # 创建一个数据字典，它的key是最后把一列的数值(即标签)，value记录当前类型（即标签）出现的次数
    for featVec in dataSet:  # 遍历整个训练集
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 初始化香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 计算香农熵
    return shannonEnt


#########按给定的特征划分数据#########
def splitDataSet(dataSet, axis, value):  # axis表示特征的索引　　value是返回的特征值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 抽取除axis特征外的所有的记录的内容
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


#######遍历整个数据集，选择最好的数据集划分方式########
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 获取当前实例的特征个数，一般最后一列是标签。
    baseEntropy = calcShannonEnt(dataSet)  # 计算当前实例的香农熵
    GainRate=0.0
    bestFeature = -1  # 这里初始化最佳的信息增益和最佳的特征
    for i in range(numFeatures):  # 遍历每一个特征　
        featList = [example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        splitInfo=0.0
        for value in uniqueVals:
            subDataSet  =splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            info=0.0
            if(prob!=0):
                info = log(prob,2)
            splitInfo-=prob*info
        newGain=baseEntropy-newEntropy
        if (splitInfo == 0):  # 修复溢出错误
            splitInfo = -0.9999 * log(0.9999, 2) - 0.0001 * log(0.0001, 2)
        newGainRate=newGain/splitInfo
        if (newGainRate >=GainRate):
                GainRate= newGainRate
                bestFeature = i
    return bestFeature,newGainRate  # 返回一个整数，注意和外层的for对齐，这很重要

######多数表决######
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 返回所有的标签，即抽出所有数据行的最后一列即标签列
    if classList.count(classList[0]) == len(classList):  # 当类别完全相同时则停止继续划分，直接返回该类的标签
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有的特征时，仍然不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)  # 由于无法简单的返回唯一的类标签，这里就返回出现次数最多的类别作为返回值
    bestFeat,newGainRate = chooseBestFeatureToSplit(dataSet)  # 获取最好的分类特征索引
    if newGainRate<0.025:
        return majorityCnt(classList)
    bestFeatLabel = labels[bestFeat]  # 获取该特征的名称
    # 这里直接使用字典变量来存储树信息，这对于绘制树形图很重要。
    myTree = {bestFeatLabel: {}}  # 当前数据集选取最好的特征存储在bestFeat中
    del (labels[bestFeat])  # 删除已经在选取的特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  #复制所有的标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

######测试分类集######
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):#如果还是一棵子树的话就继续分类
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

######以下为画树代码######
decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义文本框与箭头的格式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):#获取树节点的数目
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  #测试节点的数据类型是不是字典，如果是字典，则就需要递归的调用getNumLeafs()函数
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1   #如果节点不是字典，就是叶子节点
    return numLeafs

def getTreeDepth(myTree):#获取树节点的树的层数
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType): #绘制带箭头的注释
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',#createPlot.ax1会提供一个绘图区
             xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def plotMidText(cntrPt, parentPt, txtString):#计算父节点和子节点的中间位置，在父节点间填充文本的信息
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #首先计算树的宽和高
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)#标记子节点的属性值
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:  #是叶节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))#c存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))#存储树的深度。我们使用这两个变量计算树节点的摆放位置
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()



myDat,testDat,labels=createDataSet()

nowlabels = labels[:]#先复制下所有的标签，以免在建树以后有的标签被删除
myTree=createTree(myDat,labels)
print  "myTree is",myTree
print "训练集数目为：", len(myDat)
print "测试集数目为：", len(testDat)
correct = 0

for test_data in testDat:
    testVec=test_data[0:-1]
    label = classify(myTree, nowlabels, testVec)
    if (label == test_data[-1]):# or (label == 'NULL'):
        correct += 1
print "其中匹配正确的为:",correct
print "准确率: ", correct / float(len(testDat))
createPlot(myTree)


