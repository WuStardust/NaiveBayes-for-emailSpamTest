from numpy import *
import os


def loadDataSet(dir, rate):
    from random import uniform

    docList = []
    classList = []
    rootDir = dir

    list = os.listdir(rootDir)
    for i in range(0, len(list)):
        path0 = os.path.join(rootDir, list[i])
        for filePath in os.listdir(path0):
            path = os.path.join(path0, filePath)
            if os.path.isfile(path):
                seed = uniform(0, 1)
                if seed <= rate:
                    wordList = textParse(open(path, errors='ignore').read())
                    docList.append(wordList)
                    if path.find("spam") != -1:
                        classList.append(1)
                    elif path.find("ham") != -1:
                        classList.append(0)

    return docList, classList


def textParse(bigString):
    import re
    listOfToken = re.split(r'\W*', bigString)

    stopWords = []
    for line in open("stopWords.txt"):
        line = line.replace('\n', '')
        stopWords.append(line)

    return \
        [tok.lower() for tok in listOfToken if tok.lower() not in stopWords and len(tok) > 3 and tok.isdigit() is False]


def createVocabList(dataSet):
    thresh = 0.005 * len(dataSet)

    vocabSet = set([])
    wordFreq = {}
    for document in dataSet:
        for word in document:
            wordFreq[word] = wordFreq.get(word, 0) + 1
        vocabSet = vocabSet | set(document)

    lowFreqWord = set([])
    for word in vocabSet:
        if wordFreq.get(word, 0) < thresh:
            lowFreqWord.add(word)

    vocabSet = vocabSet - lowFreqWord

    return list(vocabSet)


def createTrainMat(rate=1):
    # print('load train data begin')
    trainDocList, trainClassList = loadDataSet("hw1_data/train", rate)
    # print('load train data complete')

    # print("create vocabulary begin")
    vocabList = createVocabList(trainDocList)
    print(len(vocabList))
    # print("create vocabulary complete")

    trainLength = len(trainDocList)

    file1 = open('trainMat.txt', 'w')
    file2 = open('trainClasses.txt', 'w')
    for i in range(trainLength):
        vec2train = bagOfWords2Vec(vocabList, trainDocList[i])
        trainClass = trainClassList[i]
        file1.write(str(vec2train) + '\n')
        file2.write(str(trainClass) + '\n')
        # print('append ', i)
    file1.close()
    file2.close()

    file = open('vocabList.txt', 'w')
    file.write(str(vocabList))
    file.close()

    print('trainMat created')


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = ones(numWords)
    p1Num = ones(numWords)

    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec


def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)

    if p1 > 1.01 * p0:
        return 1
    else:
        return 0


def trainModel():
    print("load train mat begin")
    scope = {}
    trainMat = []
    trainClasses = []
    file = open('trainMat.txt', 'r')
    for line in file:
        line = line.replace('\n', '')
        exec('vec2train = ' + line, scope)
        vec2train = scope['vec2train']
        trainMat.append(vec2train)
    file = open('trainClasses.txt', 'r')
    for line in file:
        exec('trainClass = ' + line, scope)
        trainClass = scope['trainClass']
        trainClasses.append(trainClass)
    print("load train mat complete")

    # print("train begin")
    p0V, p1V, pSpam = trainNB(array(trainMat), array(trainClasses))
    # print("train complete")

    p0V.tofile('p0V.bin')
    p1V.tofile('p1V.bin')
    file = open('pSpam.txt', 'w')
    file.write(str(pSpam))
    file.close()
    print('train complete')

    return p0V, p1V, pSpam


def spamTest(rate=1):
    print('load trainModel')
    scope = {}
    pSpam = 0
    vocabList = []

    p0V = fromfile("p0V.bin")
    p1V = fromfile("p1V.bin")

    file = open('pSpam.txt', 'r')
    exec('pSpam = ' + file.read(), scope)
    pSpam = scope['pSpam']
    file.close()

    file = open('vocabList.txt', 'r')
    exec('vocabList = ' + file.read(), scope)
    vocabList = scope['vocabList']
    file.close()

    # p0V, p1V, pSpam = trainModel()
    print('load test data')
    testDocList, testClassList = loadDataSet("./hw1_data/test", rate)

    FPcount = 0
    recallCount = 0
    Pcount = 0
    Ncount = 0
    accuracy = 0

    print("test begin")
    testLength = len(testDocList)
    for i in range(testLength):
        testVector = bagOfWords2Vec(vocabList, testDocList[i])
        testClass = classifyNB(array(testVector), p0V, p1V, pSpam)
        # print("the classifier came back with: %d, the real answer is : %d" % (testClass, testClassList[i]))
        if testClass == 1:
            accuracy += 1
        if testClassList[i] == 1:
            Pcount += 1
            if testClass == 1:
                recallCount += 1
        elif testClassList[i] == 0:
            Ncount += 1
            if testClass == 1:
                FPcount += 1

    print("test complete")

    print("testLength = ", testLength, end='   ')
    print("recallRate = ", float(recallCount) / Pcount, end='   ')
    print("FPRate = ", float(FPcount) / Ncount, end='   ')
    print("accuracy = ", float(recallCount) / accuracy)

    return recallCount, Pcount, FPcount, Ncount, accuracy
