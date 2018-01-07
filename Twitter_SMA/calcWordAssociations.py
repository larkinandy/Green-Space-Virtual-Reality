#calcWordAssociatons.py
# created by Andrew Larkin
# for Scoial Media Analytics course project
# December 5, 2017

# This script calculates the frequency that words of interest are associated with keywords used during the Twitter search.
# Input files are part of speech tags and associations created by the Stanford NLP lexicon parser

# import modules and perform setup
import numpy as np
import nltk as nt
stemmer = nt.SnowballStemmer("english")
KEYWORDS_SINGULAR = ['park','tree','nature','bush','grass','flower','plant','garden','yard','backyard','leaf','forest','trail','mountain',
                     'lawn','field','crop','hay','prarie','pasture','lake','river','riverside','stream','branch']

# create stem for each keyword
for i in range(0,len(KEYWORDS_SINGULAR)):
    KEYWORDS_SINGULAR[i] = stemmer.stem(KEYWORDS_SINGULAR[i])

inputFolder = "C:/users/larkinan/desktop/taggedFiles/"
outputFile = "C:/users/larkinan/desktop/numEvents.csv"
keywords = ['view','sunset','green','beauti','run','hike','play','walk']
allList = []
allCounts = []

# process single association from input dataset
def processLine(inLine):
    vals = ["a"]*3
    endChars = ["(","-","-"]
    startChars = ["("," ",","]
    startIndex = 0

    for i in range(0,3) :
        endIndex = inLine.find(endChars[i],startIndex)
        vals[i] = inLine[startIndex:endIndex]
        startIndex = inLine.find(startChars[i],endIndex) + 1
    vals[2] = stemmer.stem(vals[2]).lower()
    vals[1] = stemmer.stem(vals[1]).lower()
    if(vals[2].lower() in KEYWORDS_SINGULAR or vals[1].lower() in KEYWORDS_SINGULAR):
        if(vals[2] in allList):
            allCounts[allList.index(vals[2])]+=1
        else:
            allList.append(vals[2])
            allCounts.append(1)
        if(vals[1] in allList):
            allCounts[allList.index(vals[1])]+=1
        else:
            allList.append(vals[1])
            allCounts.append(1)        
    tempRelation = relation(vals[1].lower(),vals[2].lower(),vals[0].lower())    
    return(tempRelation)

# read input file and calculate word frequencies
def readFile(filename):
    sentNum = 0
    fStream = open(filename,'r')
    tempData = fStream.readline()
    stillData = True
    needList = True
    needSentence = True
    while(tempData not in [""]):
        sentNum+=1
        tempRelation = processLine(tempData)
        if(needSentence):
            tempSentence = sentenceType(tempRelation)
            needSentence = False
        else:
            tempSentence.addRelation(tempRelation)
        if(needList == True):
            sentList = sentenceList(tempSentence)
            needList = False
        else:
            sentList.addSentence(tempSentence)
        tempData = fStream.readline()
    return(sentList)


# custom class for part of speech word associations.  Stores both words and association label
class relation:
    # create a word relation
    def __init__(self,inWord1,inWord2='NaN',inType='NaN'):
        self.word1 = inWord1
        self.word2 = inWord2
        self.valType = inType

    def getWord1(self):
        return self.word1

    def getWord2(self):
        return self.word2

    def getValType(self):
        return self.valType

# all text in a tweet.  
class sentenceType:
    # instantiate a hotspot
    def __init__(self,inRelation):
        self.relationList = []
        self.relationList.append(inRelation)
        self.wordList = []
        self.wordList.append(inRelation.getWord1())
        self.wordList.append(inRelation.getWord2())
        self.numRelations = 1

    def getRelation(self,index):
        return(self.relationList[index])

    def addRelation(self,inRelation):
        self.relationList.append(inRelation)
        if(inRelation.getWord1() not in self.wordList):
            self.wordList.append(inRelation.getWord1())
        if(inRelation.getWord2() not in self.wordList):
            self.wordList.append(inRelation.getWord2())
        self.numRelations +=1

    def getWordList(self):
        return self.wordList

    def getNumRelations(self):
        return(self.numRelations)

class sentenceList:

    def __init__(self,inSentence):
        self.sentenceList = []
        self.sentenceList.append(inSentence)

    def addSentence(self,inSentence):
        self.sentenceList.append(inSentence)

    def getSentence(self,index):
        return self.sentenceList[index]

    def getRelationsWithKeyword(self,keyword,personalLimit=0,allWords=False):
        tempList = self.getSentenceWithKeyword(keyword,personalLimit,allWords)
        tempRelationList = []
        for index in range(0,len(tempList)):
            tempSentence = tempList[index]
            for j in range(0,tempSentence.getNumRelations()):
                tempRelation = tempSentence.getRelation(j)
                if(tempRelation.getWord1() in keyword or tempRelation.getWord2() in keyword):
                    tempRelationList.append(tempRelation)
        return(tempRelationList)

    def getSentenceWithKeyword(self,keyword,personalLimit,allWords=False):
        tempList = []
        tempSentenceList = []
        for index in range(0,len(self.sentenceList)):
            tempSentence = self.sentenceList[index]
            if(allWords == True):
                foundWord = False
                for index in range(0,len(KEYWORDS_PLURAL)):
                    if(not foundWord):
                        if(KEYWORDS_PLURAL[index] in tempSentence.getWordList() or KEYWORDS_SINGULAR[index] in tempSentence.getWordList()):
                            if(tempSentence.getPersonal() >= personalLimit):
                                tempSentenceList.append(tempSentence)
                                foundWord = True
            else:
                if(keyword[0] in tempSentence.getWordList() or keyword[1] in tempSentence.getWordList()):
                    if(tempSentence.getPersonal() >= personalLimit):
                        tempSentenceList.append(tempSentence)

        return(tempSentenceList)

# keywords used during Twitter search
class keyWord:
    def __init__(self,inKeyWord,inSentList,personalLimit,allVals=False):
        self.keyWord = inKeyWord
        self.relationList = inSentList.getRelationsWithKeyword(inKeyWord,personalLimit,allVals)
        self.tempWordList = []
        self.wordFreq = []
        self.setWordFreq()

    def getKeyword(self):
        return self.keyWord

    def setKeyword(self,inKeyword):
        self.keyWord = inKeyword

    def getWordFreq(self):
        return self.wordFreq

    def getWordList(self):
        return self.tempWordList

    def getSampSize(self):
        return self.numSentences


    def main():

        # setup output file with head and iteratively write output for each loop
        keyCounts = [0]*len(keywords)
        fStream = open(outputFile,'w')
        for varName in keywords:
            fStream.write(varName)
            fStream.write(',')
        fStream.write('\n')

        # one file for each weeks worth of tweets
        for inputFile in inputFiles:   
            parserFile = inputFolder + inputFile
            sentList = readFile(parserFile)
            #sentList.screenForSentiment('human',-1)
            index = 0
            for keyword in keywords:
                if(keyword in allList):
                    tempIndex = allList.index(keyword)
                    keyCounts[index] = allCounts[tempIndex]  -  keyCounts[index] 
                    allCounts[tempIndex] = keyCounts[index] 
                    fStream.write(str(keyCounts[index]))
                else:
                    fStream.write('0')
                fStream.write(',')
                index +=1
            fStream.write('\n')

        fStream.close()

        print(keyCounts)
        print("completed main function")

# end of calcWordAssociations_v2.py