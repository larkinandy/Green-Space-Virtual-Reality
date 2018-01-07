# calcWeekly_tfidf
# created by Andrew Larkin
# for Scoial Media Analytics course project
# December 5, 2017

# This script partitions Tweets from a raw csv file into weekly subsets,
# writes the subsets to text files, and calculates weekly idf scores
# for for each subset using trigrams

# import modules 
import pandas as ps
from datetime import datetime
from copy import deepcopy as dc
from collections import defaultdict
import nltk as nk
from sklearn.feature_extraction.text import TfidfVectorizer

# define input and output
outputFolder = "C:/users/user/desktop/"
inputFile = "C:/users/larkinan/desktop/testOutputNoHashtag.csv"


def getDayOfYear(dateString):
    return int(datetime.strptime(dateString,"%Y-%m-%dT%H:%M:%S").strftime('%j'))

# if a string contains a hyperlink, find the location and remove for downstream text processing
def findHttp(inputString):
    httpLoc = str.find(inputString,"http")
    if(httpLoc > -1):
        return(inputString[0:httpLoc])
    else:
        return(inputString)

# partition tweet dataset by day 
def getTextForDay(dataset):
    accumDayText = ""
    numObs = 0
    numTweets = len(dataset['text'])
    stringList = list(dataset['text'])
    for singleText in range(0,numTweets):
        newString = stringList[singleText]
        newString = findHttp(newString)
        if(newString not in accumDayText):
            
            # test if the new Tweet content is not a duplicate
            shouldCopy = True
            words1 = list((newString.split()))
            testList = ' '.join(words1[1:max(len(words1)-3,3)])
            
            # text is unique add to daily tweet set
            if(testList in accumDayText):
                shouldCopy = False
            if(shouldCopy):
                accumDayText += newString
                accumDayText += "\n"   
                numObs +=1
            
    return([accumDayText,numObs])

# patition tweet dataset by week
def partitionWeeklyTweets(rawData):
    rawData['dayOfYear'] = list(map(getDayOfYear, rawData['created']))
    uniqueDays = list(set(rawData['dayOfYear']))
    accumTexts = []    
    accumWeekTexts = [""]
    numObs = [0]*len(uniqueDays)
    weekCounter = 0
    weekIndex = 0
    numWeekley = 0

    for dayIndex in range(0,len(uniqueDays)):
        tempCopy = dc(rawData)
        tempData = tempCopy.loc[tempCopy['dayOfYear'] == uniqueDays[dayIndex]]
        [dayText,numObs[dayIndex]] = getTextForDay(tempData)
        accumTexts.append(dayText)
        accumWeekTexts[weekIndex] += dayText
        weekCounter+=1
        numWeekley += numObs[dayIndex]
        # if new week, rest counter and start adding daily tweets to new week subset
        if(weekCounter == 7):
            weekIndex+=1
            weekCounter = 0
            print("num weekley" + str(weekIndex) + " : " + str(numWeekley))
            numWeekley = 0
            accumWeekTexts.append("")
    
    print(sum(numObs))
    print(numObs)
    for index in range(0,len(numObs)):
        if(numObs[index] < 100):
            print(index)
    print(len(numObs))
    return(accumWeekTexts)

def writeWeeklySet(accumWeekTexts):
    for weekText in accumWeekTexts:
        f = open(outputFolder + "week" + str(index) + ".txt",'w')
        f.write(weekText)
        f.close()
        index+=1    

# calculate tfidf scores for entire dataset, grouped by week.  Based on script created by Mark Needham
# http://www.markhneedham.com/blog/2015/02/15/pythonscikit-learn-calculating-tfidf-on-how-i-met-your-mother-transcripts/
def calc_tfidf(accumTexts):
    tfidf_matrix =  tf.fit_transform(accumTexts)
    feature_names = tf.get_feature_names() 
    print(len(feature_names))
    dense = tfidf_matrix.todense()
    for weekIndex in range(0,len(accumTexts)):
        print('values for week' + str(weekIndex))
        weekSet = dense[weekIndex].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(weekSet)), weekSet) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:10]:
            print('{0: <10} {1}'.format(phrase, score))

# main script.  Controls program directoin
def main():
    rawData = ps.read_csv(inputFile,encoding='latin')    
    accumWeekTexts = partitionWeeklyTweets(rawData)
    writeWeeklySet(accumWeekTexts)
    calc_tfidf(accumWeekTexts)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(3,3),min_df = 0,stop_words = 'english',use_idf=False)
    index = 0


main()


# end of calcWeekly_tfidf