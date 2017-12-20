import pandas as ps
from datetime import datetime
from copy import deepcopy as dc
from collections import defaultdict
import csv



from sklearn.feature_extraction.text import TfidfVectorizer


def getDayOfYear(dateString):
    return int(datetime.strptime(dateString,"%Y-%m-%dT%H:%M:%S").strftime('%j'))




rawData = ps.read_csv("D:/Dropbox/Dropbox/smalltest2.csv")
print(rawData.head())
#= rawData['created']




rawData['dayOfYear'] = list(map(getDayOfYear, rawData['created']))

def findHttp(inputString):
    httpLoc = str.find(inputString,"http")
    if(httpLoc > -1):
        return(inputString[0:httpLoc])
    else:
        return(inputString)

def getTextForDay(dataset):
    accumDayText = ""
    numTweets = len(dataset['text'])
    stringList = list(dataset['text'])
    for singleText in range(0,numTweets):
        newString = stringList[singleText]
        newString = findHttp(newString)
        accumDayText += newString
        accumDayText += "\n"   
    return(accumDayText)




uniqueDays = list(set(rawData['dayOfYear']))
print(uniqueDays)
accumTexts = []    
for dayIndex in range(0,len(uniqueDays)):
    tempCopy = dc(rawData)
    tempData = tempCopy.loc[tempCopy['dayOfYear'] == uniqueDays[dayIndex]]
    dayText = getTextForDay(tempData)
    accumTexts.append(dayText)
print(accumTexts[0])


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform(accumTexts)
feature_names = tf.get_feature_names() 
print(len(feature_names))
dense = tfidf_matrix.todense()
for dayIndex in range(0,len(uniqueDays)):
    episode = dense[dayIndex].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
        print('{0: <20} {1}'.format(phrase, score))
    
print('a')

