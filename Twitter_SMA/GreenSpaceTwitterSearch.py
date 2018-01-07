# GreeenSpaceTwitterSearch.py
# created by Andrew Larkin
# for Scoial Media Analytics course project
# December 5, 2017

# Downloads, stores and exports tweets related to green space

import json
import tweepy
import dataset

# setup Twitter access credentials
access_token = "insert token"
access_token_secret = "insert access token"
consumer_key = "consumer key"
consumer_secret = "consumer secret"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
db = dataset.connect("sqlite:///C:/users/user/Desktop/TwitterTemp/annualTweetsp1.db")
CONNECTION_STRING = "sqlite:///C:/users/user/Desktop/TwitterTemp/annualTweetsp1.db"
CSV_NAME = "tweets.csv"
TABLE_NAME = "greenTweets"

greenSerachTerms = [u'park',u'parks',u'tree' 'trees',u'nature',u'bush',u'bushes',u'grass',u'flower',u'flowers',u'plant',u'plants',u'garden',u'yard',u'backyard',u'leaves',u'forest',u'trail',u'mountain',u'lawn',u'field',u'crop',u'hay',
                    u'prarie',u'pasture',u'lake',u'lakes',u'river',u'rivers',u'riverside',u'stream',u'streams']

# custom listener class
class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        loc = status.user.location
        text = status.text
        language = status.lang
        coords = status.coordinates
        created = status.created_at
        if coords is not None:
            coords = json.dumps(coords)
        table = db["allTweets"]
        table.insert(dict(
            user_location=loc,
            coordinates=coords,
            text=text,
            created=created,
            language = language
            ))

    def on_error(self, status_code):
        #print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        #print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

# export tweets to CSV
def dumpTweets():
    result = db["allTweets"].all()
    print(result)
    print(dataset.freeze(result, format='csv', filename=CSV_NAME))
    print("finished")
    

def main():
    sapi = tweepy.streaming.Stream(auth, CustomStreamListener())
    
    testVals = True
    while(testVals):
        try:
            searchTerms = greenSerachTerms
            print(searchTerms)
            sapi.filter(track=searchTerms)
        except Exception as e:
            print(str(e))
    
main()
    
# end of GreenSpaceTwitterSearch.py
