from textblob import TextBlob

def sentiment(text):
    return TextBlob(text).sentiment.polarity