

#Description : This is a sentiment analysis program that parses the tweets fetched from Twitter using Python

#Import the libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re #regular expression
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Load the data
from google.colab import files
uploaded = files.upload()

# Get the data
log = pd.read_csv('Login.csv')

# Twitter API Credentials
consumerKey = log['key'][0]
consumerSecret = log['key'][1]
accessToken = log['key'][2]
accessTokenSecret = log['key'][3]


#create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Create the API.object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# Extract 100 tweets from the twitter user
posts = api.user_timeline(screen_name = 'screen_name_here', count = 100, lang = "en", tweet_mode="extended")


#print the last 5 tweets
print("Displaying the 5 recent tweets: \n")
i=1
for tweet in posts[0:5]:
#for tweet in posts:
  print(str(i) + ') ' + tweet.full_text + '\n')
  i=i+1

# Create a dataframe with a column called Tweets
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])


#Show the first 5 rows of data
df.head()

# Clean the text

#Function to clean the tweets
def cleanText(text):
  text = re.sub(r'@[A-Za-z0-9]+','', text) # r signifies that the eexpression is a raw string . this line removes @ mentions
  text = re.sub(r'#','', text) # remove the #tags symbol
  text = re.sub(r'RT[\s]','',text) #removing the retweets[RT]
  text = re.sub(r'https?:\/\/\S+','', text) #S+ whitespaces - remove the hyperlink

  return text

#cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanText)

#display the cleaned text
df

# Create a function to get the subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

#Create a function to get the polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# create two new columns
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

#Show the new dataframe and new columns
df

# Plot the Word Cloud

allWords = ' '.join( [twts for twts in df['Tweets']] )
wordCloud = WordCloud(width = 500, height=300, random_state =21, max_font_size = 110).generate(allWords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

# Create a function to compute the negative, neutral an positive analysis
def getAnalysis(score):
  if score<-0.2:
    return 'Negative'
  elif score >= 0.2:
    return 'Neutral'
  else:
    return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

#show the dataframe
df

# Print all of the positive tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0,sortedDF.shape[0]):
  if (sortedDF['Analysis'][i] == 'Positive'):
    print(str(j) + ') ' + sortedDF['Tweets'][i])
    print()
    j=j+1

# print the negative tweets
j=1
sortedDF = df.sort_values(by=['Polarity'], ascending='False')
for i in range(0, sortedDF.shape[0]):
  if(sortedDF['Analysis'][i] == 'Negative'):
    print(str(j) + ') ' + sortedDF['Tweets'][i])
    print()

    j=j+1

# Plot the polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Get the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

round((ptweets.shape[0] / df.shape[0]) * 100 , 1)

# Get the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']

round( (ntweets.shape[0] / df.shape[0] * 100), 1 )

# Get the percentage of neutral tweets
nutweets = df[df.Analysis == 'Neutral']
nutweets = nutweets['Tweets']

round( (nutweets.shape[0] / df.shape[0] * 100), 1 )

# Show the  value counts

df['Analysis'].value_counts()

#plot and visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()
