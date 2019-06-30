# By: Nawaf Al-Dhelaan
# Last changed: June 22nd, 2019
# 
# Description:
#	This script interfaces with the twitter API, the emotion classifier, and runs the visualizations

# =========================== Dependencies & Function Definitions ===========================

import sys, time, re, matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from numpy import mean
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from classifier import EmotionClassifier

# read API credentials from file
def readCredentials(filename):
	credentials_file = open(filename)
	contents = credentials_file.readlines()
	credentials = {}
	for line in contents:
		parts = line.split(":")
		credentials[parts[0]] = parts[1].strip()
	credentials_file.close()
	return credentials

# Traverse twitter object JSON to find full text of tweet, cleans tweet
def parseTweet(tweet):
	try:
		# If retweet, fetch original tweet. If truncated, fetch full text
		if "retweeted_status" in response.keys():
			if response["retweeted_status"]["truncated"]:
				text = response["retweeted_status"]["extended_tweet"]["full_text"]
			else:
				text = response["retweeted_status"]["text"]
		elif response["truncated"]:
			text = response["extended_tweet"]["full_text"]
		else:
			text = response["text"]
	except KeyError:
		return None

	# Mask URLs, @usernames, and #hashtags
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
	text = re.sub('@[^\s]+', '', text)
	text = re.sub(r'#([^\s]+)', r'\1', text)

	return text

# =========================== AUTHENTICATION & SETUP ===========================

# Variables that contains the user credentials to access Twitter API 
credentials = readCredentials("assets/twitter_api.txt")
ACCESS_TOKEN = credentials['ACCESS_TOKEN']
ACCESS_SECRET = credentials['ACCESS_SECRET']
CONSUMER_KEY = credentials['CONSUMER_KEY']
CONSUMER_SECRET = credentials['CONSUMER_SECRET']

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)
stream = twitter_stream.statuses.filter(track=sys.argv[1], language="en", tweet_mode="extended")

# classifier setup
tone_analyzer = EmotionClassifier()

# Setup visualization
fig = plt.figure()
win = fig.canvas.manager.window
tones = {"Anger": [], "Fear": [], "Joy": [], "Sadness": [], "Surprise": [], "Disgust": []}
N = len(tones.keys())
rects = plt.bar(range(N), [1.0]*N, align='center')
plt.xticks(range(len(tones.keys())), tones.keys())
plt.xlabel("Tone"), plt.ylabel("Score")
for rect in rects:
	rect.set_height(0)
	
# =========================== MAIN BODY ===========================

# Fetch realtime tweets from API
#first_tweet = True
for response in stream:
	tweet = parseTweet(response)

	# Classify (call functions to classifier)
	sentiments = tone_analyzer.predict_probabilities([tweet])

	# Adjust the visualization
	print(sentiments)
	for i in range(1, len(sentiments.columns)):
		tones[sentiments.columns[i]].append(sentiments[sentiments.columns[i]])

	for rect, h in zip(rects, [mean(x) for x in tones.values()]):
		rect.set_height(h)
	fig.canvas.draw()
	plt.pause(1e-17)
	time.sleep(0.1)

#	first_tweet = False
plt.show()
