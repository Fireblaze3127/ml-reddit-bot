# Reddit libaries
import praw

# Machine learning libaries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initalizing Reddit object which we will use to interact with the site
reddit = praw.Reddit(
    client_id = 'user client id',  
    client_secret = 'user client secret',
    user_agent = 'reddit_script by /u/fireblaze3127' # Name your user agent whatever you like
)

# If this prints, then you have successfully interacted with Reddit
if reddit.read_only:
    print("Log: Contact with reddit has been Established")
else: 
    print("Log: Contact with reddit has FAILED")

# Gets the top 10 submissions from a subreddit
uw_submissions = list(reddit.subreddit('uwaterloo').hot(limit=10))
print("Log: You have obtained %d submissions from /r/uwaterloo" % len(uw_submissions))

# Get the comments from the top submission
top_uw_submission = uw_submissions[4]
uw_comments = top_uw_submission.comments.list()
print("Log: Top /r/uwaterloo submission is '%s'" % top_uw_submission.title)
print("Log: You have obtained %a comments from /r/uwaterloo" % len(uw_comments))

# Ges the comments from /r/smashbros
smashbros_submissions = list(reddit.subreddit('smashbros').hot(limit=10))
top_smashbros_submissions = smashbros_submissions[7]
smashbros_comments = top_smashbros_submissions.comments.list()

# Gather the comments and tag them as uw or smashbros
#each comment is an object, so use function comment.body
corpus = [comment.body for comment in uw_comments[:5] + smashbros_comments[:5]]
y_train = [0] * len(uw_comments[:5]) + [1] * len(smashbros_comments[:5])

# TODO: There is a problem when using 100+ comments as some comments are stored in a MoreComments object
# Put all the comments in a corpus
# Tag the comments:
#   0 = /r/uw
#   1 = /r/smashbros
print(y_train)
print(uw_comments[:5])
print(smashbros_comments[:5])

# Vectorize the corpus
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
x_train=vectorizer.transform(corpus)

# Train the Naive Bayes machine learning model
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Get testing data
test_uw_comments = uw_submissions[3].comments.list()[:5]
test_smashbros_comments = smashbros_submissions[3].comments.list()[:5]

# Put testing data in a corpus and tag them
test_corpus = [comment.body for comment in [test_smashbros_comments[0] , test_uw_comments[0]]]
y_test = [0,1]

# Vectorize testing data
x_test = vectorizer.transform(test_corpus)

# Check how the model preformed
print(classifier.predict(x_test))
print(y_test)