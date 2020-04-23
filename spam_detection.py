from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CORPUS = ["I love cats", "buy this"] ## Put email messages here
y = ["not-spam", "spam"]      ## Tag emails as spam/not-spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(CORPUS)

classifier = MultinomialNB() 
classifier.fit(X, y)

#Creates test data
z = ["I love people"]
test_z = vectorizer.transform(z)

print(classifier.predict(test_z)) ## Use the machine learning model here!
