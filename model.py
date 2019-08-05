import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import pandas as pd


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


def tokenise(text):
    filter = "!\"£$%^&*()_+=[]{}:;@',<.>/?`¬¦|\\"
    for i in filter:
        text = text.replace(i, "")
    return text.split(" ")


df = pd.read_csv("messages.csv")
df["pos"].dropna(inplace=True)
df["neg"].dropna(inplace=True)

tokenise_positive = df["pos"].apply(tokenise)
tokenise_negative = df["neg"].apply(tokenise)
tokenise_positive = tokenise_positive.tolist()
tokenise_negative = tokenise_negative.tolist()

features_positive = [
    (extract_features(tokenise_positive[f]), "Positive")
    for f in range(len(tokenise_positive))
]
features_negative = [
    (extract_features(tokenise_negative[f]), "Negative")
    for f in range(len(tokenise_negative))
]

# Split the data into train and test (80/20)
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

features_train = (
    features_positive[:threshold_positive] + features_negative[:threshold_negative]
)
features_test = (
    features_positive[threshold_positive:] + features_negative[threshold_negative:]
)
print("\nNumber of training datapoints:", len(features_train))
print("Number of test datapoints:", len(features_test))

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print(
    "\nAccuracy of the classifier:",
    nltk.classify.util.accuracy(classifier, features_test),
)

print("\nTop 10 most informative words:")
for item in classifier.most_informative_features()[:10]:
    print(item[0])

# Sample input
input_text = [
    "aaaaaaaa",
    "Hi, I'm Bert, I'm from California and I like to program in HTML",
]

print("\nPredictions:")
for intro in input_text:
    print("\nIntro:", intro)
    probdist = classifier.prob_classify(extract_features(intro.split()))
    pred_sentiment = probdist.max()
    print("Predicted sentiment:", pred_sentiment)
    print("Probability:", round(probdist.prob(pred_sentiment), 2))

import pickle

f = open("trained_classifier.pickle", "wb")
pickle.dump(classifier, f)
f.close()
