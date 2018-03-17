# Combining algos with a Vote
import nltk
import random
#from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
        def __init__(self, *classifiers):
                self._classifiers = classifiers

        def classify(self, features):
                votes = []
                for c in self._classifiers:
                        v= c.classify(features)
                        votes.append(v)
                return mode(votes)

        def confidence(self, features):
                votes=[]
                for c in self._classifiers:
                        v= c.classify(features)
                        votes.append(v)
                choice_votes = votes.count(mode(votes))
                conf = choice_votes/len(votes)
                return conf

         
short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()


all_words = []
document = []

# J is adjective, r is adverb, abd v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
        document.append((p, "pos"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())

for p in short_neg.split('\n'):
        document.append((p, "neg"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())

save_document = open("pickled_algos/document.pickle", "wb")
pickle.dump(document, save_document)
save_document.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features



featuresets = [(find_features(rev), category) for (rev, category) in document]
#featuresets
save_word_features1 = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_word_features1)
save_word_features1.close()


random.shuffle(document)

print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

######## some other algos
#Process to save the classifier
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_Classifier = SklearnClassifier(MultinomialNB())
MNB_Classifier.train(training_set)
print("MNB_Classifier Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(MNB_Classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_Classifier, save_classifier)
save_classifier.close()


BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print("BernoulliNB_Classifier Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_Classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_Classifier, save_classifier)
save_classifier.close()


LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression_Classifier Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_Classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_Classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_Classifier, save_classifier)
save_classifier.close()


LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC_Classifier Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(LinearSVC_Classifier, testing_set))*100)


save_classifier = open("pickled_algos/LinearSVC_Classifier5k.pickle", "wb")
pickle.dump(LinearSVC_Classifier, save_classifier)
save_classifier.close()


SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print("SGDClassifier_Classifier Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_Classifier, testing_set))*100)


save_classifier = open("pickled_algos/SGDClassifier_Classifier5k.pickle", "wb")
pickle.dump(SGDClassifier_Classifier, save_classifier)
save_classifier.close()


voted_classifier = VoteClassifier(
                                        classifier,
                                        MNB_Classifier,
                                        BernoulliNB_Classifier,
                                        LogisticRegression_Classifier,
                                        SGDClassifier_Classifier,
                                        LinearSVC_Classifier)

print("voted_classifier Algo accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


def sentiment(text):
        feats = find_features(text)
        return voted_classifier.classify(feats)


