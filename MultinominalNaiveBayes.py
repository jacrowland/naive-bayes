import math
import numpy as np
import csv
import random
import copy

"""
Implementation of the Multinominal Naive Bayes document classifier with extentions

"""

class NaiveBayesClassifier():
    def __init__(self):
        self._likelihoods = None
        self._priors = None
        self._classes = [] # unique classes in the training set

    """
    fit()
    Fits the training data to the model to learn off
    """
    def fit(self, X:list, y:list, countWordClassDict:dict, countClassDict:dict, wordFrequencyDict:dict):
        p = Preprocessor()
        priors = p.calculatePriors(y)
        likelihoods = p.calculateLikelihoods(countWordClassDict, countClassDict, wordFrequencyDict)
        for label in countWordClassDict: self._classes.append(label) # set unique classes
        self._priors = priors
        self._likelihoods = likelihoods

    """
    classify()
    Returns the most probable class for a given document
    """
    def classify(self, document:list) -> str:
        predictions = []
        for y in self._classes:
            probability = math.log(self._priors[y]) # Logging probabilities
            for word in document:
                try:
                    probability = probability + (math.log(self._likelihoods[y][word]) * document.count(word)) # Logging probabilities
                except:
                    pass
            predictions.append((y, probability))
        predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True) # to get most likely class
        return predictions[0][0] # return the most probable class

class Preprocessor():
    def __init__(self):
        self.stopWords = self.importStopWords("stopWords.txt")

    """
    importStopWords()
    Reads list of common english words from a text files 
    """
    def importStopWords(self, path='stopwords.txt'):
        with open('stopwords.txt', encoding='utf8') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
        return lines

    """
    getCountWordClassDict()
    Returns a dictionary containing the unique word frequencies that appear in each class
    e.g. How many times the word x appears across all documents of class y
    """
    def getCountWordClassDict(self, trainingSetDict:dict, y:list) -> dict:
        countWordClassDict = {}
        for label in y:
            if label not in countWordClassDict:
                countWordClassDict[label] = {}
        for documentID in trainingSetDict.keys():
            label = list(trainingSetDict[documentID].keys())[0]
            documentWords = list(trainingSetDict[documentID][label].keys())
            for word in documentWords:
                if word not in countWordClassDict[label]:
                    countWordClassDict[label][word] = trainingSetDict[documentID][label][word]
                else:
                    countWordClassDict[label][word] += trainingSetDict[documentID][label][word]
        return countWordClassDict

    """
    getCountClassDict()
    Returns a dictionary that contains the total number of words that appear in documents of each class.
    e.g. The sum of all word counts across documents in class y is x
    """
    def getCountClassDict(self, countWordClassDict:dict):
        countClassDict = {}
        for y in countWordClassDict:
            totalWords = 0
            for word in countWordClassDict[y]:
                wordCount = countWordClassDict[y][word]
                totalWords += wordCount
            countClassDict[y] = totalWords
        return countClassDict

    """
    calculatePriors()
    Returns a dictionary of prior probabilities.
    E.g. What is the probability that a document of class y occurs out of all documents in the training set?
    """
    def calculatePriors(self, y:list)->dict:
        priors = {}
        for label in y:
            if label not in priors:
                priors[label] = 0
        for label in y:
            priors[label] += 1
        for label in priors:
            priors[label] = priors[label] / len(y)

        return priors

    """
    getWordFrequencyDict()
    Returns a dictionary that contains the total frequency of a word across all classes 
    """
    def getWordFrequencyDict(self, countWordClassDict:dict) -> dict:
        wordFrequencyDict = {}
        for y in countWordClassDict:
            for word in countWordClassDict[y]:
                if word not in wordFrequencyDict:
                    wordFrequencyDict[word] = countWordClassDict[y][word]
                else:
                    wordFrequencyDict[word] += countWordClassDict[y][word]
        return wordFrequencyDict

    """
    transformTermFrequency()
    This method implements word frequency transform as described in section 4.1 of Rennie at el. (2003)
    """
    def transformTermFrequency(self, trainingSetDict:dict) -> dict:
        for document in trainingSetDict:
            label = list(trainingSetDict[document].keys())[0]
            for word in trainingSetDict[document][label]:
                frequency = trainingSetDict[document][label][word]
                frequency = math.log(frequency + 1) # TF transform s 4.1 d_ij = log(d_ij + 1)
                trainingSetDict[document][label][word] = frequency
        return trainingSetDict

    """
    inverseDocumentFrequencyTransform()
    This method implements inverse document frequency transform as described in section 4.2 of Rennie at el. (2003)
    """
    def inverseDocumentFrequencyTransform(self, trainingSetDict:dict) -> dict:
        wordDocumentFrequencyDict = {}
        for documentID in trainingSetDict:
            label = list(trainingSetDict[documentID].keys())[0]
            for word in trainingSetDict[documentID][label].keys():
                if word not in wordDocumentFrequencyDict:
                    wordDocumentFrequencyDict[word] = 1
                else:
                    wordDocumentFrequencyDict[word] += 1
        numDocuments = len(trainingSetDict)
        for documentID in trainingSetDict:
            label = list(trainingSetDict[documentID].keys())[0]
            for word in trainingSetDict[documentID][label].keys():
                trainingSetDict[documentID][label][word] = trainingSetDict[documentID][label][word] * math.log(numDocuments / wordDocumentFrequencyDict[word])
        return trainingSetDict

    """
    transformLength()
    This method implements the normalisation extention as described in section 4.3 of Rennie at el. (2003)
    This updates each word frequency by dividing it by the sqrt of the sum of squares for each word
    """
    def transformLength(self, trainingSetDict:dict) -> dict:
        for documentID in trainingSetDict:
            label = list(trainingSetDict[documentID].keys())[0]
            # Calculate document length
            length = 0
            for word in trainingSetDict[documentID][label].keys():
                length += pow(trainingSetDict[documentID][label][word], 2)
            length = math.sqrt(length)
            # Normalise document
            for word in trainingSetDict[documentID][label].keys():
                trainingSetDict[documentID][label][word] = trainingSetDict[documentID][label][word] / length
        return trainingSetDict

    """
    calculateLikelihood()
    Creates a dictionary of conditional probabilities for each word given a class.
    e.g. The probability that word x appears in a document of class y -> P(x|y)
    """
    def calculateLikelihoods(self, countWordClassDict:dict, countClassDict:dict, wordFrequencyDict:dict):
        likelihoodDict = {}
        for word in wordFrequencyDict:
            for y in countWordClassDict:
                if y not in likelihoodDict:
                    likelihoodDict[y] = {}
                if word not in countWordClassDict[y]:
                    countWordClass = 0
                else:
                    countWordClass = countWordClassDict[y][word]
                likelihoodDict[y][word] = (countWordClass + 1) / (countClassDict[y] + len(wordFrequencyDict))
        return likelihoodDict
    
    """
    removeStopWords()
    Removes all common english words from the dictionary so that they are not used in 
    classification of an new document
    """
    def removeStopWords(self, wordFrequencyDict:dict):
        for stopWord in self.stopWords:
            if stopWord in wordFrequencyDict:
                wordFrequencyDict.pop(stopWord)
        return wordFrequencyDict

    """
    getTopXWords()
    Finds and returns a dictionary of the top most frequent words in the training data
    """
    def getTopXWords(self, wordFrequencyDict:dict, X:int=1000) -> dict:
        topWordsDict = {}
        while (len(topWordsDict) != X):
            key = max(wordFrequencyDict, key=lambda word: wordFrequencyDict[word])
            topWordsDict[key] = wordFrequencyDict[key]
            wordFrequencyDict.pop(key)
        return topWordsDict

    """
    getTrainingSetDict()
    Returns a dictionary of training examples 
    TrainingSetDic = {attributeID:{class:word:wordCount}}
    """
    def getTrainingSetDict(self, attributeIDs:list, X:list, y:list) -> dict:
        trainingSetDict = {}
        for i in range(len(attributeIDs)):
            document = X[i]
            label = y[i]
            documentId = attributeIDs[i]
            trainingSetDict[documentId] = {label: {}}
            for word in document:
                frequency = document.count(word)
                trainingSetDict[documentId][label][word] = frequency
        return trainingSetDict

def importData(path:str, labelsInSet:bool=True) -> tuple:
    attributeIDs = []
    labels = []
    features = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        reader.__next__() # skip the attribute names line
        for row in reader:
            if labelsInSet:
                ID = row[0]
                label = row[1].upper()
                words = row[2].lower().split()
                attributeIDs.append(ID)
                labels.append(label)
                features.append(words)
            else:
                ID = row[0]
                words = row[1].lower().split()
                attributeIDs.append(ID)
                features.append(words)
    if labelsInSet:
        return attributeIDs, labels, features
    else:
        return attributeIDs, features

"""
CrossValidation()
This class implements the Stratified Cross Validation where the distribution of each fold matches 
that class distribution of the training set.
Returns the mean accuracy, the standard deviation, and N - the number of tests (which is k)
"""
class CrossValidation():
    def __init__(self, attributeIDs:list, X:list, y:list, random_state=1234, k=10, topXWords=10000, standard=False, experiment=None):
        self.attributeIDs = attributeIDs
        self.X = X # full training set - abstracts e.g. features
        self.y = y # full training set - labels
        self.k = k # k folds
        self.random_state = random_state # used to initalise the random number generator
        self.updateRandomSeed()
        self.p = Preprocessor()
        self.topXWords = topXWords # top num words (features) to use in classification 
        self.standard = standard # Test using the Standard Naive Bayes Implementation (without extentions)
        self.experiment = experiment # run each experiment by specifying its number - No experiment by default
        self.folds = self.generateStratifiedFolds()

    def generateStratifiedFolds(self):
        folds = []
        foldDistribution = {} 
        labelIndicesDict = {}
        # Calculates the number of examples to pick out for each class to match the training set distribution
        priors = self.p.calculatePriors(self.y)
        foldSize = len(self.X) / self.k

        for label in priors:
            foldDistribution[label] = round(foldSize * priors[label])

        for label in priors:
            # Get all the indexes for documents of each type
            labelIndices = [i for i, y in enumerate(self.y) if y == label]
            labelIndicesDict[label] = labelIndices
            random.shuffle(labelIndicesDict[label])
        
        # generate k folds
        for k in range(self.k):
            X_fold = []
            y_fold = []
            attributeIDs_fold = []
            for label in foldDistribution:
                numToSample = foldDistribution[label]
                numSampled = 0
                # pop number of elements that matches training set distribution
                while numSampled != numToSample and len(labelIndicesDict[label]) != 0:
                    index = labelIndicesDict[label].pop()
                    X_fold.append(self.X[index])
                    y_fold.append(self.y[index])
                    attributeIDs_fold.append(self.attributeIDs[index])
                    numSampled += 1

            fold = (X_fold, y_fold, attributeIDs_fold)
            folds.append(fold)

        return folds
                
    def run(self):
        accuracies = []
        # for each fold
        for i in range(len(self.folds)):
            self.random_state = self.random_state + i
            self.updateRandomSeed()
            print(str(round(i / len(self.folds) * 100, 2)) + "%", end="")
            print("\r", end="")
            validationFold = self.folds[i]
            X_validation = validationFold[0]
            y_validation = validationFold[1]
            # concatenate all other arrays
            X_train = np.array([])
            y_train = np.array([])
            attributeID_train = np.array([])

            # combine all other folds into one test set
            for j in range(len(self.folds)):
                if j == i:
                    pass
                X = np.array(self.folds[j][0], dtype=object)
                y = np.array(self.folds[j][1], dtype=object)
                IDs = np.array(self.folds[j][2], dtype=object)
                X_train = np.concatenate((X_train, X))
                y_train = np.concatenate((y_train, y))
                attributeID_train = np.concatenate((attributeID_train, IDs))

            trainingSetDict = self.p.getTrainingSetDict(attributeID_train, X_train, y_train)

            # Preprocessing of training set

            # Extended naive bayes transformations
            if not self.standard:
                trainingSetDict = self.p.transformTermFrequency(trainingSetDict)
                trainingSetDict = self.p.inverseDocumentFrequencyTransform(trainingSetDict) 
            countWordClassDict = self.p.getCountWordClassDict(trainingSetDict, y_train)
            countClassDict = self.p.getCountClassDict(countWordClassDict)
            wordFrequencyDict = self.p.getWordFrequencyDict(countWordClassDict)
            # Extended naive bayes transformations cont.
            if not self.standard:
                wordFrequencyDict = self.p.removeStopWords(wordFrequencyDict)
                #wordFrequencyDict = self.p.getTopXWords(wordFrequencyDict, self.topXWords)

            # Fitting model and classifying
            nb = NaiveBayesClassifier()
            nb.fit(X_train, y_train, countWordClassDict, countClassDict, wordFrequencyDict)

            correctClassifications = 0
            incorrectClassifications = 0
            for i in range(len(X_validation)):
                prediction = nb.classify(X_validation[i])
                if prediction == y_validation[i]:
                    correctClassifications += 1
                else:
                    incorrectClassifications += 1

            accuracy = correctClassifications / (correctClassifications + incorrectClassifications)
            accuracies.append(accuracy)

        return np.mean(np.array(accuracies)), np.std(np.array(accuracies)), self.k
    
    def updateRandomSeed(self):
        random.seed(self.random_state)  

def main():
    p = Preprocessor()
    skipStandard = False # skip CV of the standard naive bayes
    skipCV = False # skip cross validation altogether
    skipTest = False # skip classification of tst.csv data

    print("Importing data")
    attributeIDs, y, X = importData('trg.csv')
    testAttributeIDs, X_test = importData('tst.csv', labelsInSet=False) # y_test will be empty

    if not skipCV:
        print("Cross validation...")
        if not skipStandard:
            # Standard Naive Bayes
            # This is run by passing in a standard=True paramater to ignore any additional transformations to the dataset
            print("Performing CV on Standard Naive Bayes...")
            cv = CrossValidation(attributeIDs, X, y, random_state=1234, k=10, topXWords=1000, standard=True)
            print(cv.run(), end="\n")
        # Extended Naive Bayes
        # This includes the additional transformation extentions to the Multinominal Naive Bayes classifier
        print("Performing CV on Extended Naive Bayes...")
        cv = CrossValidation(attributeIDs, X, y, random_state=1234, k=10, topXWords=1000, standard=False)
        print(cv.run(), end="\n")
    
    # Classify tst.csv
    if not skipTest:
        print("Preprocessing training data...")
        trainingSetDict = p.getTrainingSetDict(attributeIDs, X, y)
        trainingSetDict = p.inverseDocumentFrequencyTransform(trainingSetDict)  # IDF Extention
        countWordClassDict = p.getCountWordClassDict(trainingSetDict, y)
        countClassDict = p.getCountClassDict(countWordClassDict)
        wordFrequencyDict = p.getWordFrequencyDict(countWordClassDict)
        wordFrequencyDict = p.removeStopWords(wordFrequencyDict) # SW Extention
        
        print("Classifying test set & outputting to CSV...")
        nb = NaiveBayesClassifier()
        nb.fit(X, y, countWordClassDict, countClassDict, wordFrequencyDict)
        with open('output.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['ID', 'class'])
                for i in range(len(X_test)):
                    print(str(round(i / len(X_test) * 100, 2)) + "%", end="")
                    print("\r", end="")
                    prediction = nb.classify(X_test[i])
                    ID = testAttributeIDs[i]
                    writer.writerow([ID, prediction])
        print("Complete.")
        
main()