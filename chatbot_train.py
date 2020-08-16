# nltk(Natural Language Kit) contains a whole bunch of tools for cleaning up
#text and preparing it for deep learning algorithms;
# json -> loads json files directly into Python
# pickle -> loads pickle files/  The process to converts any kind of python
#objects(list, dict, etc.) into byte streams (0s and 1s) is called pickling
#or serialization or flattening or marshalling
# numpy -> can perform linear algebra operations very efficiently
# keras -> deep learning framework
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#initialize all of the lists where we will store our natural language data.
words = []
classes = []
documents = []
ignore_words = ['?','!']
#opens the file which contains greetings etc.
data_file = open('intents.json').read()
#copies the file into the intents variable.
intents = json.loads(data_file)

#extract all of the words within "patterns" and add them to our words list.
#then add to our documents list each pair of patterns within their corresponding tag.
#we also add the tags into our classes list
#and use a simple conditional statement to prevent repeats.

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        #extend -> extends list by appending elements from the iterable
        #append -> appends object at the end.
        words.extend(w)
        documents.append((w, intent['tag']))

        #adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Take the words list and lemmatize and lowercase all the words inside
# lemmatize means to turn a word into its base meaning, or its lemma.
#For example, the words "walking","walked","walks" all have the same lemma, "walk"
#The purpose of lemmatizing our words is to narrow everything down to the simplest level it can be.
#It is very similar to stemming, which is to reduce an inflected word down to its base or root form.

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#writes words in byte to store easily. We use 'wb' and 'rb' for writing/reading bytes.
pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#BUILDING THE DEEP LEARNING MODEL


#initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    #initializing bag of words
    bag = []
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    #lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #create out bag of words array with 1, means if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
#create train and test lists. X-Patterns, Y-intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# ?????????????we will now use a deep learning model from keras called Sequential.
#The Sequential model in keras is actually one of the simplest neural networks, a multi-layer perceptron

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile mode, Stochastic gradient descent with Nesterov accelertated gradient gives good results for this model.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


