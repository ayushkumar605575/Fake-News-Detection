from django.shortcuts import render

from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow import keras
from gensim.models import Doc2Vec
model = keras.models.load_model(r"D:\Django pro\Fakenewsdetection\models\model.h5")
doc=Doc2Vec.load(r"D:\Django pro\Fakenewsdetection\models\my_doc2vec_model")

def preprocess(sentence):
    punc = list(punctuation)
    punc.append('\'')
    punc.append('"')

    for i in punctuation:
        sentence = sentence.replace(i, '')
        
    #Tokenize sentence -> all words in a new list
    tok = sentence.split(' ')

    #Define text lemmatization model (eg: walks will be changed to walk)
    lemmatizer = WordNetLemmatizer()

    #Lemmatize each word in the sentence
    for w in sentence:
        lemmatizer.lemmatize(sentence)
    stop = stopwords.words('english')




    #Remove them from our dataframe and store in a new list
    minStop = []

    for i in tok:
        if i not in stop:
            minStop.append(i)

    #Doc2Vec tags
    # tag = [TaggedDocument(minStop,[0])]

    predVec = [doc.infer_vector(minStop)]
    predVec = np.array(predVec)
    prediction = model.predict(predVec)
    return round(prediction[0][0]*100,2)

def homepage(request):
    data = {"prediction" : "NULL (Please scan a News)",'info':"NULL (Please scan a News)","news":"NULL (Please scan a News)"}
    if request.method == 'POST':
        # data = DataFrame({"text":[str(request.POST.get('Search')).lower()]})
        news = str(request.POST.get('Search'))
        prediction=preprocess(str(request.POST.get('Search')))
        if int(prediction)>=75:
            info = "This News is very true."
        elif int(prediction)>=50:
            info = "This News is lightly true."
            # info = "TRUE"
        elif int(prediction)>=25:
            info = "This News is lightly false."
        else:
            info = "This News is fully false."
        data={"prediction" : str(prediction)+"%",'info':info,"news":news}
    return render(request,'home.html',data)