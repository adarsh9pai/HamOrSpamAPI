import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
MNB = MultinomialNB()
CV = CountVectorizer()


sms = pd.read_csv('spam.csv', encoding = 'latin-1')
sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
sms.rename(columns = {'v1' : 'label', 'v2' : 'message'}, inplace = True)
sms['labelValue'] = sms.label.map({'ham':0,'spam':1})

x = sms.message
y = sms.labelValue 
xtr, xte, ytr, yte = train_test_split(x, y, random_state = 1)
xtrdt = CV.fit_transform(xtr)
xtedt = CV.transform(xte)

MNB.fit(xtrdt.toarray(),ytr)
ypred = MNB.predict(xtedt.toarray())

print "SMS spam detection accuracy using the multinomial Naive Bayes classifier is: %.2f%%"%(metrics.accuracy_score(yte,ypred)*100)