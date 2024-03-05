import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
#
import sklearn
from sklearn import datasets

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from numpy import argmax


#Ciscenje podataka i vizuelizacija!

df = pd.read_csv("/content/Cleaned-Data.csv")
df
df.info()

#sns.heatmap(df.isnull()) #Ova funkcija prikazuje toplotnu mapu koja vizuelno prikazuje prisustvo ili odsustvo nedostajućih vrednosti (NaN) u okviru DataFrame-a df.

df.columns
df.drop(['Contact_Dont-Know', 'Contact_No', 'Contact_Yes', 'Country', 'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'None_Experiencing', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Gender_Female', 'Gender_Male', 'Gender_Transgender', 'None_Sympton'], axis = 1, inplace = True) #Uklanjam kolone koje nisu potrebne za ovaj zadatak
df

df.hist() #Ova funkcija prikazuje histograme za sve numeričke kolone u DataFrame-u
plt.show()
#Deljenje dataseta na trenirajuci i testirajuci skup
train, test = train_test_split(df, test_size = 0.3, random_state = 1)
pred = test.copy()

X_train = train.iloc[:, :19].values
X_test = test.iloc[:, :19].values

Y_train = train.iloc[:, -1].values
Y_test = test.iloc[:, -1].values

#funkcija koja meri performanse
def perform(pred):
    print("Preciznost : ", precision_score(Y_test, pred)) #Preciznost se izračunava kao odnos broja tačno pozitivnih instanci (true positives) i sume broja tačno pozitivnih instanci i broja lažno pozitivnih instanci (false positives). Preciznost daje informaciju o tačnosti pozitivnih predikcija modela.
    print("Odziv : ", recall_score(Y_test, pred)) #Recall se izračunava kao odnos broja tačno pozitivnih instanci (true positives) i sume broja tačno pozitivnih instanci i broja lažno negativnih instanci (false negatives). Recall daje informaciju o sposobnosti modela da pronađe sve stvarne pozitivne instance.
    print("Tacnost : ", accuracy_score(Y_test, pred)) #Tačnost se izračunava kao odnos broja tačnih predikcija i ukupnog broja instanci. Tačnost pruža informaciju o ukupnoj sposobnosti modela da tačno predvidi klase.
    print("F1 Score : ", f1_score(Y_test, pred)) #kombinacija odziva i preciznosti
    print('')
    print(confusion_matrix(Y_test, pred), '\n')
    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, pred))
    cm.plot()
    plt.show()

brSvojstva = X_train.shape[1]
print(brSvojstva)

model = Sequential()
model.add(Dense(17, activation = 'relu', kernel_initializer = 'he_normal', input_shape = (brSvojstva,)))
model.add(Dense(14, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 10, batch_size = 64, verbose = 2)

loss, acc = model.evaluate(X_test, Y_test, verbose = 2)
print("Tacnost modela je: %.3f" % acc)

predikcija = np.argmax(model.predict(X_test), axis=1)

perform(predikcija)

print(classification_report(Y_test, predikcija)) # Ova funkcija štampa izveštaj klasifikacije koji sadrži preciznost, odziv, F1 meru i druge metrike za procenu performansi klasifikacionog modela. Ulazni argumenti su stvarne vrednosti ciljne promenljive Y_test i predikcije predikcija
pred['prediction'] = predikcija
pred
sns.displot(pred[['prediction', 'Severity_Severe']], kde=True) #Ova funkcija prikazuje disperziju (distribuciju) predikcija ozbiljnosti simptoma (prediction) u odnosu na ozbiljnost simptoma (Severity_Severe). Postavka kde=True dodaje procenjenu gustinu raspodele
plt.show()
sns.kdeplot(pred[['prediction', 'Severity_Severe']]) #Ova funkcija prikazuje procenjenu gustinu raspodele za predikcije ozbiljnosti simptoma (prediction) u odnosu na ozbiljnost simptoma (Severity_Severe)

print("Izvrsena je predikcija: %s (da li je ozbiljnost simptoma na virus Covid-19 velika ili ne (jeste = 1, nije = 0): %.3f)" % (predikcija, argmax(predikcija)))


plt.show() #bez ovoga df.hist() ne radi