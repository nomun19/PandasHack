#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:34:51 2019
МУИС-ын Open Data платформоос өгөгдлийг авч ашиглав.
@author: sgo
"""

#%%
#from sklearn import preprocessing as preproc
import numpy
#%%
import pandas
import math
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from collections import Counter
#%%
hotolbor = pandas.read_csv("-2018-autumn-huvaari.csv")
uruu = pandas.read_csv("-2018-autumn-uruu.csv")
hotolbor.head(5)
uruu.head(5)
#%%
hotolbor=hotolbor.replace('07:40',1)
hotolbor=hotolbor.replace('08:25',1.5)
hotolbor=hotolbor.replace('09:10',2)
hotolbor=hotolbor.replace('09:20',2)
hotolbor=hotolbor.replace('10:05',2.5)
hotolbor=hotolbor.replace('10:50',3)
hotolbor=hotolbor.replace('11:00',3)
hotolbor=hotolbor.replace('11:45',3.5)
hotolbor=hotolbor.replace('12:30',4)
hotolbor=hotolbor.replace('12:40',4)
hotolbor=hotolbor.replace('13:25',4.5)
hotolbor=hotolbor.replace('14:10',5)
hotolbor=hotolbor.replace('14:20',5)
hotolbor=hotolbor.replace('15:05',5.5)
hotolbor=hotolbor.replace('15:50',6)
hotolbor=hotolbor.replace('16:00',6)
hotolbor=hotolbor.replace('16:45',6.5)
hotolbor=hotolbor.replace('17:30',7)
hotolbor=hotolbor.replace('17:40',7)
hotolbor=hotolbor.replace('18:25',7.5)
hotolbor=hotolbor.replace('19:10',8)
hotolbor=hotolbor.replace('19:20',8)
hotolbor=hotolbor.replace('20:05',8.5)
hotolbor=hotolbor.replace('20:50',9)
hotolbor=hotolbor.replace('21:00',9)
hotolbor=hotolbor.replace('21:45',9.5)
hotolbor=hotolbor.replace('22:30',10)
#%%
#%%
#hotolbor=hotolbor.replace('Даваа',1)
#hotolbor=hotolbor.replace('Мягмар',2)
#hotolbor=hotolbor.replace('Лхагва',3)
#hotolbor=hotolbor.replace('Пүрэв',4)
#hotolbor=hotolbor.replace('Баасан',5)
#hotolbor=hotolbor.replace('Бямба',6)
#hotolbor=hotolbor.replace('Ням',7)
#%%
#Merged result
result2 = pandas.merge(hotolbor, uruu, left_on=['Хичээллэх_байр', 'Өрөөний_дугаар'], right_on=['Хичээлийн_байр','Өрөөний_дугаар'], how='inner')
result2
#%%

temp = pandas.DataFrame({'Пар':(hotolbor['Дуусах_цаг'].values - hotolbor['Эхлэх_цаг'].values)})
hotolbor = pandas.concat([hotolbor, temp], axis=1)
hotolbor
#%%
hotolbor.groupby(['Эхлэх_цаг']).sum()
#hotolbor.values
#%%
#Hicheeliin tsag buriin achaalal
tsag = hotolbor[['Гараг', 'Эхлэх_цаг', 'Пар']]
tsag = tsag.groupby(['Гараг', 'Эхлэх_цаг']).sum()
tsag.unstack(level=0).plot(kind='bar', subplots=True, figsize=[15,12])
plt.savefig('hicheel.png')
plt.show()

#cutRepo.to_csv("tsag.csv",sep=',',encoding='utf8')
#%%
#Angiudiin achaalal
angi = hotolbor.groupby(['Гараг', 'Хичээллэх_байр', 'Өрөөний_дугаар']).sum()
angi
#%%
#Bagshiin huwaari
bagsh = hotolbor.groupby(['Багшийн_хувийн_дугаар','Заасан_багшийн_нэр']).sum()
bagsh
#bagsh.to_csv('bagshNiit.csv', sep=',', encoding = 'utf-8-sig')
#%%
# Garaguudaar hicheeliin toog gargah
garag_counts = Counter(hotolbor['Гараг'])
result3 = pandas.DataFrame.from_dict(garag_counts, orient='index')
result3.plot(kind='bar')
#%%
#Garagiin niit paariig gargasan ni
garag = hotolbor.groupby(['Гараг']).sum()
garag.plot(kind='bar', subplots=False, y='Пар')
garag
#%%
#garig = hotolbor.loc[hotolbor['Гараг'] == 1]
sumstart = numpy.sum(hotolbor['Эхлэх_цаг'].values)
sumend = numpy.sum(hotolbor['Дуусах_цаг'].values)
(sumend - sumstart)
#%%
bairNiitpar = hotolbor.groupby(['Хичээллэх_байр']).sum()
bairNiitpar
#%%
bair = hotolbor[['Хичээллэх_байр','Өрөөний_дугаар']].drop_duplicates().groupby(['Хичээллэх_байр']).count()
bair
#bair.plot(kind='bar', x='Хичээллэх_байр', y='Өрөөний_дугаар')
#%%
result = pandas.merge(hotolbor, uruu, left_on='Өрөөний_дугаар', right_on='Өрөөний_дугаар', how='left').drop('Өрөөний_дугаар', axis=1)
result.shape
#%%
temp6 = hotolbor[['Хичээллэх_байр', 'Өрөөний_дугаар', 'Гараг']].drop_duplicates()
paruud = pandas.DataFrame({'Парууд':[1,2,3,4,5,6,7,8,9], 'Боломжит':[1,1,1,1,1,1,1,1,1]})
paruud['tmp'] = 1
temp6['tmp'] = 1
temp7 = pandas.merge(temp6, paruud, on=['tmp'])
temp7 = temp7.drop('tmp', axis=1)
temp7
#%%
cnt = 0
for elm in hotolbor.values:
    cnt +=1
    print(cnt)
    ehleh_par = elm[10]
    duusah_par = elm[11]
    for i in range(math.floor(ehleh_par), math.ceil(duusah_par)):
        temp7.loc[(temp7['Хичээллэх_байр'] == elm[15]) & (temp7['Өрөөний_дугаар'] == elm[16])
        & (temp7['Гараг'] == elm[9]) & (temp7['Парууд'] == i), 'Боломжит'] = 0
temp7.head(100)
#%%
sulAngi = temp7.loc[temp7['Боломжит'] == 1]
sulAngi
#%%

#for i in range(1,10):
#    temp7.loc[(hotolbor['Эхлэх_цаг'] == i) & (hotolbor['Дуусах_цаг'] == (i + 1)), 'Боломжит'] = 0 
#%%
# Sul uruug haih function
def searchRoom(myTime, myDay, myBair):
    selection = sulAngi.loc[(sulAngi['Парууд'] == myTime) & (sulAngi['Гараг'] == myDay) & (sulAngi['Хичээллэх_байр'] == myBair)]
    print(selection)
    #return selection[['Өрөөний_дугаар']]
#%%
def searchRoom2(myTime, myDay):
    selection = sulAngi.loc[(sulAngi['Парууд'] == myTime) & (sulAngi['Гараг'] == myDay)]
    print(selection)
#%%
searchRoom(4,'Даваа', 'Хичээлийн байр 2')
#%%
"""
def suggestTime(people) :
    selection2 = sulAngi.copy()
    peopledf = pandas.DataFrame()
    for person in people:
        peopledf = peopledf.append(hotolbor.loc[hotolbor['Заасан_багшийн_нэр'] == person])
        print(peopledf)
        for person_indf in peopledf.values:
            print(person_indf)
            selection2 = selection2.loc[(selection2['Парууд'] != person_indf['Эхлэх_цаг']) | (selection2['Гараг'] != person_indf['Гараг'])]    
#        print(selection2)
#    print(selection2)    
#%%
suggestTime(['Г.Батаа', 'Б.Баяр', 'Г.Баярмаа', 'О.Батхуяг']) 
"""       