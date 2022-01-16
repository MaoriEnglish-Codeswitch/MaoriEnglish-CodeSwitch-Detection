import numpy as np
import pandas as pd
import random
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import heapq
import math
from sklearn import metrics
from simpletransformers.classification import ClassificationModel


dfnew = pd.read_csv("label_per_sentence_Hansard_forBERT.csv")

dfnew['text'] = dfnew['text'].astype(str)
dfnew['labels'] = dfnew['labels'].astype(int)
dfnew.columns = ['text','labels']



from sklearn.model_selection import train_test_split


train, test = train_test_split(dfnew, test_size=0.2, random_state=1, stratify=dfnew['labels'])

del dfnew

# Create a ClassificationModel
model = ClassificationModel(
	"bert", 
        "bert-base-uncased",  
         num_labels=3,
         args={"num_train_epochs": 1, "train_batch_size": 16, "overwrite_output_dir": True,  "learning_rate": 1e-5,
          "max_seq_length": 256, "save_eval_checkpoints": False, "save_model_every_epoch": False, "output_dir": "out1/"}
)



from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix



f1_score_micro = []
f1_score_macro = []


myfile = open('fscore_bert_1eminus5_256_35.txt', 'w')

for epoch in range(35):
    model.train_model(train)
    to_predict = test.text.tolist()
    # Evaluate the model
 #   result, model_outputs, wrong_predictions = model.eval_model(test)
    preds, outputs = model.predict(to_predict)
    y_true = test.labels
    f1_micro = metrics.f1_score(y_true, preds, average='micro')	
    f1_macro = metrics.f1_score(y_true, preds, average='macro')
    f1_score_micro.append(f1_micro)
    f1_score_macro.append(f1_macro)
    myfile.write("This is the output bert \n")
    myfile.write(classification_report(y_true, preds))
    myfile.write("\n")

myfile.write(f"F1 Score (Micro) = {f1_score_micro}\n")
myfile.write(f"F1 Score (Macro) = {f1_score_macro}\n")
myfile.write(accuracy_score(preds, y_true))


print(classification_report(y_true, preds))
print(f"F1 Score (Micro) = {f1_score_micro}\n")
print(f"F1 Score (Macro) = {f1_score_macro}\n")
print('accuracy %s', accuracy_score(preds, y_true))



myfile.close()


print(classification_report(y_true, preds))
print(f"F1 Score (Micro) = {f1_score_micro}\n")
print(f"F1 Score (Macro) = {f1_score_macro}\n")




