#Analysis packages
import pandas as pd
import sklearn as sk
import numpy as np
import scipy.stats as sp

#Visualization packages
import matplotlib.pyplot as plt
import matplotlib as matplot
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#Models & Sklearn packages
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, make_scorer, fbeta_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score as cvs

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from tqdm import tqdm
import datetime

import warnings
warnings.filterwarnings('ignore')


class model():
    def __init__(self, A, AB, C, CD):
        self.A = A
        self.AB = AB
        self.C = C
        self.CD = CD
    
    
    def model_run(self, model_list, title):
        self.model_list = model_list
        self.title = title
        self.model_preds = []
        self.model_names = []
        self.model_acc = []
      
        for model_name, model in (self.model_list):
            model_n = model
            model_n.fit(self.A, self.C)
            model_pred = model_n.predict(self.A)
            model_pred1 = model_n.predict(self.AB)
            model_cm = confusion_matrix(self.C, model_pred)
            model_cm1 = confusion_matrix(self.CD, model_pred1)
            #model_cr = classification_report(model_pred, self.CD)
            model_f1 = f1_score(model_pred1, self.CD)
            model_acc = accuracy_score(model_pred, self.C)
            model_acc1 = accuracy_score(model_pred1, self.CD)
            self.model_preds.append(model_f1)
            self.model_acc.append(model_acc)
            self.model_names.append(model_name)
            print("Training Data Confusion Matrix:")
            print(model_cm)
            print("Testing Data Confusion Matrix:")
            print(model_cm1)
            output = "%s: Training Data Accuracy- %f, Validation Data Accuracy- %f" % (model_name, model_acc, model_acc1)
            print(output)
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12, 5)
        fig.suptitle(self.title + ' Models Scores comparison', size=20)
        sns.barplot(self.model_names, self.model_preds, ax=ax[0])
        sns.barplot(self.model_names, self.model_acc, ax=ax[1])
        ax[0].set_xlabel('models_eval', fontsize = 15)
        ax[0].set_ylabel('F1_scores', fontsize = 15)
        ax[1].set_xlabel('models_eval', fontsize = 15)
        ax[1].set_ylabel('Accuracy', fontsize = 15)
        ax[0].tick_params(axis = 'x', labelsize = 15)
        ax[1].tick_params(axis = 'x', labelsize = 15)
        ax[0].tick_params(axis = 'y', labelsize = 15)
        ax[1].tick_params(axis = 'y', labelsize = 15)

    def models_scores_plot(self, scores, names, title):
        self.title = title
        self.scores = scores
        self.names = names
        plt.figure(figsize=(15,7))
        plt.title(self.title + ' Models Scores comparison', size=20)
        plt.ylabel('MAE_scores', fontsize=15, fontweight='bold')
        plt.xlabel('models_eval', fontsize=15, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        sns.barplot(self.names, self.scores)
        plt.show()
    
    def cross_fold_val(self, model_list):
        self.model_list = model_list
        self.avg_scores = []
        self.std_dev = []
        self.model_names = []
        for model_name, model in tqdm(self.model_list):
            score = cvs(model, self.A, self.C, cv=5, scoring='accuracy')
            scores = abs(score) # MAE scoring is negative in cross_val_score
            avg_score = np.mean(scores)
            std = np.std(scores)
            self.avg_scores.append(avg_score)
            self.std_dev.append(std) 
            self.model_names.append(model_name)
            output = "%s: %f (%f)" % (model_name, avg_score, std)
            print(output)
        fig, ax = plt.subplots(figsize=(15,7))
        plt.title(' Models with Cross Validation Scores comparison', size=20)
        plt.ylabel('Avg_scores', fontsize=15, fontweight='bold')
        plt.xlabel('model_list', fontsize=15, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax.bar(self.model_names, self.avg_scores)
        plt.show()
        
    def bestmodel_eval(self, model):
        model.fit(self.A, self.C)
        preds = model.predict(self.A)
        # Confusion matrix
        confmat = confusion_matrix(self.C, preds)
        print("The Confusion matrix:\n", confmat)
        print("Precision Score:", round(precision_score(self.C, preds), 2))
        print("Recall Score:", round(recall_score(self.C, preds), 2))                   
                       
    def summary_metrics(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        confmat = confusion_matrix(self.y, self.y_pred)
        TP = confmat[0,0]
        FN = confmat[0,1]
        FP = confmat[1,0]
        TN = confmat[1,1]
 
        # Print the confusion matrix
        print("Confusion matrix:\n", confmat)
        print()
    
        # Print the Accuracy Score
        print("Accuracy:", round(accuracy_score(self.y, self.y_pred),2))

        # Print the Sensitivity/recall/true positive rate
        print("Sensitivity:", round(recall_score(self.y, self.y_pred),2))

        # Precision/positive predictive value
        print("Precision:", round(precision_score(self.y, self.y_pred),2))
    
        print("F1-Score:", round(f1_score(self.y, self.y_pred),2))
    
        print("AUC:", round(roc_auc_score(self.y, self.y_pred), 2))

    
    def feat_importance(self, model):
        self.model = model
        if hasattr(bestModel, 'feature_importances_'):
            importances = bestModel.feature_importances_

        else:
            # for linear models which don't have feature_importances_
            importances = [0]*len(train_df_X.columns)

        feature_importances = pd.DataFrame({'feature':train_df_X.columns, 'importance':importances})
        feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    
        # set index to 'feature'
        feature_importances.set_index('feature', inplace=True, drop=True)
    
        # create plot
        feature_importances[0:25].plot.bar(figsize=(20,10))
        plt.show()
    
    



