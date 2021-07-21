#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import resource 
import sys
import getopt
import os
import datetime
import csv
import pickle
import numpy as np
import pandas as pd
import random
import xgboost as xgb
import matplotlib.pyplot as plt
import itertools
from numpy import interp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import shap


# Ensures given files satisfy one of the possible pathways provided by cellpy
# Ensures user input for train or predict matches file inputs
# Certain files must appear together for training and/or validation to proceed
def check_combinations(user_train, user_predict, train_normexpr, labelinfo, train_metadata, testsplit, featrank_on, frsplit,
                       rejection_cutoff, val_normexpr, val_metadata, layer_paths):
    cellpy_train = False
    cellpy_predict = False
    passed = True
    if user_train is None:
        print('ERROR: Run mode must be provided to resume')
    train_list = [train_normexpr, labelinfo, train_metadata, testsplit, featrank_on, frsplit]
    # if the user selected train or provided train variables
    if (user_train is True) or (train_list.count(None) != len(train_list)):
        print('Training option selected')
        train_list.append(rejection_cutoff)
        if train_list.count(None) != len(train_list):
            cellpy_train = True
        if user_train is not cellpy_train:
            print('ERROR: User selection for training option contradicts inputted files')
            passed = False
        if train_list[0] is None:
            print('ERROR: Normalized expression matrix for training must be provided to resume')
            passed = False
        if train_list[1] is None:
            print('ERROR: Label information file must be provided to resume')
            passed = False
        if train_list[2] is None:
            print('ERROR: Metadata file must be provided to resume')
            passed = False
        if train_list[3] is None:
            print('ERROR: Test split amount must be provided to resume')
            passed = False
        if train_list[4] is None:
            print('ERROR: Feature ranking on/off must be provided to resume')
            passed = False
        if train_list[5] is not None and train_list[4] is False:
            print('ERROR: Feature ranking split provided but feature ranking off')
            passed = False
        if train_list[5] is None and train_list[4] is True:
            print('WARNING: Feature ranking split not provided but feature ranking on, will be automatically set to 0.3')
        if train_list[6] is None:
            print('ERROR: Rejection cutoff value must be provided to resume')
            passed = False
    
    predict_list = [val_normexpr, val_metadata, layer_paths]
    # if the user selected the 'predict' option or provided predict variables
    if (user_predict is True) or (predict_list.count(None) != len(predict_list)):
        if val_metadata is not None:
            print('Prediction option with accuracy calculation selected')
        else:
            print('Prediction option without accuracy calculation selected')
        predict_list.append(rejection_cutoff)
        if predict_list.count(None) != len(predict_list):
            cellpy_predict = True
        if user_predict is not cellpy_predict:
            print('ERROR: User selection for prediction option contradicts inputted files')
            passed = False
        if predict_list[0] is None:
            print('ERROR: Normalized expression matrix for prediction must be provided to resume')
            passed = False
        if predict_list[2] is None and len(train_list)<6: # if no objects provided and training option is not on
            print('ERROR: Path names to Layer objects must be provided to resume')
            passed = False
        if predict_list[3] is None:
            print('ERROR: Rejection cutoff value must be provided to resume')
            passed = False
    return passed



# Ensures all the user given variables for training exist or are in bounds
def check_trainingfiles(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff, frsplit):
    passed = True
    if not os.path.exists(train_normexpr):
        print('ERROR: Given normalized expression data file does not exist')
        passed = False
    if not os.path.exists(labelinfo):
        print('ERROR: Given label info file does not exist')
        passed = False
    if not os.path.exists(train_metadata):
        print('ERROR: Given metadata file does not exist')
        passed = False
    if testsplit > 1 or testsplit < 0:
        print('ERROR: Given test split percentage must be a value between 0 and 1')
        passed = False
    if rejection_cutoff > 1 or rejection_cutoff < 0:
        print('ERROR: Given rejection cutoff must be a value between 0 and 1')
        passed = False
    if frsplit is not None and (frsplit > 1 or frsplit < 0):
        print('ERROR: Given feature ranking split must be a value between 0 and 1')
        passed = False
    return passed


# Conducts training in all layers separated into different folders by name
# Creates directory 'training' in cellpy_results folder, defines 'Root' as topmost layer
# Conducts finetuning on Root layer with 50 iterations
def training(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff, featrank_on, frsplit):
    global path
    path = os.path.join(path, 'training')
    os.mkdir(path)
    os.chdir(path)
    csv2pkl(train_normexpr)
    train_normexpr = train_normexpr[:-3] + 'pkl'
    all_layers = [Layer('Root', 0)]
    construct_tree(labelinfo, all_layers)
    print(all_layers)
    for layer in all_layers:
        path = os.path.join(path, layer.name.replace(' ', ''))
        os.mkdir(path)
        os.chdir(path)
        path = path + '/'
        if layer.name == 'Root': # root top layer
            parameters = layer.finetune(2, testsplit, train_normexpr, train_metadata)
            print(parameters)
        layer.train_layer(train_normexpr, train_metadata, parameters, testsplit, [0, rejection_cutoff], featrank_on, frsplit)
        os.chdir('..') # return to training directory
        path = os.getcwd()
    path = path + '/'
    training_summary(all_layers)
    export_layers(all_layers)
    print('Training Complete')
    os.chdir('..') # return to cellpy directory
    path = os.getcwd()
    return all_layers


# Converts the normalized expression csv into a pkl
# Expression CSV file must contain genes as row names, samples as column names
# First column name (cell A1) is 'gene'
def csv2pkl(csvpath):
    tp = pd.read_csv(csvpath, iterator=True, chunksize=1000)
    norm_express = pd.concat(tp, ignore_index=True)
    print (norm_express.head())
    norm_express.set_index('gene', inplace=True)
    norm_express.index.names = [None]
    norm_express = norm_express.T
    norm_express.to_pickle(csvpath[:-3] + 'pkl')


# Constructs a list of all Layer objects from a labelinfo file
# Initalizes each Layer object with a name, level #, and label dictionary
def construct_tree(labelinfo, all_layers):
    labeldata = fill_data(labelinfo)
    fill_dict(labeldata, all_layers)


# Function of construct_tree, fills the vertical columns of the labeldata file
def fill_data(labelinfo):
    labeldata = list(csv.reader(open(labelinfo, encoding='utf-8-sig')))
    for i in range(1,len(labeldata)):
        if labeldata[i][0]=='':
            labeldata[i][0] = labeldata[i-1][0]
    for i in range(1,len(labeldata)):
        for j in range(1,len(labeldata[0])):
            if labeldata[i][j]=='' and labeldata[i][j-1]==labeldata[i-1][j-1]:
                labeldata[i][j] = labeldata[i-1][j]
    return labeldata


# Function of construct_tree after fill_data, reads the edited / filled labeldata file
# Constructs the main list structure and initializes all Layer objects in it
def fill_dict(labeldata, all_layers):
    for i in range(len(labeldata)):
        # Fills the root dictionary
        if find_layer(all_layers, labeldata[i][0]) is None:
            root_layer = find_layer(all_layers, 'Root')
            root_layer.add_dictentry(labeldata[i][0])
        # Fills dictionaries in other layers given the existence of a leaf
        for j in range(1,len(labeldata[0])):
            if labeldata[i][j]!='':
                if find_layer(all_layers, labeldata[i][j-1]) is None:
                    all_layers.append(Layer(labeldata[i][j-1], j))
                prev_layer = find_layer(all_layers, labeldata[i][j-1])
                prev_layer.add_dictentry(labeldata[i][j])
            else:
                break


# Utility function of fill_dict, searches all_layers for a layer with the given name
def find_layer(all_layers, name):
    for layer in all_layers:
        if layer.name == name:
            return layer
    return None


# Summarizes and lists the path names of all the files created during training
# Prints the summary of each Layer object
def training_summary(all_layers):
    f = open(path + 'training_summaryfile.txt', 'w')
    for layer in all_layers:
        f.write(str(layer))
        f.write('\n')


# Exports all the trained Layers as pickle files
def export_layers(all_layers):
    for layer in all_layers:
        with open(path + layer.name + '_object.pkl', 'wb') as output:
            pickle.dump(layer, output, pickle.HIGHEST_PROTOCOL)



# Ensures all the user given variables for validation exist / are in the correct format
def check_predictionfiles(val_normexpr, val_metadata=None, layer_paths=None):
    passed = True
    if not os.path.exists(val_normexpr):
        print('ERROR: Given validation normalized expression data file does not exist')
        passed = False
    if val_metadata != None and not os.path.exists(val_metadata):
        print('ERROR: Given validation metadata file does not exist')
        passed = False
    # check all layer paths are objects and contain a trained xgb model
    if layer_paths != None:
        for i in range(len(layer_paths)):
            layer_path = layer_paths[i]
            if not os.path.exists(layer_path):
                print('ERROR: Given Layer object ' + str(i) + ' does not exist')
                passed = False
            else:
                layer = pd.read_pickle(layer_path)
                if layer.trained() is False:
                    print('ERROR: Given Layer object ' + str(i) + ' is not trained')
                    passed = False
    return passed


# Conducts prediction in all layers separated into different folders by name
# Creates directory 'prediction' in cellpy_results folder, defines 'Root' as topmost layer
def prediction(val_normexpr, val_metadata, all_layers=None, object_paths=None):
    if object_paths != None:
        all_layers = import_layers(object_paths)
    global path
    path = os.path.join(path, 'prediction')
    os.mkdir(path)
    os.chdir(path)
    featurenames = all_layers[0].xgbmodel.feature_names
    reorder_pickle(val_normexpr, featurenames)
    val_normexpr = val_normexpr[:-3] + 'pkl'
    for layer in all_layers:
        path = os.path.join(path, layer.name.replace(' ', ''))
        os.mkdir(path)
        os.chdir(path)
        path = path + '/'
        layer.predict_layer([0, rejection_cutoff], val_normexpr, val_metadata)
        os.chdir('..') # return to prediction directory
        path = os.getcwd()
    print('Validation Complete')
    os.chdir('..') # return to cellpy directory
    path = os.getcwd()


# Imports Layer objects from a list of given paths
def import_layers(layer_paths):
    layers = []
    for layer_path in layer_paths:
         layer = pd.read_pickle(layer_path)
         layers.append(layer)
    return layers


# Converts the normalized expression csv into a pkl
# Expression CSV file must contain genes as row names, samples as column names
# First column name (cell A1) is 'gene'
# Reorders the csv file to match the features in a given featurenames list
# Returns path to the new pkl file
def reorder_pickle(csvpath, featurenames):
    tp = pd.read_csv(csvpath, iterator=True, chunksize=1000)
    norm_express = pd.concat(tp, ignore_index=True)
    print (norm_express.head())
    norm_express.set_index('gene', inplace=True)
    norm_express.index.names = [None]
    norm_express = norm_express.T
    print(norm_express.T.duplicated().any())
    print ('Training Data # of  genes: ' + str(len(featurenames)))
    # Manually reorder columns according to atlas index
    origfeat = list(norm_express)
    print ('Validation Data # of genes: ' + str(len(origfeat)))
    newindex = []
    for i in range(len(featurenames)):
        if featurenames[i] in origfeat:
            newindex.append(featurenames[i])
    print ('Overlapping # of genes: ' + str(len(newindex)))
    norm_express = norm_express.reindex(columns=newindex)
    # Add missing features, remove extra features to match atlas
    i = 0
    missing_counter = 0
    while i < len(list(norm_express)):
        if list(norm_express)[i] != featurenames[i]:
            # this block does not run, reindexing removes new genes already
            if list(norm_express)[i] not in featurenames:
                print ('Deleted gene: ' + list(norm_express)[i])
                del norm_express[list(norm_express)[i]]
                i -= 1
            else:
                norm_express.insert(i, featurenames[i], None)
                missing_counter += 1
        i += 1
    while i < len(featurenames):
        norm_express.insert(i, featurenames[i], None)
        i += 1
        missing_counter += 1
    # overlapping + missing = atlas total
    print ('Missing # of genes: ' + str(missing_counter))
    norm_express.to_pickle(csvpath[:-3] + 'pkl')



# Object for each layer of the model
# Contains methods for subsetting data according to the metadata and label info files,
#                      dividing subsetted data 90-10 and 10 fold cv,
#                      finetuning and training each layer, calculating metrics and SHAP feature ranking,
#                      outputting final model for validation
# Keeps track of all outputted files, stores paths and names in instance variables
class Layer:
    
    # level is the column of the metadata file in which the name of the layer appears
    # xgbmodel is a trained XGBoost classifier object
    # cvmetrics, finalmetrics, cfm, pr are path names to those files
    # predictions, roc, fr are lists of path names
    def __init__(self, name, level, labeldict=None, xgbmodel=None, finetuning=None, cvmetrics=None,
                 finalmetrics=None, predictions=None, cfm=None, roc=None, pr=None, fr=None, pickle=None):
        self.name = name
        self.level = level
        self.labeldict = {}
        self.xgbmodel = None
        self.finetuning = None
        self.cvmetrics = None
        self.finalmetrics = None
        self.predictions = []
        self.cfm = None
        self.roc = []
        self.pr = None
        self.fr = []
        self.pickle = self.name + '_object.pkl'
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, layer2):
        return self.name==layer2.name
    
    def __repr__(self):
        return "<Layer: '%s', Level: %s, labeldict: %s, Trained: %s>" % (self.name, self.level, self.labeldict, self.trained())
    
    def __str__(self):
        return_str = 'Object: ' + self.pickle + '\n'
        return_str += 'Layer: ' + self.name + '\n'
        return_str += 'Level: ' + str(self.level) + '\n'
        return_str += 'Label Dictionary: ' + str(self.labeldict) + '\n'
        if self.finetuned():
            return_str += 'Finetuning Log: ' + self.finetuning + '\n'
        if self.trained():
            return_str += 'XGB Model: ' + self.name + '_xgbmodel.sav' + '\n'
            return_str += '10-Fold Cross Validation Metrics: ' + self.cvmetrics + '\n'
            return_str += 'Final Metrics (10%): ' + self.finalmetrics + '\n'
            return_str += 'Predictions (no rejection): ' + self.predictions[0] + '\n'
            return_str += 'Predictions (' + str(rejection_cutoff) + ' rejection cutoff): ' + self.predictions[1] + '\n'
            return_str += 'Confusion Matrix (10%, no rejection): ' + self.cfm + '\n'
            return_str += 'Micro/Macro ROC Curves: ' + self.roc[0] + '\n'
            return_str += 'Per-Class ROC Curves: ' + self.roc[1] + '\n'
            return_str += 'All Combined ROC Curves: ' + self.roc[2] + '\n'
            return_str += 'Precision-Recall Graph (10%, no rejection): ' + self.pr + '\n'
            if self.fr != []:
                return_str += 'Overall Feature Ranking: ' + self.fr[0] + '\n'
                for i in range(len(self.fr)-2):
                    return_str += self.name + '--' + list(self.labeldict.values())[i] + ' Feature Ranking: ' + self.fr[i+1] + '\n'
                return_str += 'Full Feature Importance List: ' + self.fr[len(self.fr)-1] + '\n'
        return return_str
    
    def finetuned(self):
        return self.finetuning is not None
    
    def trained(self):
        return self.xgbmodel is not None
    
    # Adds a value to the label dictionary if it is not already present
    # Utility function of fill_dict
    def add_dictentry(self, value):
        curr_dict = self.labeldict
        if value not in list(curr_dict.values()):
            curr_dict[len(curr_dict)] = value
            self.labeldict = curr_dict


    # Hyperparameter tuning for XGBoost using accuracy
    # Equivalent to sklearn.model_selection.RandomizedSearchCV -- w/o cv and w/ triangular dist
    # Parameters: eta, max_depth, subsamaple, colsample_bytree
    # n randomized trials: triangular distribution around the default or recommended xgboost value
    #                      rounded to a reasonable decimal place
    # Splits data according to user-provided testplit value, NO cross validation conducted
    # Returns parameters with lowest mean absolute error (mae) on 10% for actual training
    # Outputs csv file with accuracy of each class and mae for all trials, does not output any other metrics
    def finetune(self, trials, testsplit, normexprpkl, metadatacsv):
        X, Y, X_tr, X_test, Y_tr, Y_test, _ = self.read_data(normexprpkl, metadatacsv, testsplit)
        min_mae = 100000000000000
        f = open(path + self.name + '_finetuning.csv', 'a+')
        f.write('ETA,Max Depth,Subsample,Colsample by Tree,')
        for i in range(len(self.labeldict)-1):
            f.write(self.labeldict[i] + ' Accuracy')
            f.write(',')
        f.write('MAE\n')
        for i in range(trials):
            eta_temp = round(random.triangular(0,1,0.3),1)
            max_depth_temp = round(random.triangular(4,8,6))
            subsample_temp = round(random.triangular(0.01,1,0.5),1)
            colsample_bytree_temp = round(random.triangular(0.01,1,0.5),1)
            params = {'objective': 'multi:softprob', 'eta': eta_temp, 'max_depth': max_depth_temp, 'subsample': subsample_temp,
                    'colsample_bytree': colsample_bytree_temp, 'eval_metric': 'merror', 'seed': 840}
    
            mae, output_string = self.xgboost_model_shortver(X_tr, X_test, Y_tr, Y_test, params)
            f.write(str(eta_temp) + ',' + str(max_depth_temp) + ',' + str(subsample_temp) + ',' + str(colsample_bytree_temp) + ',')
            f.write(output_string + '\n')
            if mae < min_mae:
                min_mae = mae
                final_params = params
        print(final_params)
        self.finetuning = self.name + '_finetuning.csv'
        return final_params

    # XGBoost w/o cross validation, no output files, just accuracy score on user-provided testsplit value
    # Returns mean absolute error (mae) as float and csv output string
    #       Accuracy for each class and mae in a comma-separated string for classification
    # Only for finetuning!
    def xgboost_model_shortver(self, X_tr, X_test, Y_tr, Y_test, params):
        params['num_class'] = len(self.labeldict)
    
        d_tr = xgb.DMatrix(X_tr, Y_tr, feature_names=feature_names)
        model = xgb.train(params, d_tr, 20, verbose_eval=10)
        d_test = xgb.DMatrix(X_test, Y_test, feature_names=feature_names)
        probabilities_xgb = model.predict(d_test)
    
        returned_str = ''
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        cm = confusion_matrix(Y_test, predictions_xgb)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        mae = 1-sum(cm.diagonal())/len(cm.diagonal())
        returned_str = ','.join(map(str, cm.diagonal())) + ',' + str(mae)
        return mae, returned_str
    
    
    
    # Trains one layer in the classification
    # Conducts feature ranking with SHAP if instructed by user on full final model
    def train_layer(self, normexprpkl, metadatacsv, params, testsplit, rejectcutoffs, featrank_on, frsplit):
        X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames = self.read_data(normexprpkl, metadatacsv, testsplit)
        final_model = self.xgboost_model(X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames, params, rejectcutoffs)
        if featrank_on is True:
            self.feature_ranking(final_model, normexprpkl, metadatacsv, frsplit)
    
    
    # Trains XGBoost for a layer of classification given parameters
    # Splits data according to user-provided testsplit
    # Conducts 10-fold cross validation on (1-testsplit)% of data, outputs cv metrics
    # Retrains model on (1-testsplit)% to output metrics when tested on holdout (testsplit)%
    # Retrains final saved model on 100% of data, returns for feature ranking
    def xgboost_model(self, X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames, params, rejectcutoffs):
        params['num_class'] = len(self.labeldict)
    
        # 10-fold CV on the 90%
        kfold = 10
        sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.1, random_state=720)
        for i, (train_index, test_index) in enumerate(sss.split(X_tr, Y_tr)):
            print('[Fold %d/%d]' % (i + 1, kfold))
            X_train, X_valid = X_tr[train_index], X_tr[test_index]
            Y_train, Y_valid = Y_tr[train_index], Y_tr[test_index]
            d_train = xgb.DMatrix(X_train, Y_train, feature_names=feature_names)
            cv_model = xgb.train(params, d_train, 20, verbose_eval=500)
            self.xgbmodel = cv_model # temporarily set xgbmodel to cv 0.9*(1-testsplit)% model
            self.model_metrics('cv', X_valid, Y_valid)
            self.cvmetrics = self.name + '_cvmetrics.txt'
    
        # 90% model, 10% testing
        d_tr = xgb.DMatrix(X_tr, Y_tr, feature_names=feature_names)
        temp_model = xgb.train(params, d_tr, 20, verbose_eval=500)
        self.xgbmodel = temp_model # temporarily set xgbmodel to (1-testsplit)% model
        self.model_metrics('final', X_test, Y_test)
        self.finalmetrics = self.name + '_finalmetrics.txt'
        self.cell_predictions('training', rejectcutoffs, test_cellnames, X_test, Y_test)
        for cutoff in rejectcutoffs:
            self.predictions.append(self.name + '_trainingpredictions_reject' + str(cutoff) + '.csv')
        self.cfsn_mtx('training', X_test, Y_test)
        self.cfm = self.name + '_confusionmatrix.svg'
        self.roc_curves(X_test, Y_test)
        self.roc.append(self.name + '_miacroroc.svg')
        self.roc.append(self.name + '_classroc.svg')
        self.roc.append(self.name + '_allroc.svg')
        self.pr_curves(X_test, Y_test)
        self.pr = self.name + '_allpr.svg'
    
        # final 100% model
        d_all = xgb.DMatrix(X, Y, feature_names=feature_names)
        final_model = xgb.train(params, d_all, 20, verbose_eval=500)
        pickle.dump(final_model, open(path + self.name + '_xgbmodel.sav', 'wb'))
        self.xgbmodel = final_model
        
        # retrain 100% for feature ranking and to avoid python variable mutability
        returned_model = xgb.train(params, d_all, 20, verbose_eval=500)
        return returned_model
    
    
    # Outputs predictions of the 90% model on the 10% test set in a given dataset
    # 1 csv file outputted for each rejection cutoff in the list rejectioncutoffs
    # A cutoff of 0.5 means the probability of the label must be >=0.5 for a prediction to be called
    # A cutoff of 0 is equivalent to no rejection option
    def predict_layer(self, rejectcutoffs, normexprpkl, metadatacsv=None):
        X, Y, all_cellnames = self.read_data(normexprpkl, metadatacsv) # Y will be None if metadatacsv=None
        self.cell_predictions('validation', rejectcutoffs, all_cellnames, X, Y) # can handle Y=None
        if metadatacsv != None:
            self.cfsn_mtx('validation', X, Y)
            self.model_metrics('validation', X, Y)


    # Reads the normalized gene expression data into a norm_expr dataframe
    # Reads the specified column in the metadata csv into a label dataframe
    # Metadata CSV file must contain samples as row names, first column name (cell A1) is empty
    # subsetcolumn and subsetcriterium are strings for extracting rows with the specified label in the specified column
    #                                 eg. for per timepoint classification
    #                                 can both be set to None if all samples are to be included
    # Converts norm_expr and label dataframes into numpy arrays X and Y, splits them into 90/10
    # Retrieves cell names of samples in 10% testing data
    # global feature_names defined here, list of all gene names -> should be equivalent for all csvs (except subsetted versions)
    def read_data(self, normexprpkl, metadatacsv=None, testsplit=None):
        norm_express = pd.read_pickle(normexprpkl)
        global feature_names
        feature_names = list(norm_express)
        print(norm_express.shape)
        
        # If validation w/o metadata, metadata not provided, return all data w/o subsetting
        if metadatacsv is None:
            X = norm_express.values
            norm_express.index.name = 'cells'
            norm_express.reset_index(inplace=True)
            Y = norm_express.values
            all_cellnames = Y[:,0]
            all_cellnames = all_cellnames.ravel()
            Y = None
            return X, Y, all_cellnames
        
        tp = pd.read_csv(metadatacsv, iterator=True, chunksize=1000)
        labels = pd.concat(tp, ignore_index=True)
        labels.set_index('Unnamed: 0', inplace=True)
        labels.index.names = [None]
    
        # Only root level will have no subsetcolumn, all other levels require subsetting
        # self.level is one less than what user sees since index is dropped above
        metadata_columns = list(labels.columns.values)
        labelcolumn = metadata_columns[self.level]
        subsetcolumn = metadata_columns[self.level-1] if self.level > 0 else None
        subsetcriterium = self.name
    
        # Filter out cells if subsetting necessary, keep only data from given criterium
        if subsetcolumn != None:
            # Reindex norm_express and labels based on cell names in given criterium
            temp = labels.loc[labels[subsetcolumn] == subsetcriterium]
            labels = labels.reindex(index=temp.index)
            norm_express = norm_express.reindex(index=temp.index)
        print (labels[labelcolumn].value_counts())
        # Remove cells with labels not provided in the dictionary, replace present keys with values
        temp = labels.loc[labels[labelcolumn].isin(list(self.labeldict.values()))]
        labels = labels.reindex(index=temp.index)
        norm_express = norm_express.reindex(index=temp.index)
        print (labels[labelcolumn].value_counts())
        for i in range(len(self.labeldict)):
            labels = labels.replace(self.labeldict[i],i)
        
        # If validation with metadata, testsplit not provided, return all data w/o train test split
        if testsplit is None:
            X = norm_express.values
            labels.index.name = 'cells'
            labels.reset_index(inplace=True)
            Y = labels.values
            all_cellnames = Y[:,0]
            all_cellnames = all_cellnames.ravel()
            Y = labels[labelcolumn].values
            return X, Y, all_cellnames
        
        X = norm_express.values
        
        # Get cell names in 10% test set - make cell names into a new column to save in a list
        labels.index.name = 'cells'
        labels.reset_index(inplace=True)
        Y = labels.values
        np.random.seed(0)
        X, Y = shuffle(X, Y)
        X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=testsplit, random_state=42, shuffle = True, stratify = Y[:,self.level+1])
        test_cellnames = Y_test[:,0]
        test_cellnames = test_cellnames.ravel()

        # Remake train and test split to save X, Y, X_tr, X_test, Y_tr, Y_test
        X = norm_express.values
        Y = labels[labelcolumn].values
        np.random.seed(0)
        X, Y = shuffle(X, Y)
        X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size=testsplit, random_state=42, shuffle = True, stratify = Y)
        print('Training Samples: ' + str(len(X_tr)) + ', Testing Samples: ' + str(len(X_test)))
        return X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames;
    
    
    # Calculates metrics of the Layer model on a provided test set and outputs it in a file
    # cv_final_val is a naming string to differentiate between cv, final, and validation metrics
    def model_metrics(self, cv_final_val, X_test, Y_test):
        d_test = xgb.DMatrix(X_test, Y_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        target_names = [str(x) for x in sorted(list(set(Y_test).union(predictions_xgb)))]
        metrics = classification_report(Y_test, predictions_xgb, target_names=target_names)
        with open(path + self.name + '_' + cv_final_val + 'metrics.txt', 'a+') as f:
            print(metrics, file=f)
    
    
    # Outputs predictions of the Layer model on a provided test set
    # Adds a key value pair in the labeldict for an unclassified class, removes it when completed
    # 1 csv file outputted for each rejection cutoff in the list rejectioncutoffs
    # A cutoff of 0.5 means the probability of the label must be >=0.5 for a prediction to be called
    # A cutoff of 0 is equivalent to no rejection option
    # train_val is a naming string to differentiate between train/test predictions and validation predictions
    def cell_predictions(self, train_val, rejectcutoffs, test_cellnames, X_test, Y_test=None):
        d_test = xgb.DMatrix(X_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        self.labeldict[len(self.labeldict)] = 'Unclassified'
        for cutoff in rejectcutoffs:
            counter = 0
            for i in range(len(probabilities_xgb)):
                if probabilities_xgb[i,probabilities_xgb.argmax(axis=1)[i]] < cutoff:
                    predictions_xgb[i] = len(self.labeldict)-1
                    counter += 1
            print('Unclassified # of cells w/ probability cutoff=' + str(cutoff) + ': ' + str(counter))
            f = open(path + self.name + '_' + train_val + 'predictions_reject' + str(cutoff) + '.csv','w')
            if Y_test is not None:
                f.write('Cell ID,True Label,Predicted Label')
            else:
                f.write('Cell ID,Predicted Label')
            for j in range(len(self.labeldict)-1):
                f.write(',')
                f.write(self.labeldict[j] + ' Probability')
            f.write('\n')
            for i in range(test_cellnames.size):
                f.write(test_cellnames[i])
                f.write(',')
                if Y_test is not None:
                    f.write(self.labeldict[Y_test[i]])
                    f.write(',')
                f.write(self.labeldict[predictions_xgb[i]])
                for j in range(len(self.labeldict)-1):
                    f.write(',')
                    f.write(str(probabilities_xgb[i][j]))
                f.write('\n')
            f.close()
        del self.labeldict[len(self.labeldict)-1]
    
    
    # Outputs a confusion matrix of the Layer model's results on a provided test set
    def cfsn_mtx(self, train_val, X_test, Y_test):
        d_test = xgb.DMatrix(X_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        classnames = [self.labeldict[x] for x in sorted(list(set(Y_test).union(predictions_xgb)))]
        
        cm = confusion_matrix(Y_test, predictions_xgb)
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classnames))
        plt.xticks(tick_marks, classnames, rotation=45)
        plt.yticks(tick_marks, classnames)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '%0.3f' % cm[i,j] if cm[i,j] > 0 else 0,
                horizontalalignment='center', verticalalignment='center',
                color='white' if cm[i,j] > thresh else 'black', fontsize=6)
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(path + self.name + '_' + train_val + 'confusionmatrix.svg')
        plt.clf()


    # Creates ROC curves for the Layer model on a provided test set
    # Calculates curves using probability values and thresholds
    # Outputs 3 figures: micro and macro ROC curves, per class ROC curves, and all combined
    def roc_curves(self, X_test, Y_test):
        d_test = xgb.DMatrix(X_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        n_classes = len(self.labeldict)
        lw = 2
    
        # One hot encode Y_test and predictions arrays
        y_test = np.zeros((Y_test.size, Y_test.max()+1))
        y_test[np.arange(Y_test.size),Y_test] = 1
        y_score = probabilities_xgb
    
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
        # Compute macro-average ROC curve and ROC area
    
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
        # Plot micro/macro ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
    
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Overall ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(path + self.name + '_miacroroc.svg')
        plt.clf()
    
        # Plot class ROC curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Class ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(path + self.name + '_classroc.svg')
        plt.clf()
    
        # Plot all ROC curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Trueas Positive Rate')
        plt.title('All ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(path + self.name + '_allroc.svg')
        plt.clf()
    
    
    # Creates precision-recall curves for the Layer model on a provided test set
    # Calculates curves using probability values and thresholds
    # Outputs 1 figure: per class PR curvs and a micro PR curve
    def pr_curves(self, X_test, Y_test):
        d_test = xgb.DMatrix(X_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        n_classes = len(self.labeldict)
        lw = 2
    
        # One hot encode Y_test and predictions arrays
        y_test = np.zeros((Y_test.size, Y_test.max()+1))
        y_test[np.arange(Y_test.size),Y_test] = 1
        y_score = probabilities_xgb
    
        # Compute precision and recall for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    
        # Compute micro-average precision and recall
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
    
        # Plot micro + class PR curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='PR curve of class {0} (area = {1:0.2f})'
                     ''.format(i, average_precision[i]))
        plt.plot(recall["micro"], precision["micro"],
                 label='micro-average PR curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curves')
        plt.legend(loc="lower left")
        plt.savefig(path + self.name + '_allpr.svg')
        plt.clf()

    
    # Outputs SHAP feature ranking plots overall AND for each class if conducting classification
    # Uses same procedure for subsetting norm_expr as read_data(...) if column & criterium are provided
    # frsplit automatically set to 0.3 if not provided
    def feature_ranking(self, model, normexprpkl, metadatacsv, frsplit):
        norm_express = pd.read_pickle(normexprpkl)
        tp = pd.read_csv(metadatacsv, iterator=True, chunksize=1000)
        labels = pd.concat(tp, ignore_index=True)
        labels.set_index('Unnamed: 0', inplace=True)
        labels.index.names = [None]
        # Only root level will have no subsetcolumn, all other levels require subsetting
        # self.level is one less than what user sees since cellname column is dropped above
        metadata_columns = list(labels.columns.values)
        labelcolumn = metadata_columns[self.level]
        subsetcolumn = metadata_columns[self.level-1] if self.level > 0 else None
        subsetcriterium = self.name
        # Filter out cells if subsetting necessary, keep only data from given criterium
        if subsetcolumn != None:
            # Reindex norm_express and labels based on cell names in given criterium
            temp = labels.loc[labels[subsetcolumn] == subsetcriterium]
            labels = labels.reindex(index=temp.index)
            norm_express = norm_express.reindex(index=temp.index)
        # Remove cells with labels not provided in the dictionary
        if self.labeldict != None:
            temp = labels.loc[labels[labelcolumn].isin(list(self.labeldict.values()))]
            labels = labels.reindex(index=temp.index)
            norm_express = norm_express.reindex(index=temp.index)
        norm_express, labels = shuffle(norm_express, labels)
        if frsplit is None:
            frsplit = 0.3
        norm_express = train_test_split(norm_express, labels, test_size=frsplit, random_state=42, shuffle = True, stratify = labels[labelcolumn])[0]
    
        model_bytearray = model.save_raw()[4:]
        model.save_raw = lambda: model_bytearray
        shap_values = shap.TreeExplainer(model).shap_values(norm_express)
        shap.summary_plot(shap_values, norm_express, show=False)
        plt.tight_layout()
        plt.savefig(path + self.name + '_overallfr.svg')
        plt.clf()
        self.fr.append(self.name + '_overallfr.svg')
        for i in range(len(self.labeldict)):
            shap.summary_plot(shap_values[i], norm_express, show=False)
            plt.tight_layout()
            plt.savefig(path + self.name + '_class'+str(i)+'fr.svg')
            #np.savetxt(path + name + '_class'+str(i)+'fr.csv', shap_values[i], delimiter=",")
            plt.clf()
            self.fr.append(self.name + '_class'+str(i)+'fr.svg')
        
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(norm_express.columns, sum(vals))), columns=['Gene','Feature Importance Value'])
        feature_importance.sort_values(by=['Feature Importance Value'], ascending=False, inplace=True)
        feature_importance.to_csv(path + self.name + '_featureimportances.csv', sep=',', encoding='utf-8')
        self.fr.append(self.name + '_featureimportances.csv')


# Main function: reads in user input, selects a pathway, and trains / predicts
def main():
    ## CELLPY PATHWAYS
    # 1a. automatic everything: training and validation w/ val_metadata
    #           (path, normexpr, labelinfo, metadata, rejection_cutoff, val_normexpr, val_metadata)
    # 1b. automatic everything: training and validation w/o val_metadata
    #           (path, normexpr, labelinfo, metadata, rejection_cutoff, val_normexpr)
    # 2.  just training 
    #           (path, normexpr, labelinfo, metadata, rejection_cutoff)
    # 3a. just validation: load in Layer objects and validation w/ val_metadata
    #           (path, layer_paths, rejection_cutoff, val_normexpr, val_metdata)
    # 3b. just validation: load in Layer objects and validation w/o val_metadata
    #           (path, layer_paths, rejection_cutoff, val_normexpr)
    time_start = time.perf_counter()
    
    # All variables used for training and prediction set to None
    global path, rejection_cutoff
    path = os.getcwd()
    user_train = False
    user_predict = False
    train_normexpr = None
    labelinfo = None
    train_metadata = None
    testsplit = None
    featrank_on = None
    frsplit = None
    rejection_cutoff = None
    pred_normexpr = None
    pred_metadata = None
    layer_paths = None
    
    # Command Line Interface
    # runMode must be 'trainOnly', 'predictOnly', or 'trainAndPredict'
    # trainNormExpr, labelInfo, trainMetadata are paths to their respective training files
    # testSplit is a float between 0 and 1 denoting the percentage of data to holdout for testing
    # featureRanking must be 'on' or 'off'
    # rejectionCutoff is a float between 0 and 1 denoting the minimum probability for a prediction to not be rejected
    # predNormExpr, predMetadata are paths to their respective prediction files
    # layerObjectPath is a comma-separated list of paths to the Layer objects that the user wants to predict on the predNormExpr
    args = sys.argv[1:]
    options, args = getopt.getopt(args, '',
                        ['runMode=', 'trainNormExpr=', 'labelInfo=', 'trainMetadata=', 'testSplit=', 'featureRanking=', 'featureRankingSplit=',
                         'rejectionCutoff=', 'predNormExpr=', 'predMetadata=', 'layerObjectPaths='])
    for name, value in options:
        if name in ['--runMode']:
            if value == 'trainOnly':
                user_train = True
            elif value == 'predictOnly':
                user_predict = True
            elif value == 'trainAndPredict':
                user_train = True
                user_predict = True
        if name in ['--trainNormExpr']:
            train_normexpr = value
        if name in ['--labelInfo']:
            labelinfo = value
        if name in ['--trainMetadata']:
            train_metadata = value
        if name in ['--testSplit']:
            testsplit = float(value)
        if name in ['--featureRanking']:
            if value == 'on':
                featrank_on = True
            elif value == 'off':
                featrank_on = False
        if name in ['--featureRankingSplit']:
            frsplit = value
        if name in ['--rejectionCutoff']:
            rejection_cutoff = float(value)
        if name in ['--predNormExpr']:
            pred_normexpr = value
        if name in ['--predMetadata']:
            pred_metadata = value
        if name in ['--layerObjectPaths']:
            layer_paths = value.split(',')
    
    # Check user provided variables follow an above cellpy pathway
    passed_options = check_combinations(user_train, user_predict, train_normexpr, labelinfo, train_metadata, testsplit,
                                        featrank_on, frsplit, rejection_cutoff, pred_normexpr, pred_metadata, layer_paths)
    if passed_options is False:
        raise ValueError('see printed error log above')
    
    # Create cellpy_results directory with timestamp
    newdir = 'cellpy_results_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    path = os.path.join(path, newdir)
    if not os.path.isdir(path):
        print('Created directory "cellpy_results" in cwdir: ' + path)
        os.mkdir(path)
    os.chdir(path)
    
    # Check training files exist if training option called
    passed_train = None
    passed_predict = None
    if user_train is True:
        passed_train = check_trainingfiles(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff, frsplit)
    # Check prediction files exist if prediction option called
    if user_predict is True:
        passed_predict = check_predictionfiles(pred_normexpr, pred_metadata, layer_paths) 
    if passed_train is False or passed_predict is False:
        raise ValueError('see printed error log above')
        
    # If training option is called and feasible
    if user_train is True and passed_train is True:
        all_layers = training(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff, featrank_on, frsplit)
    # If prediction option is called and feasible
    if user_predict is True and passed_predict is True:
        # if training option was called by user and passed check
        if user_train is True and passed_train is True:
            prediction(pred_normexpr, pred_metadata, all_layers=all_layers)
        # user inputs list of Layer objects
        elif layer_paths is not None:
            prediction(pred_normexpr, pred_metadata, object_paths=layer_paths)

    # Print computational time and memory required
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))


if __name__ == "__main__":
    main()
