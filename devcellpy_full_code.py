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
import scanpy as sc


# Ensures given files satisfy one of the possible pathways provided by devcellpy
# Ensures user input for train or predict matches file inputs
# Certain files must appear together for training and/or validation to proceed
def check_combinations(user_train, user_predictOne, user_predictAll, user_fr, train_normexpr, labelinfo, train_metadata, testsplit,
                       rejection_cutoff, val_normexpr, val_metadata, layer_paths, frsplit):
    passed = True
    if user_train is None:
        print('ERROR: Run mode must be provided to resume')

    # if the user selected the 'trainAll' option
    train_list = [train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff]
    if user_train is True:
        print('Training option selected')
        if train_list[0] is None:
            print('ERROR: Normalized expression matrix for training must be provided to resume')
            passed = False
        if train_list[1] is None:
            print('ERROR: Label information file must be provided to resume')
            passed = False
        if train_list[2] is None:
            print('ERROR: Metadata file for training must be provided to resume')
            passed = False
        if train_list[3] is None:
            print('WARNING: Test split amount not provided, training will proceed w/o cross-validation and metric calculations')
        if train_list[4] is None:
            print('ERROR: Rejection cutoff value must be provided to resume')
            passed = False

    # if the user selected the 'predictOne' or 'predictAll' options
    predict_list = [val_normexpr, val_metadata, layer_paths, rejection_cutoff]
    if (user_predictOne is True) or (user_predictAll is True):
        if val_metadata is not None and user_predictOne is True:
            print('Independent prediction option with accuracy calculation selected')
        elif val_metadata is not None and user_predictAll is True:
            print('WARNING: Dependent prediction option does not conduct accuracy calculation, provided metadata file will not be used')
        elif val_metadata is None and user_predictOne is True:
            print('Independent prediction option without accuracy calculation selected')
        elif val_metadata is None and user_predictOne is True:
            print('Dependent prediction option selected')
        if predict_list[0] is None:
            print('ERROR: Normalized expression matrix for prediction must be provided to resume')
            passed = False
        if predict_list[2] is None:
            print('ERROR: Path names to Layer objects must be provided to resume')
            passed = False
        if predict_list[3] is None:
            print('ERROR: Rejection cutoff value must be provided to resume')
            passed = False

    # if the user selected the 'featureRankingOne' option
    fr_list = [train_normexpr, train_metadata, layer_paths, frsplit]
    if user_fr is True:
        if fr_list[0] is None:
            print('ERROR: Normalized expression matrix for training must be provided to resume')
            passed = False
        if fr_list[1] is None:
            print('ERROR: Metadata file for training must be provided to resume')
            passed = False
        if fr_list[2] is None:
            print('ERROR: Path names to Layer objects must be provided to resume')
            passed = False
        if fr_list[3] is None:
            print('WARNING: Feature ranking split not provided but feature ranking on, will be automatically set to 0.3')
    return passed


# Ensures all the user given variables for training exist or are in bounds
def check_trainingfiles(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff):
    passed = True
    if not os.path.exists(train_normexpr):
        print('ERROR: Given normalized expression data file for training does not exist')
        passed = False
    if not os.path.exists(labelinfo):
        print('ERROR: Given label info file does not exist')
        passed = False
    if not os.path.exists(train_metadata):
        print('ERROR: Given metadata file for training does not exist')
        passed = False
    if testsplit is not None and (testsplit > 1 or testsplit < 0):
        print('ERROR: Given test split percentage must be a value between 0 and 1')
        passed = False
    if rejection_cutoff > 1 or rejection_cutoff < 0:
        print('ERROR: Given rejection cutoff must be a value between 0 and 1')
        passed = False
    return passed


# Conducts training in all layers separated into different folders by name
# Creates directory 'training' in devcellpy_results folder, defines 'Root' as topmost layer
# Conducts finetuning on Root layer with 50 iterations
def training(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff):
    global path, orig_path
    path = os.path.join(path, 'devcellpy_training_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    orig_path = path
    os.mkdir(path)
    os.chdir(path)
    if train_normexpr[-3:] == 'csv':
        csv2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-3] + 'pkl'
    elif train_normexpr[-4:] == 'h5ad':
        h5ad2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-4] + 'pkl'
    else:
        raise ValueError('Format of normalized expression data file not recognized')
    all_layers = [Layer('Root', 0, 'Root')]
    construct_tree(labelinfo, all_layers)
    print(all_layers)
    for layer in all_layers:
        path = os.path.join(path, layer.path)
        os.mkdir(path)
        os.chdir(path)
        path = path + '/'
        if skip == None or (skip != None and layer.name != skip):
            if std_params is False:
                parameters = layer.finetune(50, 30, train_normexpr, train_metadata)
            else:
                parameters = {'objective': 'multi:softprob', 'eta': 0.2, 'max_depth': 6, 'subsample': 0.5,
                              'colsample_bytree': 0.5, 'eval_metric': 'merror', 'seed': 840}
            print(parameters)
            layer.train_layer(train_normexpr, train_metadata, parameters, testsplit, [0, rejection_cutoff])
            export_layer(layer, all_layers)
        os.chdir(orig_path) # return to training directory
        path = os.getcwd()
    path = path + '/'
    training_summary(all_layers)
    print('Training Complete')
    os.chdir('..') # return to devcellpy directory
    path = os.getcwd()


# Converts the normalized expression csv into a pkl
# Expression CSV file must contain genes as row names, samples as column names
# First column name (cell A1) is 'gene'
def csv2pkl(csvpath):
    tp = pd.read_csv(csvpath, iterator=True, chunksize=1000)
    norm_express = pd.concat(tp, ignore_index=True)
    norm_express.set_index('gene', inplace=True)
    norm_express.index.names = [None]
    norm_express = norm_express.T
    print (norm_express.head())
    norm_express.to_pickle(csvpath[:-3] + 'pkl')


# Converts the Seurat h5ad object into a pkl
def h5ad2pkl(h5adpath):
    adata = sc.read_h5ad(h5adpath)
    df = pd.DataFrame(adata.X.toarray(), columns = adata.var.index, index = adata.obs.index)
    df.to_pickle(h5adpath[:-4] + 'pkl')


# Constructs a list of all Layer objects from a labelinfo file
# Initalizes each Layer object with a name, level #, and label dictionary
def construct_tree(labelinfo, all_layers):
    labeldata = fill_data(labelinfo)
    fill_dict(labeldata, all_layers)


# Function of construct_tree, fills the vertical columns of the labeldata file
def fill_data(labelinfo):
    labeldata = list(csv.reader(open(labelinfo, encoding='utf-8-sig')))
    # Fill in all the gaps in column 1 (layer 1)
    for i in range(1, len(labeldata)):
        if labeldata[i][0]=='':
            labeldata[i][0] = labeldata[i-1][0]
    # Fill in gaps in remaining columns only if the cell 1 left equals the value of the cell 1 up and 1 left
    for i in range(1, len(labeldata)):
        for j in range(1, len(labeldata[0])):
            if labeldata[i][j]=='' and labeldata[i][j-1]==labeldata[i-1][j-1]:
                labeldata[i][j] = labeldata[i-1][j]
    return labeldata


# Function of construct_tree after fill_data, reads the edited / filled labeldata file
# Constructs the main list structure and initializes all Layer objects in it
def fill_dict(labeldata, all_layers):
    for i in range(len(labeldata)):
        # Initializes the Root layer dictionary with labels in the first column of labeldata
        if find_layer(all_layers, labeldata[i][0]) is None:
            root_layer = find_layer(all_layers, 'Root')
            root_layer.add_dictentry(labeldata[i][0])
        # Fills dictionaries in layers j-1 given the existence of a label in column j
        for j in range(1,len(labeldata[0])):
            if labeldata[i][j] != '':
                if find_layer(all_layers, labeldata[i][j-1]) is None:
                    if j == 1: # prev column is label under Root
                        all_layers.append(Layer(labeldata[i][j-1], j, 'Root/' + alphanumeric(labeldata[i][j-1])))
                    else:
                        prev_layer = find_layer(all_layers, labeldata[i][j-2])
                        all_layers.append(Layer(labeldata[i][j-1], j, prev_layer.path + '/' + alphanumeric(labeldata[i][j-1])))
                prev_layer = find_layer(all_layers, labeldata[i][j-1])
                prev_layer.add_dictentry(labeldata[i][j])
            else:
                break


# Utility function, searches a list of all_layers for a layer with the given name
def find_layer(all_layers, name):
    for layer in all_layers:
        if layer.name == name:
            return layer
    return None


# Exports a trained Layer as a pickle file
def export_layer(layer, all_layers):
    if skip != None:
        tp_layer = find_layer(all_layers, skip)
    if layer is not None and layer.trained() is True:
        with open(path + alphanumeric(layer.name) + '_object.pkl', 'wb') as output:
            if skip != None and tp_layer is not None and tp_layer.inDict(layer.name):
                layer.predictname = tp_layer.name 
            pickle.dump(layer, output, pickle.HIGHEST_PROTOCOL)


# Summarizes and lists the path names of all the files created during training
# Prints the summary of each Layer object
def training_summary(all_layers):
    f = open(path + 'training_summaryfile.txt', 'w')
    for layer in all_layers:
        f.write(str(layer))
        f.write('\n')


# Ensures all the user given variables for predictOne or predictAll exist and are in the correct format
def check_predictionfiles(val_normexpr, val_metadata, layer_paths):
    passed = True
    if not os.path.exists(val_normexpr):
        print('ERROR: Given validation normalized expression data file for prediction does not exist')
        passed = False
    if val_metadata != None and not os.path.exists(val_metadata):
        print('ERROR: Given validation metadata file for prediction does not exist')
        passed = False
    # check all layer paths are objects and contain a trained xgb model
    for i in range(len(layer_paths)):
        layer_path = layer_paths[i]
        if not os.path.exists(layer_path):
            print('ERROR: Given Layer object ' + layer_path + ' does not exist')
            passed = False
        else:
            layer = pd.read_pickle(layer_path)
            if layer.trained() is False:
                print('ERROR: Given Layer object ' + layer_path + ' is not trained')
                passed = False
    return passed


# Conducts prediction in specified layers separated into different folders by name
# Creates directory 'predictionOne' in devcellpy_results folder, defines 'Root' as topmost layer
def predictionOne(val_normexpr, val_metadata, object_paths):
    global path, orig_path
    path = os.path.join(path, 'devcellpy_predictOne_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    orig_path = path
    os.mkdir(path)
    os.chdir(path)
    all_layers = import_layers(object_paths)
    featurenames = all_layers[0].xgbmodel.feature_names
    reorder_pickle(val_normexpr, featurenames)
    if val_normexpr[-3:] == 'csv':
        val_normexpr = val_normexpr[:-3] + 'pkl'
    elif val_normexpr[-4:] == 'h5ad':
        val_normexpr = val_normexpr[:-4] + 'pkl'
    for layer in all_layers:
        path = os.path.join(path, layer.name)
        os.mkdir(path)
        os.chdir(path)
        path = path + '/'
        layer.predict_layer([0, rejection_cutoff], val_normexpr, val_metadata)
        os.chdir(orig_path) # return to prediction directory
        path = os.getcwd()
    print('Prediction Complete')
    os.chdir('..') # return to devcellpy directory
    path = os.getcwd()


# Conducts prediction in all layers in one folder
# Creates directory 'predictionAll' in devcellpy_results folder, defines 'Root' as topmost layer
def predictionAll(val_normexpr, object_paths):
    global path
    path = os.path.join(path, 'devcellpy_predictAll_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.mkdir(path)
    os.chdir(path)
    path = path + '/'

    all_layers = import_layers(object_paths)
    print(all_layers)
    featurenames = all_layers[0].xgbmodel.feature_names
    reorder_pickle(val_normexpr, featurenames)
    if val_normexpr[-3:] == 'csv':
        val_normexpr = val_normexpr[:-3] + 'pkl'
    elif val_normexpr[-4:] == 'h5ad':
        val_normexpr = val_normexpr[:-4] + 'pkl'

    norm_express = pd.read_pickle(val_normexpr)
    feature_names = list(norm_express)
    print(norm_express.shape)
    X = norm_express.values

    X = norm_express.values
    norm_express.index.name = 'cells'
    norm_express.reset_index(inplace=True)
    Y = norm_express.values
    all_cellnames = Y[:,0]
    all_cellnames = all_cellnames.ravel()
    Y = None

    f = open(path + 'predictionall_reject' + str(rejection_cutoff) + '.csv','w')
    for i in range(len(all_cellnames)):
        sample = np.array(X[i])#.reshape((-1,1))
        sample = np.vstack((sample, np.zeros(len(feature_names))))
        d_test = xgb.DMatrix(sample, feature_names=feature_names)
        root_layer = find_predictlayer(all_layers, 'Root')
        root_layer.add_dictentry('Unclassified')
        probabilities_xgb = root_layer.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        if probabilities_xgb[0,probabilities_xgb.argmax(axis=1)[0]] < rejection_cutoff:
            predictions_xgb[0] = len(root_layer.labeldict)-1
        f.write(all_cellnames[i])
        f.write(',')
        f.write(root_layer.labeldict[predictions_xgb[0]])

        search_str = root_layer.labeldict[predictions_xgb[0]]
        del root_layer.labeldict[len(root_layer.labeldict)-1]
        while(True):
            curr_layer = find_predictlayer(all_layers, search_str)
            if curr_layer is not None:
                curr_layer.add_dictentry('Unclassified')
                probabilities_xgb = curr_layer.xgbmodel.predict(d_test)
                predictions_xgb = probabilities_xgb.argmax(axis=1)
                if probabilities_xgb[0,probabilities_xgb.argmax(axis=1)[0]] < rejection_cutoff:
                    predictions_xgb[0] = len(curr_layer.labeldict)-1
                f.write(',')
                f.write(curr_layer.labeldict[predictions_xgb[0]])
                search_str = curr_layer.labeldict[predictions_xgb[0]]
                del curr_layer.labeldict[len(curr_layer.labeldict)-1]
            else:
                break
        f.write('\n')
    f.close()

    print('Prediction Complete')
    os.chdir('..') # return to devcellpy directory
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
def reorder_pickle(path, featurenames):
    # Convert data into pickles
    if train_normexpr[-3:] == 'csv':
        csvpath = path
        tp = pd.read_csv(csvpath, iterator=True, chunksize=1000)
        norm_express = pd.concat(tp, ignore_index=True)
        norm_express.set_index('gene', inplace=True)
        norm_express.index.names = [None]
        norm_express = norm_express.T
        # print (norm_express.head())
        # print(norm_express.T.duplicated().any())
        norm_express.to_pickle(csvpath[:-3] + 'pkl')
    elif train_normexpr[-4:] == 'h5ad':
        h5adpath = path
        adata = sc.read_h5ad(h5adpath)
        norm_express = pd.DataFrame(adata.X.toarray(), columns = adata.var.index, index = adata.obs.index)
        norm_express.to_pickle(h5adpath[:-4] + 'pkl')
    else:
        raise ValueError('Format of normalized expression data file not recognized')
    print ('Training Data # of  genes: ' + str(len(featurenames)))

    ## Manually reorder columns according to training data index
    # Reorder overlapping genes, remove genes not in training data
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
            norm_express.insert(i, featurenames[i], None)
            missing_counter += 1
        i += 1
    while i < len(featurenames):
        norm_express.insert(i, featurenames[i], None)
        i += 1
        missing_counter += 1
    # Overlapping + missing = training total
    print ('Missing # of genes: ' + str(missing_counter))
    norm_express.to_pickle(csvpath[:-3] + 'pkl')


# Utility function, searches a list of all_layers for a layer with the given name
def find_predictlayer(all_layers, name):
    for layer in all_layers:
        if layer.predictname == name:
            return layer
    return None


def check_featurerankingfiles(train_normexpr, train_metadata, layer_paths, frsplit):
    passed = True
    if not os.path.exists(train_normexpr):
        print('ERROR: Given normalized expression data file for training does not exist')
        passed = False
    if not os.path.exists(train_metadata):
        print('ERROR: Given metadata file for training does not exist')
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
    if frsplit is not None and (frsplit > 1 or frsplit < 0):
        print('ERROR: Given feature ranking split must be a value between 0 and 1')
        passed = False
    return passed


# Conducts prediction in all layers separated into different folders by name
# Creates directory 'prediction' in devcellpy_results folder, defines 'Root' as topmost layer
def featureranking(train_normexpr, train_metadata, object_paths, frsplit):
    global path, orig_path
    path = os.path.join(path, 'devcellpy_featureranking_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    orig_path = path
    os.mkdir(path)
    os.chdir(path)
    if train_normexpr[-3:] == 'csv':
        csv2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-3] + 'pkl'
    elif train_normexpr[-4:] == 'h5ad':
        h5ad2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-4] + 'pkl'
    else:
        raise ValueError('Format of normalized expression data file not recognized')
    all_layers = import_layers(object_paths)
    for layer in all_layers:
        path = os.path.join(path, layer.name)
        os.mkdir(path)
        os.chdir(path)
        path = path + '/'
        layer.featurerank_layer(train_normexpr, train_metadata, frsplit)
        os.chdir(orig_path) # return to prediction directory
        path = os.getcwd()
    print('Feature Ranking Complete')
    os.chdir('..') # return to devcellpy directory
    path = os.getcwd()


# Remove non-alphanumeric characters from string
def alphanumeric(str):
    temparr = list([val for val in str if val.isalpha() or val.isnumeric()])
    cleanstr = ''.join(temparr)
    return cleanstr


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
    def __init__(self, name, level, path, labeldict=None, xgbmodel=None, finetuning=None, cvmetrics=None,
                 finalmetrics=None, predictions=None, cfm=None, roc=None, pr=None, pickle=None):
        self.name = name
        self.level = level
        self.path = path
        self.labeldict = {}
        self.xgbmodel = None
        self.finetuning = None
        self.cvmetrics = None
        self.finalmetrics = None
        self.predictions = []
        self.cfm = None
        self.roc = []
        self.pr = None
        self.pickle = self.name + '_object.pkl'
        self.predictname = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, layer2):
        return self.name==layer2.name and self.level==layer2.level

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
            if self.cvmetrics is not None:
                return_str += '10-Fold Cross Validation Metrics: ' + self.cvmetrics + '\n'
                return_str += 'Final Metrics: ' + self.finalmetrics + '\n'
                return_str += 'Predictions (no rejection): ' + self.predictions[0] + '\n'
                return_str += 'Predictions (' + str(rejection_cutoff) + ' rejection cutoff): ' + self.predictions[1] + '\n'
                return_str += 'Confusion Matrix (w/ rejection): ' + self.cfm + '\n'
                return_str += 'Micro/Macro ROC Curves: ' + self.roc[0] + '\n'
                return_str += 'Per-Class ROC Curves: ' + self.roc[1] + '\n'
                return_str += 'All Combined ROC Curves: ' + self.roc[2] + '\n'
                return_str += 'Precision-Recall Curves: ' + self.pr + '\n'
        return return_str

    def finetuned(self):
        return self.finetuning is not None

    def trained(self):
        return self.xgbmodel is not None

    def inDict(self, value):
        return value in list(self.labeldict.values())

    # Adds a value to the label dictionary if it is not already present
    # Utility function of fill_dict
    def add_dictentry(self, value):
        if value not in list(self.labeldict.values()):
            self.labeldict[len(self.labeldict)] = value


    # Hyperparameter tuning for XGBoost using accuracy
    # Equivalent to sklearn.model_selection.RandomizedSearchCV -- w/o cv and w/ triangular dist
    # Parameters: eta, max_depth, subsamaple, colsample_bytree
    # n randomized trials: triangular distribution around the default or recommended xgboost value
    #                      rounded to a reasonable decimal place
    # Splits data according to user-provided testplit value, NO cross validation conducted
    # Returns parameters with lowest mean absolute error (mae) on 10% for actual training
    # Outputs csv file with accuracy of each class and mae for all trials, does not output any other metrics
    def finetune(self, trials, testsplit, normexprpkl, metadatacsv):
        # If user skips cross validation, testsplit automatically set to 10% for finetuning
        if testsplit is None:
            testsplit = 0.1
        X, Y, X_tr, X_test, Y_tr, Y_test, _ = self.read_data(normexprpkl, metadatacsv, 0.5)
        min_mae = 100000000000000
        with open(path + self.name + '_finetuning.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            row = 'ETA,Max Depth,Subsample,Colsample by Tree,'
            for i in range(len(self.labeldict)):
                row += self.labeldict[i] + ' Accuracy,'
            row += 'MAE'
            print(row, file=f)
            f.flush()
            for i in range(trials):
                print('Trial', i)
                eta_temp = round(random.triangular(0,1,0.3),1)
                max_depth_temp = round(random.triangular(4,8,6))
                subsample_temp = round(random.triangular(0.01,1,0.5),1)
                colsample_bytree_temp = round(random.triangular(0.01,1,0.5),1)
                params = {'objective': 'multi:softprob', 'eta': eta_temp, 'max_depth': max_depth_temp, 'subsample': subsample_temp,
                        'colsample_bytree': colsample_bytree_temp, 'eval_metric': 'merror', 'seed': 840}
                mae, output_string = self.xgboost_model_shortver(X_tr, X_test, Y_tr, Y_test, params)
                row = str(eta_temp) + ',' + str(max_depth_temp) + ',' + str(subsample_temp) + ',' + str(colsample_bytree_temp) + ','
                row += output_string
                print(row, file=f)
                f.flush()
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
    # Trains XGBoost for a layer of classification given parameters
    # Splits data according to user-provided testsplit
    # Conducts 10-fold cross validation on (1-testsplit)% of data, outputs cv metrics
    # Retrains model on (1-testsplit)% to output metrics when tested on holdout (testsplit)%
    # Retrains final saved model on 100% of data, returns for feature ranking
    # Conducts feature ranking with SHAP if instructed by user on full final model
    def train_layer(self, normexprpkl, metadatacsv, params, testsplit, rejectcutoffs):
        params['num_class'] = len(self.labeldict)

        if testsplit is not None:
            # 10-fold CV on (1-testsplit)% of data
            X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames = self.read_data(normexprpkl, metadatacsv, testsplit)
            kfold = 10
            sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.1, random_state=720)
            for i, (train_index, test_index) in enumerate(sss.split(X_tr, Y_tr)):
                print('[Fold %d/%d]' % (i + 1, kfold))
                X_train, X_valid = X_tr[train_index], X_tr[test_index]
                Y_train, Y_valid = Y_tr[train_index], Y_tr[test_index]
                d_train = xgb.DMatrix(X_train, Y_train, feature_names=feature_names)
                cv_model = xgb.train(params, d_train, 20, verbose_eval=500)
                self.xgbmodel = cv_model # temporarily set xgbmodel to cv 0.9*(1-testsplit)% model
                self.model_metrics('cv', rejectcutoffs[1], X_valid, Y_valid)
                self.cvmetrics = self.name + '_cvmetrics.txt'

            # 90% model, 10% testing
            d_tr = xgb.DMatrix(X_tr, Y_tr, feature_names=feature_names)
            temp_model = xgb.train(params, d_tr, 20, verbose_eval=500)
            self.xgbmodel = temp_model # temporarily set xgbmodel to (1-testsplit)% model
            self.model_metrics('final', rejectcutoffs[1], X_test, Y_test)
            self.finalmetrics = self.name + '_finalmetrics.txt'
            self.cell_predictions('training', rejectcutoffs, test_cellnames, X_test, Y_test)
            for cutoff in rejectcutoffs:
                self.predictions.append(self.name + '_trainingpredictions_reject' + str(cutoff) + '.csv')
            self.cfsn_mtx('training', rejectcutoffs[1], X_test, Y_test)
            self.cfm = self.name + '_confusionmatrix.svg'
            self.roc_curves(X_test, Y_test)
            self.roc.append(self.name + '_miacroroc.svg')
            self.roc.append(self.name + '_classroc.svg')
            self.roc.append(self.name + '_allroc.svg')
            self.pr_curves(X_test, Y_test)
            self.pr = self.name + '_allpr.svg'

        # skip cross validation and metric calculations
        if testsplit is None:
            X, Y, all_cellnames = self.read_data(normexprpkl, metadatacsv)

        # final 100% model
        d_all = xgb.DMatrix(X, Y, feature_names=feature_names)
        final_model = xgb.train(params, d_all, 20, verbose_eval=500)
        pickle.dump(final_model, open(path + self.name + '_xgbmodel.sav', 'wb'))
        self.xgbmodel = final_model


    # Outputs predictions of the 90% model on the 10% test set in a given dataset
    # 1 csv file outputted for each rejection cutoff in the list rejectioncutoffs
    # A cutoff of 0.5 means the probability of the label must be >=0.5 for a prediction to be called
    # A cutoff of 0 is equivalent to no rejection option
    def predict_layer(self, rejectcutoffs, normexprpkl, metadatacsv=None):
        X, Y, all_cellnames = self.read_data(normexprpkl, metadatacsv) # Y will be None if metadatacsv=None
        self.cell_predictions('validation', rejectcutoffs, all_cellnames, X, Y) # can handle Y=None
        if metadatacsv != None:
            self.cfsn_mtx('validation', rejectcutoffs[1], X, Y)
            self.model_metrics('validation', rejectcutoffs[1], X, Y)


    # Outputs SHAP feature ranking plots overall and for each class
    # Outputs csv table containing overall SHAP values
    # Uses same procedure for subsetting norm_expr as read_data(...) if column & criterium are provided
    # frsplit automatically set to 0.3 if not provided
    def featurerank_layer(self, normexprpkl, metadatacsv, frsplit):
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
        temp = labels.loc[labels[labelcolumn].isin(list(self.labeldict.values()))]
        labels = labels.reindex(index=temp.index)
        norm_express = norm_express.reindex(index=temp.index)
        norm_express, labels = shuffle(norm_express, labels)
        if frsplit is None:
            frsplit = 0.3
        if frsplit != 1:
            norm_express = train_test_split(norm_express, labels, test_size=frsplit,
                                            random_state=42, shuffle = True, stratify = labels[labelcolumn])[1]

        model = self.xgbmodel
        model_bytearray = model.save_raw()[4:]
        model.save_raw = lambda: model_bytearray
        shap_values = shap.TreeExplainer(model).shap_values(norm_express)
        shap.summary_plot(shap_values, norm_express, show=False)
        plt.tight_layout()
        plt.savefig(path + self.name + '_overallfr.svg')
        plt.clf()
        for i in range(len(self.labeldict)):
            shap.summary_plot(shap_values[i], norm_express, show=False)
            plt.tight_layout()
            plt.savefig(path + self.name + '_' + alphanumeric(self.labeldict[i]) + 'fr.svg')
            #np.savetxt(path + name + '_class'+str(i)+'fr.csv', shap_values[i], delimiter=",")
            plt.clf()

        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(norm_express.columns, sum(vals))), columns=['Gene','Feature Importance Value'])
        feature_importance.sort_values(by=['Feature Importance Value'], ascending=False, inplace=True)
        feature_importance.to_csv(path + self.name + '_featureimportances.csv', sep=',', encoding='utf-8')

        print('Overall Feature Ranking: ' + self.name + '_overallfr.svg')
        for i in range(len(self.labeldict)):
            print(self.name + '--' + list(self.labeldict.values())[i] + ' Feature Ranking: ' + self.name + '_' + alphanumeric(self.labeldict[i]) + 'fr.svg')
        print('Full Feature Importance List: ' + self.name + '_featureimportances.csv')


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
        # print (labels[labelcolumn].value_counts())
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
        return X, Y, X_tr, X_test, Y_tr, Y_test, test_cellnames


    # Calculates metrics of the Layer model on a provided test set and outputs it in a file
    # cv_final_val is a naming string to differentiate between cv, final, and validation metrics
    def model_metrics(self, cv_final_val, rejectcutoff, X_test, Y_test):
        self.add_dictentry('Unclassified')
        d_test = xgb.DMatrix(X_test, Y_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        for i in range(len(probabilities_xgb)):
            if probabilities_xgb[i,probabilities_xgb.argmax(axis=1)[i]] < rejectcutoff:
                predictions_xgb[i] = len(self.labeldict)-1
        target_names = [str(x) for x in sorted(list(set(Y_test).union(predictions_xgb)))]
        metrics = classification_report(Y_test, predictions_xgb, target_names=target_names)
        with open(path + self.name + '_' + cv_final_val + 'metrics.txt', 'a+') as f:
            print(metrics, file=f)
        del self.labeldict[len(self.labeldict)-1]


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
        self.add_dictentry('Unclassified')
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
    # train_val is a naming string to differentiate between train/test predictions and validation predictions
    def cfsn_mtx(self, train_val, rejectcutoff, X_test, Y_test):
        d_test = xgb.DMatrix(X_test, feature_names=feature_names)
        probabilities_xgb = self.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        self.add_dictentry('Unclassified')
        for i in range(len(probabilities_xgb)):
            if probabilities_xgb[i,probabilities_xgb.argmax(axis=1)[i]] < rejectcutoff:
                predictions_xgb[i] = len(self.labeldict)-1
        cm = confusion_matrix(Y_test, predictions_xgb)
        print(cm)
        del self.labeldict[len(self.labeldict)-1]

        for i in range(len(predictions_xgb)-1, -1, -1):
            if predictions_xgb[i] == len(self.labeldict):
                predictions_xgb = np.delete(predictions_xgb, i)
                Y_test = np.delete(Y_test, i)
        cm_removed = confusion_matrix(Y_test, predictions_xgb)
        print(cm_removed)
        classnames = [self.labeldict[x] for x in sorted(list(set(Y_test).union(predictions_xgb)))]
        print(classnames)
        cm_removed = cm_removed.astype('float') / cm_removed.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(15,10))
        plt.imshow(cm_removed, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classnames))
        plt.xticks(tick_marks, classnames, rotation=45)
        plt.yticks(tick_marks, classnames)

        thresh = cm_removed.max() / 2.
        for i, j in itertools.product(range(cm_removed.shape[0]), range(cm_removed.shape[1])):
            plt.text(j, i, '%0.3f' % cm_removed[i,j] if cm_removed[i,j] > 0 else 0,
                horizontalalignment='center', verticalalignment='center',
                color='white' if cm_removed[i,j] > thresh else 'black', fontsize=6)

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
        y_test = np.zeros((Y_test.size, Y_test.max()+1))
        y_test[np.arange(Y_test.size),Y_test] = 1
        y_score = probabilities_xgb
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

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
        
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='ROC curve of {0} (area = {1:0.2f})'
                     ''.format(self.labeldict[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Class ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(path + self.name + '_classroc.svg')
        plt.clf()

        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='ROC curve of {0} (area = {1:0.2f})'
                     ''.format(self.labeldict[i], roc_auc[i]))
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
        y_test = np.zeros((Y_test.size, Y_test.max()+1))
        y_test[np.arange(Y_test.size),Y_test] = 1
        y_score = probabilities_xgb
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

        plt.figure()
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], color='#%06X' % random.randint(0, 0xFFFFFF), lw=lw,
                     label='PR curve of {0} (area = {1:0.2f})'
                     ''.format(self.labeldict[i], average_precision[i]))
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


# Main function: reads in user input, selects a pathway, and trains / predicts
def main():
    ## DEVCELLPY RUN OPTIONS
    # 1a. training w/ cross validation and metrics
    #       (runMode = trainAll, trainNormExpr, labelInfo, timepointLayer, trainMetadata, testSplit, rejectionCutoff)
    # 1b. training w/o cross validation and metrics
    #       (runMode = trainAll, trainNormExpr, labelInfo, timepointLayer, trainMetadata, rejectionCutoff)
    # 2a. prediction w/ metadata
    #       (runMode = predictOne, predNormExpr, predMetadata, layerObjectPaths, rejectionCutoff)
    # 2b. prediction w/o metadata, each layer's prediction independent of predictions from other layers
    #       (runMode = predictOne, predNormExpr, layerObjectPaths, rejectionCutoff)
    # 3. prediction w/o metadata, each layer's prediction influences next layer's prediction
    #       (runMode = predictAll, predNormExpr, layerObjectPaths, rejectionCutoff)
    # 4.  feature ranking
    #       (runMode = featureRankingOne, trainNormExpr, trainMetadata, layerObjectPaths, featureRankingSplit)
    time_start = time.perf_counter()

    # All variables used for training and prediction set to None
    global path, path_cda, rejection_cutoff, skip, std_params
    path = os.getcwd()
    path_cda = os.getcwd()
    user_train = False
    user_predictOne = False
    user_predictAll = False
    user_fr = False
    train_normexpr = None
    labelinfo = None
    skip = None
    train_metadata = None
    testsplit = None
    rejection_cutoff = None
    std_params = False
    pred_normexpr = None
    pred_metadata = None
    layer_paths = None
    frsplit = None

    ## Command Line Interface
    # runMode must be 'trainAll', 'predictOne', 'predictAll', or 'featureRankingOne'
    # trainNormExpr, labelInfo, trainMetadata are paths to their respective training files
    # testSplit is a float between 0 and 1 denoting the percentage of data to holdout for testing
    #           if not provided, cross validation is skipped, 100% model trained w/o metrics
    # rejectionCutoff is a float between 0 and 1 denoting the minimum probability for a prediction to not be rejected
    # stdParams is a boolean denoting whether or not to finetune or use automatic parameters
    # predNormExpr, predMetadata are paths to their respective prediction files
    # layerObjectPaths is a comma-separated list of paths to the Layer objects that the user wants to predict on the predNormExpr
    # featureRankingSplit is a float between 0 and 1 denoting the percentage of data to calculate SHAP importances
    args = sys.argv[1:]
    options, args = getopt.getopt(args, '',
                        ['runMode=', 'trainNormExpr=', 'labelInfo=', 'timepointLayer=', 'trainMetadata=', 'testSplit=', 'rejectionCutoff=',
                         'stdParams=', 'predNormExpr=', 'predMetadata=', 'layerObjectPaths=', 'featureRankingSplit='])
    for name, value in options:
        if name in ['--runMode']:
            if value == 'trainAll':
                user_train = True
            elif value == 'predictOne':
                user_predictOne = True
            elif value == 'predictAll':
                user_predictAll = True
            elif value == 'featureRankingOne':
                user_fr = True
        if name in ['--trainNormExpr']:
            train_normexpr = value
        if name in ['--labelInfo']:
            labelinfo = value
        if name in ['--timepointLayer']:
            skip = value
        if name in ['--trainMetadata']:
            train_metadata = value
        if name in ['--testSplit']:
            testsplit = float(value)
        if name in ['--rejectionCutoff']:
            rejection_cutoff = float(value)
        if name in ['--stdParams']:
            std_params = True
        if name in ['--predNormExpr']:
            pred_normexpr = value
        if name in ['--predMetadata']:
            pred_metadata = value
        if name in ['--layerObjectPaths']:
            layer_paths = value.split(',')
        if name in ['--featureRankingSplit']:
            frsplit = float(value)

    # Check user provided variables follow an above devcellpy pathway
    passed_options = check_combinations(user_train, user_predictOne, user_predictAll, user_fr, train_normexpr, labelinfo, train_metadata, testsplit,
                                        rejection_cutoff, pred_normexpr, pred_metadata, layer_paths, frsplit)
    if passed_options is False:
        raise ValueError('see printed error log above')

    # Check training files exist if training option called
    passed_train = None
    passed_predict = None
    passed_fr = None
    if user_train is True:
        passed_train = check_trainingfiles(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff)
    # Check prediction files exist if either of the prediction options is called
    if (user_predictOne is True) or (user_predictAll is True):
        passed_predict = check_predictionfiles(pred_normexpr, pred_metadata, layer_paths)
    # Check feature ranking files exist if feature ranking option called
    if user_fr is True:
        passed_fr = check_featurerankingfiles(train_normexpr, train_metadata, layer_paths, frsplit)
    if (passed_train is False) or (passed_predict is False) or (passed_fr is False):
        raise ValueError('see printed error log above')

    # If training option is called and feasible
    if user_train is True and passed_train is True:
        training(train_normexpr, labelinfo, train_metadata, testsplit, rejection_cutoff)
    # If prediction one option is called and feasible
    if user_predictOne is True and passed_predict is True:
        predictionOne(pred_normexpr, pred_metadata, layer_paths)
    # If prediction all option is called and feasible
    if user_predictAll is True and passed_predict is True:
        predictionAll(pred_normexpr, layer_paths)
    # If feature ranking option is called and feasible
    if user_fr is True and passed_fr is True:
        featureranking(train_normexpr, train_metadata, layer_paths, frsplit)

    # Print computational time and memory required
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))


if __name__ == "__main__":
    main()
