#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .importing_modules import *
from . import config, train, predict, featurerank
from .layer import Layer

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
                config.user_train = True
            elif value == 'predictOne':
                config.user_predictOne = True
            elif value == 'predictAll':
                config.user_predictAll = True
            elif value == 'featureRankingOne':
                config.user_fr = True
            else:
                raise ValueError('Run mode option not available')
        if name in ['--trainNormExpr']:
            config.train_normexpr = value
        if name in ['--labelInfo']:
            config.labelinfo = value
        if name in ['--timepointLayer']:
            config.skip = value
        if name in ['--trainMetadata']:
            config.train_metadata = value
        if name in ['--testSplit']:
            config.testsplit = float(value)
        if name in ['--rejectionCutoff']:
            config.rejection_cutoff = float(value)
        if name in ['--stdParams']:
            config.std_params = True
        if name in ['--predNormExpr']:
            config.pred_normexpr = value
        if name in ['--predMetadata']:
            config.pred_metadata = value
        if name in ['--layerObjectPaths']:
            config.layer_paths = value.split(',')
        if name in ['--featureRankingSplit']:
            config.frsplit = float(value)

    # Check user provided variables follow an above devcellpy pathway
    passed_options = check_combinations(config.user_train, config.user_predictOne, config.user_predictAll, config.user_fr,
                                        config.train_normexpr, config.labelinfo, config.train_metadata,
                                        config.testsplit, config.rejection_cutoff, config.pred_normexpr,
                                        config.pred_metadata, config.layer_paths, config.frsplit)
    if passed_options is False:
        raise ValueError('see printed error log above')

    # Check training files exist if training option called
    passed_train = None
    passed_predict = None
    passed_fr = None
    if config.user_train is True:
        passed_train = train.check_trainingfiles(config.train_normexpr, config.labelinfo, config.train_metadata, 
                                                 config.testsplit, config.rejection_cutoff)
    # Check prediction files exist if either of the prediction options is called
    if (config.user_predictOne is True) or (config.user_predictAll is True):
        passed_predict = predict.check_predictionfiles(config.pred_normexpr, config.pred_metadata, config.layer_paths)
    # Check feature ranking files exist if feature ranking option called
    if config.user_fr is True:
        passed_fr = featurerank.check_featurerankingfiles(config.train_normexpr, config.train_metadata,
                                                          config.layer_paths, config.frsplit)
    if (passed_train is False) or (passed_predict is False) or (passed_fr is False):
        raise ValueError('see printed error log above')

    # If training option is called and feasible
    time_start = time.perf_counter()
    if config.user_train is True and passed_train is True:
        train.training(config.train_normexpr, config.labelinfo, config.train_metadata, config.testsplit, config.rejection_cutoff)
    # If prediction one option is called and feasible
    if config.user_predictOne is True and passed_predict is True:
        predict.predictionOne(config.pred_normexpr, config.pred_metadata, config.layer_paths)
    # If prediction all option is called and feasible
    if config.user_predictAll is True and passed_predict is True:
        predict.predictionAll(config.pred_normexpr, config.layer_paths)
    # If feature ranking option is called and feasible
    if config.user_fr is True and passed_fr is True:
        featurerank.featureranking(config.train_normexpr, config.train_metadata, config.layer_paths, config.frsplit)

    # Print computational time and memory required
    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))


if __name__ == "__main__":
    main()
