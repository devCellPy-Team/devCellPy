from .importing_modules import *
from . import config, helpers
from .layer import Layer

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
    config.path = os.path.join(config.path, 'devcellpy_predictOne_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    config.path += '/'
    os.makedirs(config.path)
    os.chdir(config.path)
    all_layers = helpers.import_layers(object_paths)
    featurenames = all_layers[0].xgbmodel.feature_names
    reorder_pickle(val_normexpr, featurenames)
    val_normexpr = val_normexpr[:val_normexpr.index('.')] + '_reordered.pkl'
    for layer in all_layers:
        path = os.path.join(config.path, layer.path)
        os.makedirs(path)
        os.chdir(path)
        path = path + '/'
        layer.predict_layer([0, config.rejection_cutoff], val_normexpr, val_metadata)
        os.chdir(config.path) # return to prediction directory
        path = os.getcwd()
    print('Prediction Complete')
    os.chdir('..') # return to devcellpy directory


# Conducts prediction in all layers in one folder
# Creates directory 'predictionAll' in devcellpy_results folder, defines 'Root' as topmost layer
def predictionAll(val_normexpr, object_paths):
    config.path = os.path.join(config.path, 'devcellpy_predictAll_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    config.path += '/'
    os.makedirs(config.path)
    os.chdir(config.path)

    all_layers = helpers.import_layers(object_paths)
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

    f = open(config.path + 'predictionall_reject' + str(config.rejection_cutoff) + '.csv','w')
    for i in range(len(all_cellnames)):
        sample = np.array(X[i])#.reshape((-1,1))
        sample = np.vstack((sample, np.zeros(len(feature_names))))
        d_test = xgb.DMatrix(sample, feature_names=feature_names)
        root_layer = find_predictlayer(all_layers, 'Root')
        root_layer.add_dictentry('Unclassified')
        probabilities_xgb = root_layer.xgbmodel.predict(d_test)
        predictions_xgb = probabilities_xgb.argmax(axis=1)
        if probabilities_xgb[0,probabilities_xgb.argmax(axis=1)[0]] < config.rejection_cutoff:
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
                if probabilities_xgb[0,probabilities_xgb.argmax(axis=1)[0]] < config.rejection_cutoff:
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


# Converts the normalized expression csv into a pkl
# Expression CSV file must contain genes as row names, samples as column names
# First column name (cell A1) is 'gene'
# Reorders the csv file to match the features in a given featurenames list
# Returns path to the new pkl file
def reorder_pickle(path, featurenames):
    # Convert data into pickles
    if path[-3:] == 'csv':
        csvpath = path
        tp = pd.read_csv(csvpath, iterator=True, chunksize=1000)
        norm_express = pd.concat(tp, ignore_index=True)
        norm_express.set_index('gene', inplace=True)
        norm_express.index.names = [None]
        norm_express = norm_express.T
        # print (norm_express.head())
        # print(norm_express.T.duplicated().any())
        norm_express.to_pickle(csvpath[:-3] + 'pkl')
    elif path[-4:] == 'h5ad':
        h5adpath = path
        adata = sc.read_h5ad(h5adpath)
        norm_express = pd.DataFrame(adata.X.toarray(), columns = adata.var.index, index = adata.obs.index)
        norm_express.to_pickle(h5adpath[:-4] + 'pkl')
    elif path[-3:] == 'pkl':
        norm_express = pd.read_pickle(path)
    else:
        raise ValueError('Format of normalized expression data file not recognized')
    print ('Training Data # of  genes: ' + str(len(featurenames)))
    origfeat = list(norm_express)
    print ('Validation Data # of genes: ' + str(len(origfeat)))
    print ('Overlapping # of genes: ' + str(len(set(featurenames).intersection(origfeat))))

    ## Manually reorder columns according to training data index
    # Reorder overlapping genes, remove genes not in training data
    norm_express = norm_express.reindex(columns=featurenames, fill_value=None)
    norm_express.to_pickle(path[:-4] + '_reordered.pkl')


# Utility function, searches a list of all_layers for a layer with the given name
def find_predictlayer(all_layers, name):
    for layer in all_layers:
        if layer.predictname == name:
            return layer
    return None