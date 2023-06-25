from .importing_modules import *
from . import config, helpers
from .layer import Layer

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
    config.path = os.path.join(config.path, 'devcellpy_featureranking_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    config.path += '/'
    os.makedirs(config.path)
    os.chdir(config.path)
    if train_normexpr[-3:] == 'csv':
        helpers.csv2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-3] + 'pkl'
    elif train_normexpr[-4:] == 'h5ad':
        helpers.h5ad2pkl(train_normexpr)
        train_normexpr = train_normexpr[:-4] + 'pkl'
    elif train_normexpr[-3:] != 'pkl':
        raise ValueError('Format of normalized expression data file not recognized')
    all_layers = helpers.import_layers(object_paths)
    for layer in all_layers:
        path = os.path.join(config.path, layer.path)
        os.makedirs(path)
        os.chdir(path)
        path = path + '/'
        layer.featurerank_layer(train_normexpr, train_metadata, frsplit)
        os.chdir(config.path) # return to prediction directory
        path = os.getcwd()
    print('Feature Ranking Complete')