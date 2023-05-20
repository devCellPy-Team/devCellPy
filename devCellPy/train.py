from .importing_modules import *
from . import config, helpers
from .layer import Layer

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
    config.path = os.path.join(config.path, 'devcellpy_training_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
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
    all_layers = [Layer('Root', 0, 'Root')]
    construct_tree(labelinfo, all_layers)
    print(all_layers)
    for layer in all_layers:
        path = os.path.join(config.path, layer.path)
        os.makedirs(path)
        os.chdir(path)
        path = path + '/'
        if config.skip == None or (config.skip != None and layer.name != config.skip):
            if config.std_params is False:
                parameters = layer.finetune(50, 30, train_normexpr, train_metadata)
            else:
                parameters = {'objective': 'multi:softprob', 'eta': 0.2, 'max_depth': 6, 'subsample': 0.5,
                              'colsample_bytree': 0.5, 'eval_metric': 'merror', 'seed': 840}
            print(parameters)
            layer.train_layer(train_normexpr, train_metadata, parameters, testsplit, [0, rejection_cutoff])
            export_layer(layer, all_layers)
        os.chdir(config.path) # return to training directory
        path = os.getcwd()
    training_summary(all_layers)
    print('Training Complete')
    os.chdir('..') # return to devcellpy directory


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
                        all_layers.append(Layer(labeldata[i][j-1], j, 'Root/' + helpers.alphanumeric(labeldata[i][j-1])))
                    else:
                        prev_layer = find_layer(all_layers, labeldata[i][j-2])
                        all_layers.append(Layer(labeldata[i][j-1], j, prev_layer.path + '/' + helpers.alphanumeric(labeldata[i][j-1])))
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
    if config.skip != None:
        tp_layer = find_layer(all_layers, config.skip)
    if layer is not None and layer.trained() is True:
        with open(config.path + helpers.alphanumeric(layer.name) + '_object.pkl', 'wb') as output:
            if config.skip != None and tp_layer is not None and tp_layer.inDict(layer.name):
                layer.predictname = tp_layer.name 
            pickle.dump(layer, output, pickle.HIGHEST_PROTOCOL)


# Summarizes and lists the path names of all the files created during training
# Prints the summary of each Layer object
def training_summary(all_layers):
    f = open(config.path + 'training_summaryfile.txt', 'w')
    for layer in all_layers:
        f.write(str(layer))
        f.write('\n')