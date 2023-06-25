from .config import *

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


# Remove non-alphanumeric characters from string
def alphanumeric(str):
    temparr = list([val for val in str if val.isalpha() or val.isnumeric()])
    cleanstr = ''.join(temparr)
    return cleanstr


# Imports Layer objects from a list of given paths
def import_layers(layer_paths):
    layers = []
    for layer_path in layer_paths:
         layer = pd.read_pickle(layer_path)
         layers.append(layer)
    return layers