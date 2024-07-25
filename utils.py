try:
    import pandas as pd
    import numpy as np
    import argparse
except Exception:
    print('Missing dependency, try: "pip install -r requirements.txt"')
    exit(0)

def error(txt):
    print(txt)
    exit(0)

def read_file(arg, csv):
    try:
        if csv:
            data = pd.read_csv(str(arg))
        else:
            data = np.load(str(arg), allow_pickle=True)
    except FileNotFoundError:
        error('File not found.')
    except pd.errors.EmptyDataError:
        error('No data.')
    except pd.errors.ParserError:
        error('Parse error.')
    except Exception:
        error('Error read_csv exception.')
    return data

def parse_arguments(ex):
    txt = "training" if ex == 0 else "prediction"
    parser = argparse.ArgumentParser(description="Multilayer Perceptron " + txt)
    parser.add_argument('dataset', help="Data(csv) for our " + txt)
    if ex:
        parser.add_argument('weights', help="Weights compute by the training for our prediction")
    if not ex:
        parser.add_argument('-i', '--iterations', type=int, help="Number of iterations for the gradient descent algorithm", default=100)
        parser.add_argument("-l", '--layer', type=int, nargs="+", help="Define the number and the size of each hidden layers", default=[8, 8])
        parser.add_argument("-s", '--early_stop', action='store_true', help="Enable early stopping")
        parser.add_argument("-S", '--seed', type=int, help="Define the seed for numpy random function", default=0)
        parser.add_argument("-b", '--batch', action='store_true', help="Perform training with mini-batch optimization")
    parser.add_argument('-v', '--verbose', action='store_true', help="Display " + txt + " process")
    args = parser.parse_args()
    if not ex:
        if (args.iterations < 0):
            error("Arg Error: bad iterations (min:0 max:5000)")
    return (args)

class InitData:
    def __init__(self, data, ex):
        self.data = data
        self.data.columns = self.set_columns_name()
        self.data = self.data.drop('ID', axis=1)
        self.data['diagnosis'] = self.data['diagnosis'].replace(['B', 'M'], [0, 1])
        self.error_handling(ex)
        self.X, self.y = self.set_xy(self.data)

    def split_train_test(self, seed, prc=0.8):
        if seed:
            train = self.data.sample(frac = prc, random_state=seed)
        else:
            train = self.data.sample(frac = prc)
        test = self.data.drop(train.index)
        X_train, y_train = self.set_xy(train)
        X_test, y_test = self.set_xy(test)
        y_train = y_train.reshape((1, y_train.shape[0]))
        y_test = y_test.reshape((1, y_test.shape[0]))
        return X_train.T, y_train, X_test.T, y_test

    def minMaxScaler(self, feature):
        f_min = min(feature)
        f_max = max(feature)
        for i in range(0, len(feature)):
            feature[i] = (feature[i] - f_min) / (f_max - f_min)

    def set_xy(self, df):
        x = df.drop('diagnosis', axis=1)
        np.apply_along_axis(self.minMaxScaler, 0, x)
        X = np.array(x.fillna(0.5))
        y = np.array(df['diagnosis'])
        return X, y

    def set_columns_name(self):
        columns = ['ID', 'diagnosis']
        types = ['mean_', 'stde_', 'worst_']
        for type in types:
            columns.append(type + 'radius')
            columns.append(type + 'texture')
            columns.append(type + 'perimeter')
            columns.append(type + 'area')
            columns.append(type + 'smoothness')
            columns.append(type + 'compactness')
            columns.append(type + 'concavity')
            columns.append(type + 'concave_pts')
            columns.append(type + 'symmetry')
            columns.append(type + 'fractal_dim')
        return columns

    def error_handling(self, ex):
        if not 'diagnosis' in self.data and not ex:
            error('Missing "diagnosis" column')
        if self.data['diagnosis'].count() < 2 and not ex:
            error('Not enough diagnosis')
        if len(self.data.columns) < 2:
            error('Not enough columns in dataset')