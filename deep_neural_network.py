try:
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.metrics import confusion_matrix
    from utils import *
except Exception:
    print('Missing dependency, try: "pip install -r requirements.txt"')
    exit(0)

class multilayer_perceptron:
    def __init__(self, X, y, args, Xt=None, yt=None):
        if type(args.layer) == int:
            self.network_layers = [args.layer]
        else:
            self.network_layers = list(args.layer)
        self.network_layers.insert(0, X.shape[0])
        self.network_layers.append(y.shape[0])
        self.best_epoch = 0
        self.best_loss = 10
        if args.verbose: print(self.network_layers)
        self.l_size = len(self.network_layers)
        if Xt is not None and yt is not None:
            self.params = self.init_params(self.network_layers, args.seed)
            self.nesterov = self.init_params(self.network_layers, args.seed, zero=True)
            self.training_history = self.gradient_descent(X, y, Xt, yt, args)
        else:
            self.params = args.params

    def init_params(self, network_layers, seed, zero=False):
        ret = {}
        for l in range(1, self.l_size):
            if not zero:
                if seed:
                    np.random.seed(seed)
                ret['W' + str(l)] = np.random.randn(network_layers[l], network_layers[l - 1])
                if seed:
                    np.random.seed(seed)
                ret['b' + str(l)] = np.random.randn(network_layers[l], 1)
            else:
                ret['mW' + str(l)] = np.zeros((network_layers[l], network_layers[l - 1]))
                ret['mb' + str(l)] = np.zeros((network_layers[l], 1))
        return ret

    def forward_propagation(self, X):
        activations = {'A0': X}
        for l in range(1, self.l_size):
            Z = self.params['W' + str(l)].dot(activations['A' + str(l - 1)]) + self.params['b' + str(l)]
            activations['A' + str(l)] = 1 / (1 + np.exp(-Z))
        return activations

    def back_propagation(self, y, activations):
        m = y.shape[1]
        dZ = activations['A' + str(self.l_size - 1)] - y
        gradients = {}
        for l in reversed(range(1, self.l_size)):
            gradients['dW' + str(l)] = 1/m * np.dot(dZ, activations['A' + str(l - 1)].T)
            gradients['db' + str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
              dZ = np.dot(self.params['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] * (1 - activations['A' + str(l - 1)])
        return gradients

    def update(self, gradients, l_rate, momentum):
        for l in range(1, self.l_size):
            self.nesterov['mW' + str(l)] = (momentum * self.nesterov['mW' + str(l)]) + l_rate * gradients['dW' + str(l)]
            self.nesterov['mb' + str(l)] = (momentum * self.nesterov['mb' + str(l)]) + l_rate * gradients['db' + str(l)]
            self.params['W' + str(l)] = self.params['W' + str(l)] - self.nesterov['mW' + str(l)]
            self.params['b' + str(l)] = self.params['b' + str(l)] - self.nesterov['mb' + str(l)]

    def predict(self, X):
        activations = self.forward_propagation(X)
        Af = activations['A' + str(self.l_size - 1)]
        return Af >= 0.5, Af

    def create_mini_batches(self, state, X, y, batch_size):
        mini_batches = []
        if state:
            n_minibatches = y.shape[1] // batch_size
            i = 0
            for i in range(n_minibatches + 1):
                X_mini = X[:, i * batch_size:(i + 1)*batch_size]
                Y_mini = y[:, i * batch_size:(i + 1)*batch_size]
                mini_batches.append((X_mini, Y_mini))
            return mini_batches
        else:
            mini_batches.append((X, y))
            return mini_batches

    def gradient_descent(self, X, y, Xt, yt, args):
        n_iter = args.iterations
        training_history = np.zeros((int(n_iter), 5))
        momentum = 0.9
        l_rate = 1 - momentum
        epoch_progress = (tqdm(total=n_iter, desc='Loading', position=0))

        for i in range(n_iter):
            mini_batches = self.create_mini_batches(args.batch, X, y, 32)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                activations = self.forward_propagation(X_mini)
                gradients = self.back_propagation(y_mini, activations)
                self.update(gradients, l_rate, momentum)

            self.save_training_history(X, Xt, y, yt, training_history, i, n_iter, args.verbose) 
            if args.early_stop and i > self.best_epoch + (10 if args.batch else 80):
                tqdm.write(f"epoch {i + 1}/{n_iter} - loss {round(training_history[i, 1], 4)} - val_loss {round(training_history[i, 0], 4)} - acc {round(training_history[i, 3], 4)}")
                break
            epoch_progress.update(1)

        if args.verbose:
            self.display_training(training_history, Xt, yt, X, y)

        return training_history

    def save_training_history(self, X, Xt, y, yt, training_history, i, n_iter, verbose):
        activations = self.forward_propagation(Xt)
        Af = activations['A' + str(self.l_size - 1)]

        training_history[i, 0] = (log_loss(yt.flatten(), Af.flatten()))
        if training_history[i, 0] < self.best_loss:
            self.best_loss = training_history[i, 0]
            self.best_epoch = i

        activations = self.forward_propagation(X)
        Af = activations['A' + str(self.l_size - 1)]

        training_history[i, 1] = (log_loss(y.flatten(), Af.flatten()))
        y_pred, p = self.predict(X)
        training_history[i, 2] = (accuracy_score(y.flatten(), y_pred.flatten()))
        y_pred, p = self.predict(Xt)
        training_history[i, 3] = (accuracy_score(yt.flatten(), y_pred.flatten()))
        if verbose:
            if not i % int((n_iter / 20)) or n_iter <= 20:
                tqdm.write(f"epoch {i + 1}/{n_iter} - loss {round(training_history[i, 1], 4)} - val_loss {round(training_history[i, 0], 4)} - acc {round(training_history[i, 3], 4)}")
        if i + 1 == n_iter:
            tqdm.write(f"epoch {i + 1}/{n_iter} - loss {round(training_history[i, 1], 4)} - val_loss {round(training_history[i, 0], 4)} - acc {round(training_history[i, 3], 4)}")

    def confusion_matrix(self, nb, title, X, y):
        plt.subplot(2, 2, nb)
        plt.title(title)
        y_pred, p = self.predict(X)
        cm = confusion_matrix(y.flatten(), y_pred.flatten())
        plt.imshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='x-large')
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.xticks([1, 0])
        plt.yticks([0, 1])

    def curves(self, nb, title, d1, d2, l1, l2, ly):
        plt.subplot(2, 2, nb)
        plt.title(title)
        plt.plot(d1, label=l1)
        plt.plot(d2, label=l2)
        plt.legend()
        plt.xlabel(xlabel='Epoch')
        plt.ylabel(ylabel=ly)
        plt.axvline(x = self.best_epoch, color = 'r', label='best epoch')

    def display_training(self, training_history, Xt, yt, X, y):
        fig = plt.figure(figsize=(11, 8))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
        self.curves(1, 'Loss - Mean Squared Error', training_history[:, 1], training_history[:, 0], 'train loss', 'test loss', 'Loss')
        self.curves(2, 'Learning Curve', training_history[:, 2], training_history[:, 3], 'train acc', 'test acc', 'Accuracy')
        self.confusion_matrix(3, 'Confusion Matrix - Train', X, y)
        self.confusion_matrix(4, 'Confusion Matrix - Test', Xt, yt)
        plt.show()