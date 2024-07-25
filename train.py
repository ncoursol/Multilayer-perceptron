try:
    import numpy as np
    from utils import *
    from deep_neural_network import multilayer_perceptron
except Exception:
    print('Missing dependency, try: "pip install -r requirements.txt"')
    exit(0)

def save_results(result, layer):
    results = {}
    results['layers'] = layer
    results['weights'] = result.params
    outfile = 'weights.npy'
    try:
        np.save(outfile, [results])
    except Exception:
        print("Can't write to {}.".format(outfile))
        exit(0)

if __name__ == "__main__":
    args = parse_arguments(0)
    dataset = read_file(args.dataset, 1)
    datas = InitData(dataset, 0)

    X_train, y_train, X_test, y_test = datas.split_train_test(args.seed, prc=0.8)

    result = multilayer_perceptron(X_train, y_train, args, Xt=X_test, yt=y_test)

    save_results(result, args.layer)