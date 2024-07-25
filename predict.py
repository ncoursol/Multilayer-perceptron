try:
    import numpy as np
    import sys
    from utils import *
    from deep_neural_network import multilayer_perceptron
except Exception:
    print('Missing dependency, try: "pip install -r requirements.txt"')
    exit(0)

def save_results(results):
    outfile = 'results.csv'
    try:
        pd.DataFrame(results).to_csv(outfile)
    except Exception:
        print("Can't write to {}.".format(outfile))
        exit(0)

if __name__ == "__main__":
    arguments = parse_arguments(1)
    dataset = read_file(arguments.dataset, 1)
    training_params = read_file(arguments.weights, 0)
    datas = InitData(dataset, 1)

    arguments.layer = training_params[0]['layers']
    arguments.params = training_params[0]['weights']

    mult_p = multilayer_perceptron(datas.X.T, datas.y, arguments)

    y, p = mult_p.predict(datas.X.T)
    acc = [1 - i if i < 0.5 else i for i in p[0]]
    results = pd.DataFrame(list(zip(y[0], np.round(acc, 4))), columns=['Diagnosis', 'Accuracy'])
    results = results.replace([False, True], ['B', 'M'])
    results.index.names = ['Index']

    sum = 0
    for i in range(len(y)):
        sum += y[0][i] * np.log(p[0][i]) + (1 - y[0][i]) * np.log(1 - p[0][i])

    print(f'Binary cross-entropy error function : {np.round(-(1 / len(y)) * sum, 6)}')
    print(f'Mean accuracy : {np.round(np.mean(acc), 4)}')
    save_results(results)
    print('Results saved in results.csv')