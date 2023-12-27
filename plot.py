import matplotlib.pyplot as plt

def plot_loss(name, loss_history):
    plt.figure(figsize=(10,4))
    dct = loss_history[name]
    line = []
    for key in dct:
        if key == 1 or key == '1':
            line.append(dct[key][0])
            line.append(dct[key][1])
        line.append(sum(dct[key])/len(dct[key]))
    plt.plot(line, label=name, marker='*')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel(f'{name} loss')
    plt.title('Learning curve')
    plt.show()

def plot_metric(history, metric='NDCG'):
    plt.figure(figsize=(10,4))
    color = dict(zip(history.keys(), ['r', 'g', 'b', 'orange']))
    for name in history:
        line = []
        for epoch in list(history[name].keys()):
            line.append(history[name][epoch][0])

        plt.plot(line, c=color[name], marker='o', label=name)
    plt.legend()
    plt.grid()
    plt.xlabel("EPOCH")
    plt.ylabel(metric)
    plt.title("Model evaluation")
    plt.show()