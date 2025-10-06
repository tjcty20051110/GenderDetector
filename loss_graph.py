import numpy as np
from matplotlib import pyplot as plt

#参数类型为列表，包含训练和验证的损失值和精度
def plot_plot(train_loss, val_loss, train_acc, val_acc):
    x_arr = np.arange(len(train_loss)) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, train_loss, '-o', label='Train loss')
    ax.plot(x_arr, val_loss, '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, train_acc, '-o', label='Train acc.')
    ax.plot(x_arr, val_acc, '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()