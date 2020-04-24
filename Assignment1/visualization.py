import matplotlib.pyplot as plt
import torch
import numpy as np

def rank_graph(data, fake_data):
    '''
    creates visualizations for correlation between increasing rank and UUAS scores
    '''
    rankings = [1, 2, 4, 8, 16, 32, 64, 128]
    plt.style.use('ggplot')
    colors = ['b', 'r', 'g']
    markers = ['o', '^', 's']
    labels = ['GPT-2-base', 'XLNet-base', 'LSTM (Gulordava et al, 2018)']

    for i in range(len(data)):
        plt.plot(np.arange(len(rankings)), data[i]*100, color=colors[i], marker=markers[i], label=labels[i])
        plt.plot(np.arange(len(rankings)), fake_data[i]*100, color=colors[i], marker=markers[i], linestyle='dashed', label=labels[i]+'-selective')

    plt.xticks(np.arange(len(rankings)), rankings)
    plt.xlabel('Maximum Probing Rank')
    plt.ylabel('UUAS')
    plt.legend(loc='best', fontsize='large')

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.show()


if __name__ == '__main__':
    data = []
    fake_data = []

    data_scores = ['GPT2-scores.pt', 
                   'XLNet-scores.pt',
                   'RNN-scores.pt',
                   #'GPT2-medium-scores.pt',
                   #'GPT2-distil-scores.pt'
                   ]
    data_fake_scores = ['GPT2-fake-scores.pt',
                        'XLNet-fake-scores.pt',
                        'RNN-fake-scores.pt',
                        #'GPT2-medium-fake-scores.pt',
                        #'GPT2-distil-fake-scores.pt'
                        ]

    for i in range(len(data_scores)):
        score = np.array(torch.load(data_scores[i]))
        fake = score - np.array(torch.load(data_fake_scores[i]))
        data.append(score)
        fake_data.append(fake)

    rank_graph(data, fake_data)