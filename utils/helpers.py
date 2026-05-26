import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def similarity_score(output1, output2):
    distance = F.pairwise_distance(output1, output2)
    score = 1 / (1 + distance)  # higher = more similar
    return score.item()


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
