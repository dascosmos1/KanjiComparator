import torch.nn.functional as F


def similarity_score(output1, output2):
    distance = F.pairwise_distance(output1, output2)
    score = 1 / (1 + distance)  # higher = more similar
    return score.item()
