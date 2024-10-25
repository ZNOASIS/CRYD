import torch
from tqdm import tqdm
import numpy as np
import pickle
import torch.nn.functional as F
from skopt import Optimizer

def load_results(model_files):
    results = []
    for file in model_files:
        with open(file, 'rb') as f:
            results.append(list(pickle.load(f).items()))
    return results

def objective(alpha, results, labels):
    num_samples = len(results[0])
    num_classes = 155
    ensemble_scores = torch.zeros((num_samples, num_classes), dtype=torch.float32, device=labels.device)

    for i in range(num_samples):
        weighted_sum = torch.zeros(num_classes, dtype=torch.float32, device=labels.device)

        for j in range(len(results)):
            _, r_j = results[j][i]
            r_j_tensor = torch.FloatTensor(r_j).to(labels.device)
            weighted_sum += r_j_tensor * alpha[j]

        ensemble_scores[i] = weighted_sum

    pred = F.softmax(ensemble_scores, dim=1)
    preds = pred.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total * 100

    return -accuracy


def tqdm_gp_minimize(func, space, n_calls, *args, **kwargs):

    opt = Optimizer(space)
    for i in tqdm(range(n_calls)):
        x = opt.ask()
        y = func(x, *args, **kwargs)
        opt.tell(x, y)


    best_x = opt.Xi[np.argmin(opt.yi)]
    best_fun = min(opt.yi)

    return best_x, best_fun

model_files = [
    'CRYD-main/preds/epoch1_test_score.pkl',
    'CRYD-main/preds/epoch1_test_score (1).pkl',
    'CRYD-main/preds/epoch1_test_score (2).pkl',
    'CRYD-main/preds/epoch1_test_score (3).pkl',
    'CRYD-main/preds/epoch1_test_score (4).pkl',
    'CRYD-main/preds/epoch1_test_score (5).pkl',
    'CRYD-main/preds/epoch1_test_score (6).pkl',
    'CRYD-main/preds/epoch1_test_score (7).pkl',
    'CRYD-main/preds/epoch1_test_score (8).pkl',
    'CRYD-main/preds/epoch1_test_score (9).pkl',
    'CRYD-main/preds/epoch1_test_score (10).pkl',
    'CRYD-main/preds/epoch1_test_score (11).pkl',
    'CRYD-main/preds/epoch1_test_score (12).pkl',
    'CRYD-main/preds/epoch1_test_score (13).pkl',
    'CRYD-main/preds/epoch1_test_score (14).pkl',
    'CRYD-main/SkateFormer-main/test_work_dir/bA/epoch1_test_score.pkl',
    'CRYD-main/SkateFormer-main/test_work_dir/jmA/epoch1_test_score.pkl',
    'CRYD-main/SkateFormer-main/test_work_dir/bmA/epoch1_test_score.pkl'
]

results = load_results(model_files)

labels = np.load('CRYD-main/data/test_label_A.npy')
for i in range(len(labels)):
    if labels[i] == -1:
        labels[i] = 0
labels = torch.LongTensor(labels)


space = [(0.0, 20.0) for _ in range(len(model_files))]


best_alpha, best_accuracy = tqdm_gp_minimize(
    lambda alpha: objective(torch.FloatTensor(alpha), results, labels),
    space,
    n_calls=600
)


print(f'Best Alpha: {best_alpha}')
print(f'Best accuracy: {-best_accuracy}')