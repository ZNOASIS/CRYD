import argparse
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='pred.npy')

if __name__ == "__main__":

    args = parser.parse_args()

    # load label and pred
    label =np.load('autodl-tmp/competition/data/test_A_label.npy')
    with open(args.pred_path,'rb') as file:
        data = pickle.load(file)

    nump = list(data.values())
    nump = np.array(nump)
    pred = nump.argmax(axis=1)

    correct = (pred == label).sum()

    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))
    np.save('pred.npy',nump)