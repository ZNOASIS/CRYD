import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--output-dir',
                        help='Directory to save the fused scores in a pickle file')

    arg = parser.parse_args()
    label = np.load('autodl-tmp/competition/data/fake_B_label.npy')

    fused_scores = {}

    with open('autodl-tmp/competition/HDBN/joint/epoch1_test_score.pkl', 'rb') as r1:
        r1 = list(pickle.load(r1).items())
    with open('autodl-tmp/competition/HDBN/bone/epoch1_test_score.pkl', 'rb') as r2:
        r2 = list(pickle.load(r2).items())
    with open('autodl-tmp/competition/HDBN/joint_motion/epoch1_test_score.pkl', 'rb') as r3:
        r3 = list(pickle.load(r3).items())
    with open('autodl-tmp/competition/HDBN/bone_motion/epoch1_test_score.pkl', 'rb') as r4:
        r4 = list(pickle.load(r4).items())
    with open('autodl-tmp/competition/HDBN/k2/epoch1_test_score.pkl', 'rb') as r5:
        r5 = list(pickle.load(r5).items())
    with open('autodl-tmp/competition/HDBN/k2_motioin/epoch1_test_score.pkl', 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    right_num = total_num = right_num_5 = 0
    fused_scores = {}


    #arg.alpha = [2.4, 1.1, 0.4, 0.8]
    #arg.alpha = [2.2, 1.0, 0.3, 0]
    arg.alpha = [1.2, 0.8, 0.4, 0.2, 0.6, 0]
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]

        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4] + r66 * arg.alpha[5]
        fused_scores[i] = r  # Append the fused score
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # Save the fused scores to a pickle file
    if arg.output_dir is not None:
        output_file = os.path.join(arg.output_dir, 'fused_scores.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(fused_scores, f)
        print(f'Fused scores saved to {output_file}')
