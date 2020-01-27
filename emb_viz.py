import argparse
import os
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import umap
import matplotlib.pyplot as plt
import torch

from preprocess import loadpkl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="path for the scores")
    return parser.parse_args()


def dim_red_plot(plt_type, emb, vocab, output_dir, n_components=2, random_state=42):
    print(f"-- Start {plt_type} --")
    if plt_type == 'tsne':
        new_values = TSNE(
            n_components=n_components,
            random_state=random_state,
            n_jobs=10,
            verbose=2
        ).fit_transform(emb)
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

    elif plt_type == 'umap':
        new_values = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        ).fit_transform(emb)
        x, y = new_values[:, 0], new_values[:, 1]

    print("-- Start ploting --")
    plt.figure(figsize=(16, 16))
    plt.scatter(x, y)
    # for i in range(len(x)):
    #     plt.annotate(vocab[i], xy=(x[i], y[i]), xytext=(
    #         5, 2), textcoords="offset points", ha="right", va="bottom")
    plt.savefig(os.path.join(output_dir, f'viz/emb_{plt_type}.png'))


if __name__ == "__main__":
    args = get_args()
    output_dir = f'./output/{args.path}'
    model = torch.load(os.path.join(output_dir, 'model.pt'))

    vocab = loadpkl('./data/vocab_2D_10-50_complete.pkl')
    emb = model['embeddings.weight'].cpu().data.numpy()
    dim_red_plot('umap', emb, vocab, output_dir, 3)
