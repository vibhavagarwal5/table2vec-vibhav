# 2D CNN approach for Table2Vec

- Run **`python index.py`** with flags **`--comment='insert comment'`** to add comments.
- Run **`tensorboard --logdir outputs/`** for the tensorboard. Select the last 2 runs in to display the latest results.
- Incase the model runs but you forget to get the trec ndcg score or the model fails to do it, then run **`python trec --path=<output path>`** and then **`python TREC_score --path=<output path>`** to compute the rest. Aditionally there is **`--data_prep`** flag to prepare the tokenized version of the table and query.
