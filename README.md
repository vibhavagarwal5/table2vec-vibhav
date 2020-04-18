# 2D CNN approach for Table2Vec

-   To prepare the data

    -   Run **`python preprocess.py`** to generate the positive and negative sample data.
    -   Aditionally, run **`python trec.py -d`** flag to prepare the tokenized version of the annotated table and query dataset.
    -   Run **`python dataset.py -p`** and **`python dataset.py -p`** to pregenerate the padded postive/negative dataset.

-   Run **`python index.py`** with the following flags:

    -   `--comment=<insert comment>` to add comments.
    -   `--model_type=<type>` for the model type. Currently, there's convE inspired model and Computer Vision inspired model.
    -   `--cuda_no=<GPU NO>` to choose the gpu node.
    -   `--config=./config_no_conv.toml` **(Optional)** To add additional/overriding configuration file.

-   Run **`tensorboard --logdir outputs/`** for the tensorboard.
