# 2D CNN approach for Table2Vec

<!-- -   To prepare the data

    -   Run **`python preprocess.py`** to generate the positive and negative sample data.
    -   Aditionally, run **`python trec.py -d`** flag to prepare the tokenized version of the annotated table and query dataset.
    -   Run **`python dataset.py -p`** and **`python dataset.py -p`** to pregenerate the padded postive/negative dataset. -->

-   Run **`python training.py`** with the following flags:

    -   `--comment=<insert comment>` to add comments.
    -   `--model_type=<type>` for the model type. Currently, there's convE inspired model and Computer Vision inspired model.
    -   `--gpu=<GPU NO>` to choose the gpu rank. **Skip** when running distributed training.
    -   `--config=./config_no_conv.toml` **(Optional)** To add additional/overriding configuration file.
    -   `--distributed` to run the distributed training.

eg:

-   for distributed training

    -   `python -m torch.distributed.launch --nproc_per_node=4 training.py -m='cv_insp' --comment='testing all data with new random table gen with/wo distr-gpu' --distributed`

-   for training on single GPU:2

    -   `python training.py -g=2 -m='cv_insp' --comment='testing all data with new random table gen with/wo distr-gpu'`

-   Run **`tensorboard --logdir outputs/`** for the tensorboard.

<!-- Note
for data files:
x_tokenized_preprocessed_.pkl -> with the new fillup and w2i -> creates full table with no [] and token are IDs. [] are filled with <PAD>
x_tokenized_preprocessed_pad_unk -> full padded and w2i x_tokenized_preprocessed_.pkl
x_tokenized_preprocessed_qfix.pkl -> x_tokenized_preprocessed_qfix.pkl but [] are filled with <UNK>
x_tokenized_preprocessed_qfix_pad_unk -> full padded and w2i x_tokenized_preprocessed_qfix.pkl
vocab_5-15.pkl -> full vocab for that x_tokenized_preprocessed.pkl
vocab_5-15_unk.pkl -> partial vocab (will cause unk tokens when w2i)
 -->
 
 # Data and model files
 
 You can find the data and model dumps to replicate our results through [this](https://drive.google.com/drive/folders/1UxtV_XOrrm31YcoXZYxj9jWpAZzbxLNK?usp=sharing) Google drive link.
