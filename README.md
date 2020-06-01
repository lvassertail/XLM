# Similarity-Based Weighted Cross-Entropy Loss for Unsupervised Neural Machine Translation

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)


## Unsupervised MT

### Download / preprocess data

To download the data required for the unsupervised MT experiments, simply run:

```
git clone https://github.com/facebookresearch/XLM.git
cd XLM
```

And then run the following script:

```
wget https://dl.fbaipublicfiles.com/XLM/codes_enfr
wget https://dl.fbaipublicfiles.com/XLM/vocab_enfr

./get-data-nmt.sh --src en --tgt fr --reload_codes codes_enfr --reload_vocab vocab_enfr
```

The script will successively:
- download Moses scripts, download and compile fastBPE
- download, extract, tokenize, apply BPE to monolingual and parallel test data
- binarize all datasets


`get-data-nmt.sh` contains a few parameters defined at the beginning of the file:
- `N_MONO` number of monolingual sentences for each language (default 5000000)
- `CODES` number of BPE codes (default 60000)
- `N_THREADS` number of threads in data preprocessing (default 16)


The script should output a data summary that contains the location of all files required to start experiments:

```
===== Data summary
Monolingual training data:
    en: ./data/processed/en-fr/train.en.pth
    fr: ./data/processed/en-fr/train.fr.pth
Monolingual validation data:
    en: ./data/processed/en-fr/valid.en.pth
    fr: ./data/processed/en-fr/valid.fr.pth
Monolingual test data:
    en: ./data/processed/en-fr/test.en.pth
    fr: ./data/processed/en-fr/test.fr.pth
Parallel validation data:
    en: ./data/processed/en-fr/valid.en-fr.en.pth
    fr: ./data/processed/en-fr/valid.en-fr.fr.pth
Parallel test data:
    en: ./data/processed/en-fr/test.en-fr.en.pth
    fr: ./data/processed/en-fr/test.en-fr.fr.pth
```

### Train on unsupervised MT from a pretrained model

You can now use the pretrained model for Machine Translation. To download a model trained with the command above on the MLM objective, and the corresponding BPE codes, run:

```
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_enfr_1024.pth
```

If you preprocessed your dataset in `./data/processed/en-fr/` with the provided BPE codes `codes_enfr` and vocabulary `vocab_enfr`, you can pretrain your NMT model with `mlm_enfr_1024.pth` and run:

```
python train.py

## main parameters
--exp_name unsupMT_enfr                                       # experiment name
--dump_path ./dumped/                                         # where to store the experiment
--reload_model 'mlm_enfr_1024.pth,mlm_enfr_1024.pth'          # model to reload for encoder,decoder

## data location / training objective
--data_path ./data/processed/en-fr/                           # data location
--lgs 'en-fr'                                                 # considered languages
--ae_steps 'en,fr'                                            # denoising auto-encoder training steps
--bt_steps 'en-fr-en,fr-en-fr'                                # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1024                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--tokens_per_batch 1000                                       # use batches with a fixed number of words
--batch_size 32                                               # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_en-fr_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_en-fr_mt_bleu'                    # end experiment if stopping criterion does not improve

## training settings for weighted loss
--weighted_loss true \                                        # true - weighted loss approach, false - baseline approach
--weighted_loss_k 5 \                                         # number of top predictions to consider
--weighted_loss_similarity 'l2'                               # similarity function to use. additional option - "cosine"
```


## License

See the [LICENSE](LICENSE) file for more details.

