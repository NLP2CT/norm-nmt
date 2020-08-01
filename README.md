# Norm-Based Curriculum Learning for Neural Machine Translation (ACL 2020)
This repo contains the source code and instructions to reproduce the results of our <a href="https://github.com/lkfo415579/norm-nmt/blob/master/NBCL4NMT.pdf">paper</a>.
### Reference:
```bibtex
@inproceedings{NORMCL20,
  title={Norm-Based Curriculum Learning for Neural Machine Translation},
  author={Liu, Xuebo and Lai, Houtim and Wong, Derek F. and Chao, Lidia S.},
  booktitle={ACL 2020},
  year={2020}
}
```
## INSTALLATION

### Requirements
```
* [Boost] 1.64.0
* [CMAKE] 3.13.2
* [CUDA] 8.0
* [Fasttext] https://github.com/facebookresearch/fastText
* Please review Marian for more installation details: https://marian-nmt.github.io/
```

### Clone this repository
```
git clone https://github.com/NLP2CT/norm-nmt
cd norm-nmt
mkdir build
cd build
cmake ..
make -j
```

## Training and Testing

### Train Fasttext to get norm-based sentence difficulty
```
# Install Fasttext
mkdir ~/fast && cd ~/fast
wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
unzip v0.9.1.zip
cd fastText-0.9.1
make -j
cp fasttext ../
# Use tokenized data for fasttext
mkdir -p cl/mod
cat $TRAIN.$SRCL | $MARIAN_VOCAB > cl/vocab.$SRCL.yml
python CL_tools/process_fasttext.py -i $TRAIN.$SRCL -o $SRCL.emb -v $cl/vocab.$SRCL.yml -w ~/fast/fasttext
python CL_tools/build_cdf_mod.py --emb_vector $SRCL.emb.orig.vec $TRAIN.$SRCL cl/mod/$SRCL-mod
```

### Translation
```
# Training (8 GPUs)
 $MARIAN_TRAIN \
        --model $MODEL_DIR/model_revo.npz --type transformer \
        --train-sets $TRAIN.$SRCL $TRAIN.$TGTL \
        --max-length 140 \
        --vocabs $MODEL_DIR/vocab.$SRCL.yml $MODEL_DIR/vocab.$TGTL.yml \
        --mini-batch-fit -w 9250 --maxi-batch 5000 \
        --early-stopping 10 --cost-type=ce-mean-words \
        --valid-freq 2500 --save-freq 2500 --disp-freq 1 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $CORPUS_DIR/$VALID.$SRCL $CORPUS_DIR/$VALID.$TGTL \
        --valid-script-path "bash ./validate-"$SRCL\-$TGTL".sh" \
        --valid-translation-output $OUTPUT_DIR/$ID.tf.$SRCL$TGTL.single --quiet-translation \
        --valid-mini-batch 64 \
        --beam-size 6 --normalize 0.6 \
        --log $MODEL_DIR/train.log --valid-log $MODEL_DIR/valid.log \
        --enc-depth 6 --dec-depth 6 \
        --transformer-heads 8 \
        --transformer-postprocess-emb d \
        --transformer-postprocess dan \
        --transformer-dropout 0.1 --label-smoothing 0.1 \
        --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
        --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
        --devices $GPUS --sync-sgd --seed $ID$ID$ID$ID --keep-best --overwrite \
        --exponential-smoothing --no-nccl --filter_corpus 0.85 \
        --sr-freq-file cl/mod/en-mod.txt cl/mod/en-mod-cdf_base.npz 2.5 0.01 mod d 0.5 \
        --after-batches 100000

# Evaluation
./decode_validate.sh $MODEL_DIR > $MODEL_DIR/result

# For more details:
# runner/run.sh

# Competence Parameters
# NBCL : --sr-freq-file cl/mod/en-mod.txt cl/mod/en-mod-cdf_base.npz 2.5 0.01 mod [d] [0.5]
# params : word_stat_file CDF_file [MOD: ratio] percentage_of_starting_corpus(c0) mode(mod) [dynamic_weight] [dynamic_ratio]
```

### Procedure
All tools can be found in the runner folder.
1. compile marian source code [Please review marian repo for detail]
2. install fasttext
3. prepare your corpus data
4. use NBCL tools to build word_stat and cdf files
5. train model

### Mainly modified code:
```
src/data/competence.h
src/data/gap_training.h
src/data/batch_generator.h
```
### Great thanks to Marian community
This project is based on the codebase forked from <a href="https://github.com/marian-nmt/marian">Marian</a> (version 13 Dec 2018).

