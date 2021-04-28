## A new knowledge-enhance framework

### Environment

check requirements.txt

The experiments in the paper are worked on 4 V100 GPUs.

### Datasets

#### 1. ReCoRD

For train and dev set, download from [**Re**Co**RD** ](https://sheng-z.github.io/ReCoRD-explorer/)

For test set, download from [SuperGlue](https://super.gluebenchmark.com/tasks)

### Knowledge Graph

We uses two knowledge graphs: [WordNet](https://wordnet.princeton.edu/ ) and [NELL](http://rtw.ml.cmu.edu/rtw/)

For NELL, name entity recognition is performed by [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html). For WordNet, we find text matching from this [repository](https://github.com/villmow/datasets_knowledge_embedding)

### Data Preprocess

download bert-large-cased model from [huggingface](https://huggingface.co/bert-large-cased) to ./cache/bert-large-cased

For train set, 

```
sh data_preprocess_train.sh
```

For dev set, 

```
sh data_preprocess_dev.sh
```

### Train

We follow the same "two-staged" training strategy with [KT-NET](https://github.com/tanvinerkar/KTNET)

Firstly, freeze language model  and run

```
sh run_first.sh
```

Then unfreeze language model and run from the saved model(move saved model to ./checkpoint) from the first stage

```
sh run_second.sh
```

We train 10 epochs for first stages and 8 epochs for second stages

### Inference

```
sh run_test.sh
```

