# KELM: Knowledge Enhanced Pre-Trained Language Representations with Message Passing on Hierarchical Relational Graphs

Incorporating factual knowledge into pre-trained language models (PLM) such as BERT is an emerging trend in recent NLP studies. However, most of the existing methods combine the external knowledge integration module with a modified pre-training loss and re-implement the pre-training process on the large-scale corpus. Re-pretraining these models is usually resource-consuming, and difficult to adapt to another domain with a different knowledge graph (KG). Besides, those works either cannot embed knowledge context dynamically according to textual context or struggle with the knowledge ambiguity issue. In this paper, we propose a novel knowledge-aware language model framework based on fine-tuning process, which equips PLM with a unified knowledge-enhanced text graph that contains both text and multi-relational sub-graphs extracted from KG. We design a hierarchical relational-graph-based message passing mechanism, which can allow the representations of injected KG and text to mutually updates each other and can dynamically select ambiguous mentioned entities that share the same text. Our empirical results show that our model can efficiently incorporate world knowledge from KGs into existing language models such as BERT, and achieve significant improvement on the machine reading comprehension (MRC) task compared with other knowledge-enhanced models.

<p align="center">
<img src="./images/architecture.jpg" width = "800" height = "550" /> <br />
Framework of KELM (left) and illustrates how to generate knowledge-enriched token embeddings (right)
</p>



## Environment

To install requirements:

```
pip install -r requirements.txt
```

The experiments in the paper are worked on 4 V100 GPUs. For using distributed training with multiple GPUs, please utilize Pytorch DistributedDataParallel.

## Datasets and Knowledge Graph

### 1. ReCoRD

For train and dev set, download from [**Re**Co**RD** ](https://sheng-z.github.io/ReCoRD-explorer/)

For test set, download from [SuperGlue](https://super.gluebenchmark.com/tasks)

### 2. **MultiRC** and COPA

Download from from [SuperGlue](https://super.gluebenchmark.com/tasks)

### 3. Knowledge Graph

We uses two knowledge graphs: [WordNet](https://wordnet.princeton.edu/ ) and [NELL](http://rtw.ml.cmu.edu/rtw/)

For NELL, name entity recognition is performed by [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html). For WordNet, we find text matching from this [repository](https://github.com/villmow/datasets_knowledge_embedding)



For convenience, you can download all relative files from following google drive links:

 [link](https://drive.google.com/file/d/1Ax-xSw4QTaluhFqSTQbeNsxnU7YpDs_h/view?usp=sharing) unzip and replace ./data/

## Data Preprocess

download bert-large-cased model from [huggingface](https://huggingface.co/bert-large-cased) to ./cache/bert-large-cased

For train set, 

```
sh data_preprocess_$dataset$_train.sh
```

For dev set, 

```
sh data_preprocess_$dataset$_dev.sh
```

## Train

We follow the same "two-staged" training strategy with [KT-NET](https://github.com/tanvinerkar/KTNET)

Firstly, freeze language model  and run

```
sh run_first_$dataset$.sh
```

Then unfreeze language model and run from the saved first stage model(replace INIT_DIR with local address of saved model)

```
sh run_second_$dataset$.sh
```

## Results

Our model(backbone: BERT) achieves the following performance on:

ReCoRD:

| Model name               | EM/F1(Dev)    | EM/F1(Test)   |
| ------------------------ | ------------- | ------------- |
| BERT<sub>large</sub>     | 70.2/72.2     | 71.3/72.0     |
| SKG+BERT<sub>large</sub> | 70.9/71.6     | 72.2/72.8     |
| KT-NET<sub>WordNet</sub> | 70.6/72.8     | -             |
| KT-NET<sub>NELL</sub>    | 70.5/72.5     | -             |
| KT-NET<sub>BOTH</sub>    | 71.6/73.6     | 73.0/74.8     |
| KELM<sub>WordNet</sub>   | **75.4/75.9** | 75.9/76.5     |
| KELM<sub>NELL</sub>      | 74.8/75.3     | 75.9/76.3     |
| KELM<sub>BOTH</sub>      | 75.1/75.6     | **76.2/76.7** |

MultiRC:

| Model name                        | EM/F1(Dev)    | EM/F1(Test)   |
| --------------------------------- | ------------- | ------------- |
| BERT<sub>large</sub>              | -             | 24.1/70.0     |
| KT-NET<sub>BOTH</sub><sup>*</sup> | 26.7/**71.7** | 25.4/**71.1** |
| KELM<sub>WordNet</sub>            | 29.2/70.6     | 25.9/69.2     |
| KELM<sub>NELL</sub>               | 27.3/70.4     | 26.5/70.6     |
| KELM<sub>BOTH</sub>               | **30.3**/71.0 | **27.2**/70.8 |

*from our implementation

COPA:

| Model name             | Accuracy(Dev) | Accuracy(Test) |
| ---------------------- | ------------- | -------------- |
| BERT<sub>large</sub>   | -             | 70.6           |
| KELM<sub>WordNet</sub> | 76.1          | **78.0**       |

We also implement KELM base on RoBERTa<sub>large</sub> and evaulate it on ReCoRD:

| Model name              | EM/F1(Dev)    | EM/F1(Test)   |
| ----------------------- | ------------- | ------------- |
| RoBERTa<sub>large</sub> | 87.9/88.4     | 88.4/88.9     |
| KELM<sub>BOTH</sub>     | **88.2/88.7** | **89.1/89.6** |



## Acknowledgement

The part of code is implemented based on the open source code of [huggingface](https://github.com/huggingface/transformers/tree/master/examples/legacy/question-answering), [DGL](https://github.com/dmlc/dgl) and [KT-NET](https://github.com/tanvinerkar/KTNET), [jiant](https://github.com/nyu-mll/jiant)

## License

We use Apache License 2.0.