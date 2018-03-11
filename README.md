Tf_modeling
==================
Firstly, this repo was called tf_classification, served as a 
freamwork/example for text_classification.Then I find sequence tagging
and word2vec in fact share many codes with oridinary classification,
so i decide to merge these in and call this repo tf_modeling.

Whether classification or sequence tagging or unsupvervised sequence modling,
they has to support: data reader, various models and objects, training and tensorboard, prediction.They share code in these parts or subparts.

As the freamwork is moudled and light, you can eaily chage code of some subparts for your specific requirments.

### Supports
* data_loader with moduliary and composability
    * tf_utils/data.py: low_level util 
    * data_loaders/\*.py: high level data loader
    * data/\*: data example
* text modeling(sequence modeling):
    * word/char,word_char based embedding
    * add, add_idf; if sequence clf, add localily
    * deep and wide(TODO):which in fact combine wide(logistic) influence(tf has code)
    but they use which called tf.estimator.DNNLinearCombinedClassifier, whereas i want wide features combined with any net model
    * text-cnn: multi-filter-sizes, multi-layer, gated-activation; if sequence clf, without pooling.
    * explanation-cnn: tell which ngram contibute how much score to every label, it could combined with industry or user defined setiment dict.（not support sequence clf)
    * rnn(rnn,lstm, gru, sru, fsmn,..), birnn, cnn_rnn, rnn_cnn, rnn_attn
    * hs_rnn_attn(TODO): represents sequence of sents
    * denpendency/graph cnn(TODO): implement batch structural cnn on graphs,
    if tagging on every tok, no pooling.
    * recursive(TODO) : structural rnn
    * multi_meaning(TODO)
    * all attention(TODO)
* trainning object
    * tagging on whole sequence:
        * exclusive/nonexclusive
        * tag_num small/big
    * tagging on every tok
        * tag_num small/big
    * generation 
* training method and regulization
* tensorboard log
    * loss
    * f1,p,r of train/dev
    * graph
    * once sess run inspect of memory/time of graph
    * visualization word embedding
    * visualization sent embedding
* prediction: 
    * with model code
    * without model code

### Tutorial Usage
It seems i can't auto-test all config parameters for composability.
So i give some examples, every example share basic process:
0. python train.py -i example_configs/config[x].py 
1. nohup tensorboad --logdir logdir &  : keep log watching
you should try change the config.

#### Simple Topic Tagging
data/clf(分类) classes=4
** train.txt/dev.txt 未分词，可以用来作为字模型的分类
** train_tok.txt/dev_tok.txt 分词，可以用来作为词模型或者字词模型的分类
相似的任务包括情感分析、话题分类。

#### ted多语多标签
* data/tag_small(标签) tags=12
train.txt/dev.txt/test.txt 多语词向量迁移学习, train.txt/dev.txt是英文, test.txt是dev.txt对应的德文
如果 word_vec.txt捕捉了多语的词向量，那么在英文上训练的模型在德文上也会有一定效果

#### word segment, pos tag, ner, dependency parsing
使用序列标注模型

#### word2vec
* data/word2vec
词向量也是一个文本分类模型，它的标签数量是相当大的。

#### data/wiki_tag
多语维基标签数据,通过wikidata对齐标签与文本。
ps : clf/utils clf/tf_utils 是2个独立的我自己使用的基础库，所以可能包含其他一些这个项目无关的util。

#### one net for everything
NLP多任务网络：话题分类、情感分析、分词、词性、NER、依存文法、语言模型、[翻译/多语]

### Design Problem
There're several design type of package:
* tensorflow, keras, tensorlayer: Modularity
* tensorflow/nmt, xxx/package: scriptity, relative simple

Some points:
* why I merge various possibilty into one package
because they share code. Same code should writen in one place.
Globally, this style make the code keep consistent and clean.(like multi-task learning)
* shortcomings
As more and more variable added, the code become over-complicated, 
    whereas simple packages keep clean and easy to change.
Code to accept possibility variable and make model is ugly!
* how about sequence-tagging, unsupverised learning, semi-supervised, etc
in fact, they also share code blocks of this tf_classification.
I'll make they share tf_utils, utils, part of model.py(but duplicate), some other code style; 
    make they relative independent.
