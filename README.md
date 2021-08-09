# Discourse-BERT
This is a repository for *disBERT* model (a traditional BERT-[base]} model with discourse extension) implementation.

The repository contains two datasets used for the experiments ([UKP Corpus](https://github.com/lizagonch/Discourse-BERT/tree/master/datasets/discourse) and [Amazon Review dataset](https://github.com/lizagonch/Discourse-BERT/tree/master/datasets/amazon), these are already augmented datasets).

1. Data pre-processing procedure is made with DPLP discourse parser - https://github.com/jiyfeng/DPLP
2. To launch *disBERT* further pre-training on the modified discourse-enhanced task run the training procedure with the command *python discourse_bert_sequence.py --task_name *name_of_task**
3. To provide classification with the pre-trained *disBERT* run the classification procedure with the command *python bert_discourse-classification.py*
4. Use global.config file to set the necessary parameters

The pre-trained wights for the model on the UKP corpus are stored in https://drive.google.com/file/d/1YTI7dmnD4W082a_pNuzniKbcTQDBJLOj/view?usp=sharing; AR dataset are stored in https://drive.google.com/file/d/1VA7d9n3E7kBww0Iqm0TeeCcK87KoINGF/view?usp=sharing

Two [ipython notebooks](https://github.com/lizagonch/Discourse-BERT/tree/master/data_pre-processing) show the data-preprocessing procedure and DDG construction.

The retraining of the segment layer is based on implementation of the C-BERT introduced in https://github.com/1024er/cbert_aug
