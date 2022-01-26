from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score
from numpy import mean
import json

import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, AdamW

import bert_utils

# PYTORCH_PRETRAINED_BERT_CACHE = ".pytorch_pretrained_bert"

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OutputObject:
    """docstring"""

    def __init__(self, segment, predicted_class, cl):
        """Constructor"""
        self.predicted_class = predicted_class
        self.segment = segment
        self.cl = cl


def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = bert_utils.rev_wordpiece(tokens)
    return outputs


def main():
    parser = argparse.ArgumentParser()

    ## required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--save_model_dir", default="cbert_model", type=str,
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--task_name", default="discourse", type=str,
                        help="The path of the classification results.")
    parser.add_argument("--output_dir", default="classification_results", type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=9.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument('--sample_num', default=1, type=int,
                        help="sample number")
    parser.add_argument('--sample_ratio', default=7, type=int,
                        help="sample ratio")
    parser.add_argument('--gpu', default=0, type=int,
                        help="gpu id")
    parser.add_argument('--temp', default=1.0, type=float,
                        help="temperature")
    parser.add_argument('--istrain', default=False, type=bool,
                        help="train or dev datasets")
    parser.add_argument('--istest', default=True, type=bool,
                        help="train or dev datasets")

    args = parser.parse_args()
    with open("global.config", 'r') as f:
        configs_dict = json.load(f)

    args.task_name = configs_dict.get("dataset")
    print(args)

    """prepare processors"""
    AugProcessor = bert_utils.AugProcessor()
    processors = {
        "amazon": AugProcessor,
        "discourse": AugProcessor
    }

    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    task_name = 'discourse'
    processor = processors[task_name]
    label_list = processor.get_labels(task_name)

    ## prepare for model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    def accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def precision(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        precision = precision_score(labels_flat, preds_flat, average='weighted')
        return precision

    def recall(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        recall = recall_score(labels_flat, preds_flat, average='weighted')
        return recall

    def f1(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        f1 = f1_score(labels_flat, preds_flat, average='weighted')
        return f1

    def load_model(model_name):
        weights_path = os.path.join(args.save_model_dir, model_name)
        model = torch.load(weights_path)
        return model

    args.data_dir = os.path.join(args.data_dir, task_name)
    args.output_dir = os.path.join(args.output_dir, task_name)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    #shutil.copytree("aug_data/{}".format(task_name), args.output_dir)
    
    a = 1
    print("This is ", a)
    ## prepare for training
    if (a == 0):#args.istrain):
        print('TRAIN')
        train_examples = processor.get_train_examples(args.data_dir)
    elif (a == 1):#args.istest):
        print('TEST')
        train_examples = processor.get_test_examples(args.data_dir)
    else:
        print('DEV')
        train_examples = processor.get_dev_examples(args.data_dir)
    train_features, num_train_steps, train_dataloader = \
        bert_utils.construct_train_dataloader(train_examples, label_list, args.max_seq_length,
                                              args.train_batch_size, args.num_train_epochs, tokenizer, device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    save_model_dir = os.path.join(args.save_model_dir, task_name)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    MASK_id = bert_utils.convert_tokens_to_ids(['[MASK]'], tokenizer)[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #origin_train_path = os.path.join(args.output_dir, "train_origin.tsv")
    #save_train_path = os.path.join(args.output_dir, "train.tsv")
    #train_examples = processor.get_train_examples(args.data_dir)
    train_features, num_train_steps, train_dataloader = bert_utils.construct_train_dataloader(train_examples, label_list,
                                                                                              args.max_seq_length,
                                                                                              args.train_batch_size,
                                                                                              args.num_train_epochs, tokenizer,
                                                                                              device)

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        output_examples = []
        save_train_path = os.path.join(args.output_dir, "train_epoch" + str(e) + ".tsv")
        save_train_file = open(save_train_path, 'a')
        tsv_writer = csv.writer(save_train_file, delimiter='\t')
        precision_list = []
        recall_list = []
        f1_list = []

        torch.cuda.empty_cache()
        cbert_name = "{}/BertForMaskedLM_{}_epoch_{}".format(task_name.lower(), task_name.lower(), e + 1)
        model = load_model(cbert_name)
        if (torch.cuda.is_available()):
            model.cuda()
        #shutil.copy(origin_train_path, save_train_path)
        #save_train_file = open(save_train_path, 'a')
        #tsv_writer = csv.writer(save_train_file, delimiter='\t')
        for _, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _, segment_span, class_span = batch
            # print(init_ids)
            outputs = model(init_ids, input_mask, segment_ids, labels=class_span)
            # predictions = torch.nn.functional.softmax(predictions[0]/args.temp, dim=2)
            loss = outputs.loss
            logits = outputs.logits
            precision_list.append(precision(logits.detach().cpu().numpy(), class_span.detach().cpu().numpy()))
            recall_list.append(recall(logits.detach().cpu().numpy(), class_span.detach().cpu().numpy()))
            f1_list.append(f1(logits.detach().cpu().numpy(), class_span.detach().cpu().numpy()))
            
            #print(precision_list[len(precision_list)-1])
            #print(recall_list[len(recall_list)-1])
            #print(f1_list[len(f1_list)-1])
            
            output_examples.append(OutputObject(segment=segment_span.detach().cpu().numpy(),
                                                predicted_class=np.argmax(logits.detach().cpu().numpy(), axis=1).flatten(),
                                                cl=class_span.detach().cpu().numpy()))
        for out_ex in output_examples:
            tsv_writer.writerow(np.hstack([int(out_ex.segment[0]), int(out_ex.predicted_class[0]), int(out_ex.cl[0])]))
        #logger.info("epoch %d augment best precision: %d", e, mean(precision_list))
        print("epoch {} augment best precision:{}".format(e, mean(precision_list)))
        print("epoch {} augment best recall:{}".format(e, mean(recall_list)))
        print("epoch {} augment best F1:{}".format(e, mean(f1_list)))


if __name__ == "__main__":
    main()
