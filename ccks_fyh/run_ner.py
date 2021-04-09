import time
import torch
from torch import nn
from torch.nn.parallel import DataParallel
import logging
import numpy as np
import random
from torch.utils.data import SequentialSampler,RandomSampler,DataLoader
from tqdm import tqdm,trange
from itertools import cycle
from torch import nn
from transformers.modeling_bert import BertConfig
from models.nerModel import BertCrfForNer
# from transformers.tokenization_bert import MyBertTokenizer
from data_utils.utils import MyBertTokenizer
from transformers import AdamW
import pandas as pd
from data_utils import NerDataset
from sklearn.metrics import classification_report, f1_score
import os

def get_logger(args=None):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)
logger = get_logger()


class Instructor(object):

    def __init__(self,args):
        self.args = args
        self.set_seed()
        self.load_model() ##加载模型
        self.data_set = NerDataset(tokenizer=self.tokenizer, args=self.args)
        self.train_set,self.dev_set = self.data_set.get_train_dev_set()
        self.test_set =None



    def train(self):
        sampler = RandomSampler(self.train_set)
        train_loader = DataLoader(self.train_set,batch_size=self.args.batch_size,sampler=sampler)
        logging.info("train loader length: {}".format(len(train_loader)))
        bar = tqdm(range(len(train_loader) * self.args.epochs), total=len(train_loader) * self.args.epochs)
        train_loader = cycle(train_loader)
        param_optimizer = list(self.model.named_parameters())

        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=len(bar))

        self.set_seed() ##for reproduction
        best_f1=0
        for step in bar:
            batch = next(train_loader)
            input_ids = batch[0]
            mask_ids = batch[1]
            labels = batch[2]
            if self.args.n_gpus>0:
                input_ids = input_ids.cuda()
                mask_ids = mask_ids.cuda()
                labels = labels.cuda()
            loss,logits = self.model(input_ids,attention_mask=mask_ids,labels=labels)
            if self.args.n_gpus>1:
                loss = loss.mean()
            loss.backward() ##反向传播计算梯度
            optimizer.step() ##优化器进行优化
            optimizer.zero_grad()  ##清除一下梯度
            bar.set_description("train loss: {}".format(loss.item()))
            if (step+1)%self.args.eval_steps==0:
                eval_f1 = self.evaluate()
                logger.info("dev set F1: {}".format(eval_f1))
                if eval_f1>best_f1:
                    best_f1=eval_f1
                    logger.info("saving model on best F1: {}".format(best_f1))
                    self.save_model()
        logger.info("Training finished")



    def evaluate(self):
        self.model.eval()
        sampler = RandomSampler(self.dev_set)
        dev_loader = DataLoader(self.dev_set, batch_size=self.args.eval_batch_size, sampler=sampler)
        dev_len = len(dev_loader)
        dev_loader = iter(dev_loader)
        bar = tqdm(range(dev_len),total=dev_len)
        eval_loss = 0
        y_true = []
        y_pred = []
        for step in bar:
            batch  = next(dev_loader)
            input_ids= batch[0]
            mask_ids = batch[1]
            labels = batch[2]
            if self.args.n_gpus>0:
                input_ids = input_ids.cuda()
                mask_ids = mask_ids.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                loss, logits = self.model(input_ids,attention_mask = mask_ids,labels = labels)
            if self.args.n_gpus>1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            bar.set_description("eval loss: {}".format(loss))
            labels = labels.detach().cpu().numpy()
            tags = self.model.module.crf.decode(logits,mask_ids).squeeze(0).detach().cpu().numpy()
            input_lens = mask_ids.sum(axis=1).detach().cpu().numpy().tolist()
            assert len(labels) == len(tags)==len(input_lens), "labels shape: {}\ntags shape: {}\n input lens shape: {}".format(len(labels),len(tags),len(input_lens))
            for i, label in enumerate(labels):
                tag = tags[i]
                input_len = input_lens[i]
                tmp_true,tmp_pred=self._turn_into_binary_classification(tag,label,input_len)
                y_true += tmp_true
                y_pred += tmp_pred
        eval_f1= f1_score(y_true,y_pred,average='binary')
        print("evaluation loss:  {}".format(eval_loss/dev_len))
        return eval_f1




    def _turn_into_binary_classification(self,labels, tags,input_len):
        '''
        将整个实体预测用F1值来表示，正确预测一个实体则TP+1.否则在FP或者FN中+1.（TP: True positive真正例，FP：False Negtive假正例）
        :param labels:由0,1,2组成的token标签序列，[0,0,0,1,2,0,1,2,2,0,0,0,0]
        :param tags: 有0,1,2组成的预测序列，      [0,0,0,0,1,0,1,2,2,0,0,0,0]
        :param input_len:输入的句子长度，防止padding序列加入计算中，虽然加入了也没什么问题
        :return: y_true = [1,0,1], y_pred = [0,1,1]
        '''
        y_true = []
        y_pred = []
        i = 0
        while i < input_len:
            if tags[i] == 1 and labels[i] == 1:
                pred_entity_len = 1
                label_entity_len = 1
                j = i + 1
                while tags[j] == 2:
                    pred_entity_len += 1
                    j += 1
                j = i + 1
                while labels[j] == 2:
                    label_entity_len += 1
                    j += 1
                if pred_entity_len == label_entity_len:
                    y_pred.append(1)
                    y_true.append(1)
                elif pred_entity_len < label_entity_len:
                    y_pred.append(1)
                    y_true.append(0)
                else:
                    y_pred.append(0)
                    y_true.append(1)
                i += min(label_entity_len, pred_entity_len)
            elif tags[i] == 0 and labels[i] == 1:
                y_true.append(1)
                y_pred.append(0)
                i += 1
            elif tags[i] == 1 and labels[i] == 0:
                y_pred.append(1)
                y_true.append(0)
                i += 1
            else:
                i += 1
        return y_true, y_pred




    def prediction(self):
        if not self.test_set:
            self.test_set = self.data_set.get_test_set()
        self.model.eval()
        sampler = SequentialSampler(self.test_set)
        test_loader = DataLoader(self.test_set, batch_size=self.args.eval_batch_size, sampler=sampler)
        test_len  = len(test_loader)
        test_loader = iter(test_loader)
        bar = tqdm(range(test_len), total=test_len)
        all_entities = []
        eval_loss = 0
        for step in bar:
            batch = next(test_loader)
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            if self.args.n_gpus > 0:
                input_ids = input_ids.cuda()
                mask_ids = mask_ids.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                loss, logits = self.model(input_ids = input_ids, attention_mask = attention_mask,labels=labels)
            if self.args.n_gpus > 1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            bar.set_description("eval loss: {}".format(loss))
            tags = self.model.crf.decode(logits).squeeze(0).detach().cpu().numpy()
            input_lens = mask_ids.sum(axis=0).detach().cpu().numpy()
            entities_one_batch = self.get_entities(tags,input_lens)
            all_entities +=entities_one_batch
        self.out_prediction(all_entities)
        self.model.train()  ##convert to train mode
        return

    def out_prediction(self,all_entities):
        time_stamp = "_".join(time.ctime().split(":")[0:2]).replace(" ", "_")
        filename = self.args.memo + time_stamp + "_submit.csv"
        out_path = os.path.join(self.args.out_dir, filename)
        assert len(self.data_set.test_data)==len(all_entities), "test data length and entities length don't match"
        test_df = pd.DataFrame(self.data_set.test_data)
        out_series = []
        for idx, row in test_df.iterrows():
            chunk_list = all_entities[idx]
            entity_list = []
            sentence = row['content']
            for pos in chunk_list:
                start = pos[0]-1
                end = pos[1]-1
                assert start >=0 and end<len(sentence) and start<end,"pos: {}\n sentence: {}".format(pos,sentence)
                entity = sentence[start:end]
                entity_list.append(entity)
            out_series.append(pd.Series(row['uid'],entity_list))
        result_df = pd.concat(out_series).reset_index()
        result_df.columns = ["entity","uid"]
        result_df[['uid',"entity"]].to_csv(out_path,index=False)






    def get_entities(self,tags,input_len):
        '''
        :param tags: [0,0,0,0,1,2,2,1,0,0,0] tag sequence for  "[CLS]XXXXXX[SEP]000000"
        :return:
        '''
        batch_entities = []
        for seq in tags:
            entities = []
            for idx, value in enumerate(seq):
                if value==1:
                    end = idx
                    while end + 1 < len(seq) and seq[end+1] == 2:
                        end += 1
                    entities.append((idx,end))
            batch_entities.append(entities)
        return batch_entities


    def save_model(self):
        if hasattr(self.model,"module"):
            model_to_save=self.model.module
        model_path = os.path.join(self.args.out_dir,"model/pytorch_model.bin")
        torch.save(model_to_save.state_dict(),model_path)


    def load_model(self):
        self.tokenizer = MyBertTokenizer.from_pretrained(self.args.pretrained_path,do_lower_case=self.args.do_lower_case)
        self.config = BertConfig.from_pretrained(self.args.pretrained_path,num_labels=self.args.num_labels)
        if self.args.resume_model:
            self.model = BertCrfForNer.from_pretrained(self.args.resume_model_path,config=self.config)
        else:
            self.model = BertCrfForNer.from_pretrained(self.args.pretrained_path,config=self.config)
        if self.args.cuda:
            self.model.cuda()
            if self.args.n_gpus>1:
                self.model = DataParallel(self.model)



    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpus>0:
            torch.cuda.manual_seed_all(self.args.seed)



if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    from ner_config import base_config
    args = base_config
    args.cuda = torch.cuda.is_available()
    args.n_gpus = torch.cuda.device_count()
    trainer = Instructor(args)
    model_path  = os.path.join(args.out_dir,"model")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if args.do_train:
        trainer.train()
    if args.do_dev:
        trainer.evaluate()
    if args.do_test:
        trainer.prediction()
    # trainer.data_set.get_test_set()
    # num_test = len(trainer.data_set.test_data)
    # all_entities = [ [(1,3),(1,3)] for i in range(num_test)]
    # trainer.out_prediction(all_entities)


