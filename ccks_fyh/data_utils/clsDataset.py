from __future__ import absolute_import
import torch
from torch.utils.data import Dataset,DataLoader,SequentialSampler, TensorDataset
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
import os
from cls_config import base_config
from data_utils.utils import get_cls_data_by_txt
import math
import numpy as np
import random
import re
import json
import concurrent.futures
import time
TOKEN_MAP = {"(":"（",")":"）"}


class ClsDataset(object):
    '''
    分类任务的数据集，进行（新闻内容, 实体）的句对分类任务，在新闻内容中会对实体用< >进行标记，据经验能提升分类效果。
    其中测试集的实体应来自于第一步ner任务的结果。
    类别一共有29类。
    '''

    def __init__(self, tokenizer,args):
        self.args=args
        self.tokenizer = tokenizer
        self.train_path = os.path.join(self.args.data_dir,self.args.train_name)
        self.test_path = os.path.join(self.args.data_dir,self.args.test_name)
        self.label2id = {k:v for v,k in enumerate(self.get_labels())}
        self.id2label = {k:v for k,v in enumerate(self.get_labels())}
        self.test_data = None

    def get_train_dev_set(self):
        '''

        :return: *train_set* TensorDataset 类型，作为分类任务的训练集
        *dev_set* TensorDataset 类型，作为分类任务的测试集
        '''
        input_ids, attention_masks, labels = self._get_set(self.train_path, mode=0)
        num_samples = input_ids.shape[0]
        random.seed(self.args.seed)
        indexs = list(range(num_samples))
        dev_index = random.sample(indexs,k=math.floor(self.args.dev_rate*num_samples))
        train_index = list(set(indexs) -set(dev_index))
        train_set = TensorDataset(input_ids[train_index,:], attention_masks[train_index,:], labels[train_index,:])
        dev_set = TensorDataset(input_ids[dev_index,:], attention_masks[dev_index,:], labels[dev_index,:])
        return train_set, dev_set

    def get_test_set(self):
        '''
        :return: test_set: TensorDataset 类型， 返回句对分类的测试集
        '''
        input_ids, attention_masks, labels = self._get_set(self.test_path, mode=1)
        test_set = TensorDataset(input_ids, attention_masks, labels)
        return test_set

    def get_labels(self):
        return ['业绩下滑', '提现困难', '交易违规', '失联跑路', '涉嫌违法', '不能履职', '涉嫌传销', '投诉维权',
       '财务造假', '涉嫌非法集资', '资金账户风险', '资产负面', '实控人股东变更', '高管负面', '涉嫌欺诈',
       '歇业停业', '重组失败', '履行连带担保责任', '债务违约', '业务资产重组', '股票转让-股权受让',
       '实际控制人变更', '债务重组', '商业信息泄露', '资金紧张', '实际控制人涉诉仲裁', '财务信息造假','信批违规','评级调整']

    def _denoise(self,content):
        '''
        :param content: 待处理的实体或新闻文本内容
        :return:
        '''
        content = content.replace("#", "")
        content = re.sub("[①②③④⑤⑥⑦⑧⑨⑩]", "", content)
        content = re.sub("[\(（][0-9]{6}\.?S?s?H?h?[\)）]", "", content)
        content = content.replace("(", "（")
        content = content.replace(")", "）")
        content = content.replace(" ","")
        return content

    def _get_set(self,path,mode):
        '''

        :param path: 数据集csv文件
        :param mode: 0 为训练集，1为测试集
        :return: 模型所需要的输入
        '''
        data = get_cls_data_by_txt(path,mode)
        input_ids = []
        attention_masks = []
        token_type_ids = []
        labels = []
        for example in data:
            input_id,attention_mask,label = self.transform(example)
            if input_id==None:
                continue
            input_ids.append(input_id.reshape((1,-1)))
            attention_masks.append(attention_mask.reshape((1,-1)))
            # token_type_ids.append(token_type_id.reshape((1,-1)))
            labels.append(label.reshape(1,-1))
        input_ids = torch.cat(input_ids,axis = 0)
        attention_masks = torch.cat(attention_masks,axis = 0)
        # token_type_ids = torch.cat(token_type_ids,axis = 0)
        labels = torch.cat(labels,axis = 0)
        if mode!=0:
            self.test_data = data
        return input_ids,attention_masks,labels

    def transform(self,example):
        '''
        将example四元组数值化, 转换成bert模型需要的数据
        :param example: (id,content,entity,type)组成的四元组
        :return: content_ids：tokenize并转换长id后的新闻文本
        attention_mask： 对应的文本和padding的mask
        token_type_ids： [00000,11111,0000]
        label
        '''
        id = example[0]
        content = example[1]
        # entity = example[2]
        type = example[2]
        if type==None:
            label = 0
        else:
            try:
                label = self.label2id[type]
            except:
                print(id)
                print(content)
                # print(entity)
                print(type)
            content = self._denoise(content)
            # entity = self._denoise(entity) ##preprocessing the text
            # content = self._add_mark(content,entity)
        content_tokens  = self.tokenizer.tokenize(content)
        # entity_tokens = self.tokenizer.tokenize(entity)
        if len(content_tokens) > self.args.max_seq_len -2:
            content_tokens = content_tokens[:self.args.max_seq_len-2]
            padding_len = 0
        else:
            padding_len = self.args.max_seq_len -2 -len(content_tokens)
        content_tokens = ['CLS']+content_tokens+['SEP']
        # token_type_ids = [0] * len(content_tokens)
        # entity_tokens = entity_tokens+['SEP']
        # token_type_ids += [1] *len(entity_tokens)
        # tokens = content_tokens + entity_tokens
        tokens = content_tokens
        attention_mask = [1]*len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [0] * padding_len
        attention_mask += [0] * padding_len
        # token_type_ids +=[0]*padding_len
        assert len(input_ids)==len(attention_mask)==self.args.max_seq_len, "content ids length error"
        content_ids = torch.tensor(input_ids)
        attention_mask  = torch.tensor(attention_mask)
        # token_type_ids = torch.tensor(token_type_ids)
        label = torch.tensor(label, dtype=torch.long)
        return content_ids,attention_mask,label

    def _add_mark(self,content,entity):
        return content.replace(entity,"<"+entity+">")




if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained("/home/fuyonghao/implicit_sentiment/pretrained_models/bert_base")
    data_set = ClsDataset(tokenizer,base_config)
    # train_set,dev_set = data_set.get_train_dev_set()
    test_set = data_set.get_test_set()
    print(test_set[0])
    print(test_set.shape)



