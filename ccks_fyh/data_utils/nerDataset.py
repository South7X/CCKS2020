from __future__ import absolute_import
import torch
from torch.utils.data import Dataset,DataLoader,SequentialSampler, TensorDataset
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from data_utils.utils import MyBertTokenizer
import os
from ner_config import base_config
from data_utils.utils import get_ner_data_by_txt
import math
import numpy as np
import random
import re
import json
import concurrent.futures
import time
TOKEN_MAP = {"(":"（",")":"）"}


class NerDataset(object):
    '''
    对新闻文本数据进行实体识别，采用NER模型，首先需要构建数据集，将同一文本对应的所有实体找到，平均一个文本有1.5个实体。
    采用BIO标签进行标注
    '''


    def __init__(self, tokenizer,args):
        self.args=args
        self.tokenizer = tokenizer
        self.train_path = os.path.join(self.args.data_dir,self.args.train_name)
        self.test_path = os.path.join(self.args.data_dir,self.args.test_name)


    def get_train_dev_set(self):
        '''

        :return: train_set:训练集，label为使用BIO标记的数值化token标签
        dev_set: 开发集
        '''
        input_ids, attention_masks, labels = self._get_set(self.train_path, mode=0)
        num_samples = input_ids.shape[0]
        random.seed(self.args.seed)
        indexs = list(range(num_samples))
        dev_index = random.sample(indexs,k=math.floor(self.args.dev_rate*num_samples))  # split dev set from train set
        print(num_samples,dev_index)
        train_index = list(set(indexs) -set(dev_index))
        train_set = TensorDataset(input_ids[train_index,:],attention_masks[train_index,:],labels[train_index,:])
        dev_set = TensorDataset(input_ids[dev_index,:],attention_masks[dev_index,:],labels[dev_index,:])
        return train_set, dev_set



    def get_test_set(self):
        input_ids, attention_masks, labels = self._get_set(self.train_path, mode=1)
        train_set = TensorDataset(input_ids, attention_masks, labels)
        return train_set


    def get_labels(self):
        return ["O","B","I"]



    def _denoise(self,content):
        content = content.replace("#","")
        content = re.sub("[①②③④⑤⑥⑦⑧⑨⑩]","",content)
        content = content.replace("(","（")
        content = content.replace(")","）")
        content = content.lower()
        content = content.replace(" ","")
        return content

    def _get_set(self,path,mode):
        '''
        :param path:csv 文件的路径
        :param mode: 训练集为0,测试集为1
        :return:
        '''
        data = get_ner_data_by_txt(path,mode)
        if mode==0:
            self.train_data = data
        else:
            self.test_data = data
        input_ids = []
        attention_masks = []
        labels = []
        for example in data:
            input_id,attention_mask,label = self.transform(example)
            if input_id is None:
                continue
            input_ids.append(input_id.reshape((1,-1)))
            attention_masks.append(attention_mask.reshape((1,-1)))
            labels.append(label.reshape(1,-1))
        input_ids = torch.cat(input_ids,axis = 0)
        attention_masks = torch.cat(attention_masks,axis = 0)
        labels = torch.cat(labels,axis = 0)
        return input_ids,attention_masks,labels


    def _out_ner_set(self,file_path,mode,out_path):
        '''
        以json的格式输出ner数据集，用来跑其他人的ner模型，忽略
        :param file_path:
        :param mode: 训练集为0,测试集为1
        :param out_path:
        :return:
        '''
        data = get_ner_data_by_txt(file_path, mode)
        out_data = []
        for sample in data:
            sentence = sample["content"]
            entity_list = sample["entity_list"]
            sentence,entity_list_2,labels = self._get_label_seq(sentence,entity_list)
            assert len(sentence)==len(labels),"{}\n{}".format(len(sentence),len(labels))
            if labels==['O']*len(sentence):
                print("{}\n{}".format(sentence,entity_list))
            else:
                out_data.append({"content":sentence,"labels":labels})
        num_samples = len(out_data)
        # assert num_samples==33516, "number of samples error: {}".format(num_samples)
        indexs = list(range(num_samples))
        random.seed(42)
        dev_index = random.sample(indexs,k=math.floor(0.2*num_samples))
        train_index = list(set(indexs)-set(dev_index))
        dev_data = []
        train_data = []
        for i in dev_index:
            dev_data.append(out_data[i])
        for i in train_index:
            train_data.append(out_data[i])
        with open(os.path.join(out_path,"train.json"),'w') as file:
            json.dump(train_data,file,ensure_ascii=False,indent="\t")
        with open(os.path.join(out_path,"dev.json"),'w') as file:
            json.dump(dev_data,file,ensure_ascii=False,indent="\t")



    def _get_label_seq(self,sentence,entity_list):
        '''
        为了运行以json格式输出训练集，请忽略
        :param sentence:
        :param entity_list:
        :return:
        '''
        Label_schema = ['O','B-ENT','I-ENT']
        labels = ['O']*len(sentence)
        sentence = sentence.replace("(","（")
        sentence = sentence.replace(")","）")
        sentence = sentence.lower()
        for index, entity in enumerate(entity_list):
            start = 0
            pos = []
            entity_len = len(entity)
            entity = entity.replace("(", "（")
            entity = entity.replace(")", "）")
            entity = entity.lower()
            while start<len(sentence):
                p = sentence.find(entity,start)
                if p!=-1:
                    pos.append(p)
                    labels[p:p+entity_len] = ["I-ENT"] *entity_len
                    labels[p] = "B-ENT"
                    start = p+entity_len
                else:
                    start +=1
            if not pos:
                sentence = sentence.replace(" ","")
                entity = entity.replace(" ","")
                labels = ['O'] * len(sentence)
                entity_list[index] = entity ##update the entity list
                start = 0
                entity_len = len(entity)
                entity = entity.replace("(", "（")
                entity = entity.replace(")", "）")
                entity = entity.lower()
                while start < len(sentence):
                    p = sentence.find(entity, start)
                    if p != -1:
                        pos.append(p)
                        labels[p:p + entity_len] = ["I-ENT"] * entity_len
                        labels[p] = "B-ENT"
                        start = p + entity_len
                    else:
                        start += 1
            if not pos:
                print("sentence: {}\nentity: {}".format(sentence,entity))
        return sentence,entity_list,labels




    def _find_all(self,token1, token2):
        '''
        找到实体token在新闻token中出现的所有位置
        :param token1:
        :param token2:
        :return:
        '''
        for tokens in [token1,token2]:
            for i in range(len(tokens)):
                if tokens[i] in TOKEN_MAP:
                    tokens[i] = TOKEN_MAP[tokens[i]]
        pos = []
        t2_len= len(token2)
        for i in range(len(token1)):
            if token1[i:i+t2_len]==token2:
                pos.append(i)
        return pos





    def transform(self,example):
        '''
        将一个具有（content，entity_list）的数据进行数值化转换，获得模型所需要的输入数据和标签
        :param example: {content: str,entity_list: []}
        :return:
        '''
        entity_list = example["entity_list"]
        content = example["content"]
        if entity_list:
            content = self._denoise(content)
        content_tokens = self.tokenizer.tokenize(content)
        labels = [0] * len(content_tokens)
        if not entity_list:
            if len(content_tokens) > self.args.max_seq_len -2:
                content_tokens=content_tokens[:self.args.max_seq_len-2]
                labels = labels[:self.args.max_seq_len-2]
                padding_len = 0
            else:
                padding_len = self.args.max_seq_len-2-len(content_tokens)
        else:
            for entity in entity_list:
                entity = self._denoise(entity)
                entity_tokens = self.tokenizer.tokenize(entity)
                pos = self._find_all(content_tokens,entity_tokens)
                if pos==[]:
                    print("no pos: {}\n{}\n".format(entity,content))
                    pass
                for p in pos:
                    labels[p] = 1
                    if len(entity_tokens)==1:
                        pass
                    else:
                        labels[p:p+len(entity_tokens)] = [2]*len(entity_tokens)  # 'B': 1, 'I': 2, 'O': 0
                        labels[p] = 1
            # try:
            #     assert np.sum(labels)>0, "{}\n{}\n{}".format(content,entity,content)
            # except Exception as e:
            #     print(e)
            #     return None,None,None
            if len(content_tokens) > self.args.max_seq_len -2:
                content_tokens = content_tokens[:self.args.max_seq_len-2]
                labels = labels[:self.args.max_seq_len -2]    ##可能存在截断
                padding_len = 0
            else:
                padding_len  = self.args.max_seq_len -2 - len(content_tokens)
        content_tokens = ["[CLS]"]+content_tokens +["[SEP]"]
        content_ids = self.tokenizer.convert_tokens_to_ids(content_tokens)
        attention_mask = [1] * len(content_ids)
        labels = [0] + labels +[0] + [0]* padding_len
        content_ids += [0]*padding_len
        attention_mask += [0]*padding_len
        assert len(content_ids)==len(labels)==len(attention_mask)==self.args.max_seq_len, "content ids length error"
        # try:
        #     assert np.sum(labels)> 0, "len of content:{}\n entity:{}\ncontent:{}\n".format(len(content),entity_list,content)
        # except Exception as e:
        #     print(e)
        #     return None,None,None
        content_ids = torch.tensor(content_ids)
        attention_mask  = torch.tensor(attention_mask)
        labels = torch.tensor(labels,dtype=torch.long)
        return content_ids,attention_mask,labels




if __name__=="__main__":
    tokenizer = MyBertTokenizer.from_pretrained("/home/fuyonghao/ccks_event/pretrained_models/bert_base")
    ss = "我 爱 北京 天安门"
    tokens = tokenizer.tokenize(ss)
    print(tokens)
    # data_set = NerDataset(tokenizer,base_config)
    # data_set.get_train_dev_set()



