import pandas as pd
import json
from transformers.tokenization_bert import BertTokenizer


def read_data(file_path,mode=0):
    if mode==0:
        df = pd.read_csv(file_path,sep="\t",header=None,names=['id','content',"type","entity"])
    else:
        df = pd.read_csv(file_path,sep="\t",header=None,names=["id","content"])
    return df


def get_ner_data_by_df(file_path,mode):
    data = []
    df = read_data(file_path,mode)
    if mode!=0:
        for row in df.iterrows():
            example = {}
            example["content"] = row[1]
            example["entity_list"] = []
            data.append(example)
    else:
        df = df[df.type.apply(lambda x: type(x)!=float)] ## erase examples without type and entity
        unique_sentences = df.content.unique()
        for sentence in unique_sentences:
            example = {}
            entity_list = _get_entity_list(df,sentence)
            example["content"] = sentence
            example['entity_list'] = entity_list
            data.append(example)
    return data



def get_ner_data_by_txt(file_path,mode):
    data_dict = {}
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file.readlines():
            row = line.split("\t")
            if mode==0 and len(row)==4:
                id = row[0]
                content = row[1]
                type  = row[2]
                entity = row[3].strip()
                if type!="NaN":
                    if content not in data_dict:
                        data_dict[content] = [entity]
                    else :
                        data_dict[content]+=[entity]
                else:
                    continue
            elif mode==1 and len(row)==4:
                id = row[0]
                content = row[1].strip()
                data_dict[id] = content
            else:
                print("reading error: {}".format(line))
    if mode==0:
        data = [{"content":k,"entity_list":v} for k , v in data_dict.items()]
    if mode==1:
        data = [{"uid":k, "content":v,"entity_list": []} for k , v in data_dict.items()]
    return data



def get_cls_data_by_txt(file_path,mode):
    data = []
    if mode == 0:
        train_df = pd.read_csv(file_path, sep='\t', header=None,
                               names=['id', 'content', 'type', 'entity'])
        neg_df = train_df[train_df.type.isna()]
        pos_df = train_df[~train_df.type.isna()]
        new_train_df = pos_df.groupby(by=['content']).type.aggregate(set).reset_index()
        for idx, row in new_train_df.iterrows():
            uid = idx
            content = row['content']
            types = list(row['type'])
            data.append((uid, content, types))
        # for idx, row in neg_df.iterrows():
        #     # 加上负例
        #     uid = idx
        #     content = row['content']
        #     types = None
        #     data.append((uid, content, types))
        #     continue
    elif mode==1:
        test_df = pd.read_csv(file_path,sep = "\t",header =None,names = ['uid','content'])
        print(test_df.shape)
        for _, row in test_df.iterrows():
            na_check = ~row.isna()
            id = row[0]
            content = row[1] if na_check[1] else ""
            type = None
            data.append((id,content,type))
    return data


def _get_entity_list(df,content):
    entity_list = list(df[df.content==content].entity)
    return entity_list

class MyBertTokenizer():

    @classmethod
    def from_pretrained(cls, path,*args,**kwargs):
        obj = cls()
        obj.bert_tokenizer = BertTokenizer.from_pretrained(path,*args,**kwargs)
        return obj

    def tokenize(self,sentence):
        tokens = []
        for c in sentence:
            if c in self.bert_tokenizer.vocab:
                tokens.append(c)
            else:
                tokens.append("[UNK]")
        return tokens

    def convert_tokens_to_ids(self,tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self,ids):
        return self.bert_tokenizer.convert_ids_to_tokens(ids)





