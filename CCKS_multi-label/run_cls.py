import torch
from torch import nn
from torch.nn.parallel import DataParallel
from torch.nn import BCEWithLogitsLoss
import logging
import numpy as np
import random
from torch.utils.data import SequentialSampler,RandomSampler,DataLoader
from tqdm import tqdm,trange
from itertools import cycle
from torch import nn
from transformers.modeling_bert import BertConfig, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers import AdamW
from data_utils import ClsDataset
from model import BertForMultiLable
from metrics import F1Score
from sklearn.metrics import classification_report, f1_score
import os
import time
import pandas as pd

def get_logger(args=None):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)
logger = get_logger()


class Instructor(object):

    def __init__(self,args):
        self.args = args
        self.threshold = 0.2
        self.threshold_path = os.path.join(self.args.out_dir, 'model/best_threshold.txt')
        self.set_seed()
        self.load_model() ##加载模型
        self.data_set = ClsDataset(tokenizer=self.tokenizer, args=self.args)
        self.train_set,self.dev_set = self.data_set.get_train_dev_set()
        self.test_set = None
        self.f1_score = F1Score()


    def train(self):
        sampler = RandomSampler(self.train_set)
        train_loader = DataLoader(self.train_set,batch_size=self.args.batch_size,sampler=sampler)
        logging.info("train loader length: {}".format(len(train_loader)))
        bar = tqdm(range(len(train_loader) * self.args.epochs), total=len(train_loader) * self.args.epochs)
        train_loader = cycle(train_loader)
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
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
            logits = self.model(input_ids,attention_mask=mask_ids)
            loss = self.criterion(logits, labels)
            if self.args.n_gpus > 1:
                loss = loss.mean()
            loss.backward() ##反向传播计算梯度
            optimizer.step() ##优化器进行优化
            optimizer.zero_grad()  ##清除一下梯度
            bar.set_description("train loss: {}".format(loss.item()))
            if (step+1)%self.args.eval_steps==0:
                eval_f1, threshold = self.evaluate()
                logger.info("dev set F1: {}".format(eval_f1))
                if eval_f1>best_f1:
                    best_f1=eval_f1
                    self.threshold = threshold  # 记录最好的阈值
                    logger.info("saving model on best F1: {}".format(best_f1))
                    self.save_model()
        logger.info("Training finished")

    def evaluate(self, report_path=None):
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
                logits = self.model(input_ids,attention_mask=mask_ids)
                loss = self.criterion(logits, labels)
            if self.args.n_gpus>1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            bar.set_description("eval loss: {}".format(loss))
            logits_ = logits.detach().cpu().numpy()
            assert logits_.shape[-1]==len(self.data_set.get_labels()), "logits shape error"
            y_pred.append(logits.cpu().detach())
            y_true.append(labels.cpu().detach())
            # assert len(labels) == len(logits), "labels shape: {}\ntags shape: {}\n input lens shape: {}".format(len(labels),len(preds))
        # print("logits dtype:{}".format(y_pred[0].dtype))  # test for dtype of logits float
        y_pred = torch.cat(y_pred, dim=0).cpu().detach()
        y_true = torch.cat(y_true, dim=0).cpu().detach()
        y_pred, eval_f1, threshold = self.f1_score(y_pred, y_true)  # logits, target
        y_true = y_true.cpu().numpy()
        assert y_true.shape == y_pred.shape, "shape error! y_true shape:{}, y_pred shape:{}".format(y_true.shape, y_pred.shape)
        report = classification_report(y_true, y_pred, target_names=self.data_set.get_labels())
        print("evaluation loss:  {}".format(eval_loss/dev_len))
        print("evaluation F1:  {}".format(eval_f1))
        if report_path:
            with open(report_path,'w') as out_file:
                out_file.write("evaluation F1 {}\n".format(eval_f1))
                out_file.write(report)
        return eval_f1, threshold


    def prediction(self):
        if not self.test_set:
            self.test_set = self.data_set.get_test_set()
        self.model.eval()
        sampler = SequentialSampler(self.test_set)
        test_loader = DataLoader(self.test_set, batch_size=self.args.eval_batch_size, sampler=sampler)
        test_len = len(test_loader)
        test_loader = iter(test_loader)
        bar = tqdm(range(test_len), total=test_len)
        all_preds = []
        for step in bar:
            batch = next(test_loader)
            input_ids = batch[0]
            attention_mask = batch[1]
            if self.args.n_gpus > 0:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # print("predict logits shape:{}".format(logits.shape))
            all_preds.append(logits.cpu().detach())
        all_preds = torch.cat(all_preds, dim=0).cpu().detach()
        # print('all_preds shape after cat before sigmoid:{}'.format(all_preds.shape))  # 900, 29
        all_preds = all_preds.sigmoid().data.cpu().numpy()  # nomalization
        # print('all_preds shape before threshold:{}'.format(all_preds.shape))  # 900, 29
        all_preds = (all_preds > self.threshold).astype(int)  # 超过最佳阈值的作为有效标签
        # transfer from [0,1,0,1,...,0] to ['type1','type3']
        all_preds_ = []
        final_preds = []
        for line in all_preds:
            temp = list(np.where(line == 1))
            all_preds_.append(temp[0].tolist())
        for ans in all_preds_:
            final_preds.append([self.data_set.id2label[i] for i in ans])
        print("all_preds length:{} \n all_preds_ length:{}\n final_preds length:{}\n "
              .format(all_preds.shape[0], len(all_preds_), len(final_preds)))
        # print(final_preds)
        time_stamp = "_".join(time.ctime().split(":")[0:2]).replace(" ","_")
        filename = self.args.memo+time_stamp+"_cls.csv"
        out_path = os.path.join(self.args.out_dir,filename)
        df_out = pd.DataFrame(self.data_set.test_data,columns=["uid","content","type"])
        df_out["label"] = final_preds
        # df_out from uid-['type1','type3'] to uid-'type1' \n uid-'type3'
        # new_df_out = pd.DataFrame({'uid': df_out.uid.repeat(df_out.label.str.len()),
        #                            'label': np.concatenate(df_out.label.values)})
        # new_df_out[["uid","label"]].to_csv(out_path, sep="\t", header=False, index=False, encoding="utf-8")
        df_out[["uid", "label"]].to_csv(out_path, sep="\t", header=False, index=False, encoding="utf-8")
        self.model.train()  ##convert to train mode

    def save_model(self):
        if hasattr(self.model,"module"):
            model_to_save=self.model.module
        model_path = os.path.join(self.args.out_dir,"model/pytorch_model.bin")
        torch.save(model_to_save.state_dict(),model_path)
        with open(self.threshold_path, 'w') as f:
            f.write(str(self.threshold))  # save the best threshold


    def load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_path,do_lower_case=self.args.do_lower_case)
        self.config = BertConfig.from_pretrained(self.args.pretrained_path,num_labels=self.args.num_labels)
        if self.args.resume_model:
            self.model = BertForMultiLable.from_pretrained(self.args.resume_model_path,config=self.config)
            with open(self.threshold_path, 'r') as f:
                self.threshold = float(f.read())   # read the best model's threshold
        else:
            self.model = BertForMultiLable.from_pretrained(self.args.pretrained_path,config=self.config)
        if self.args.cuda:
            self.model.cuda()
            if self.args.n_gpus>1:
                self.model = DataParallel(self.model)

    def criterion(self, pred, target):
        loss_fn = BCEWithLogitsLoss()
        pred = pred.float()
        target = target.float()
        loss = loss_fn(input=pred, target=target)
        return loss

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpus>0:
            torch.cuda.manual_seed_all(self.args.seed)



if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    from cls_config import base_config
    args = base_config
    args.cuda = torch.cuda.is_available()
    args.n_gpus = torch.cuda.device_count()
    trainer = Instructor(args)
    model_path = os.path.join(args.out_dir,"model")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if args.do_train:
        trainer.train()
    if args.do_dev:
        time_stamp = "_".join(time.ctime().split(":")[:2]).replace(" ", "_")
        report_path = time_stamp + "prediction_report.txt"
        trainer.evaluate(os.path.join(trainer.args.out_dir, report_path))
    if args.do_test:
        trainer.prediction()


