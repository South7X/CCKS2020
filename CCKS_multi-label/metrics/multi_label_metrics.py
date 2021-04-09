import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report

__call__ = ['F1Score']


class F1Score(object):
    '''
        F1 Score for multi-label classification, search for best thresh
    '''
    def __init__(self,thresh=0.5, average='macro', search_thresh=True):
        super(F1Score).__init__()
        self.y_pred = 0
        self.y_true = 0
        self.thresh = thresh
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们队Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        best_pred = 0
        for threshold in [i * 0.01 for i in range(100)]:
            self.y_pred = (y_prob > threshold).astype(int)  # y_pred 没有保存最好的threshold对应的pred值
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
                best_pred = self.y_pred
        return best_threshold, best_score, best_pred

    def __call__(self, logits, target):
        '''
        计算整个结果
        :return:
        '''
        self.y_true = target.cpu().numpy()
        y_prob = logits.sigmoid().data.cpu().numpy()
        if self.thresh and self.search_thresh is False:
            self.y_pred = (y_prob > self.thresh).astype(int)
            return self.y_pred, self.value()
        else:
            thresh, f1, best_pred = self.thresh_search(y_prob=y_prob)
            print("Thresh Search: Best thresh: {} - F1 Score: {}".format(thresh, f1))
            return best_pred, f1, thresh

    def value(self):
        '''
         计算指标得分
         '''
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

