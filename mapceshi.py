import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

np.random.seed(100)


# AP，mAP
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        print(mpre)
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        print(mpre)
    return ap


if __name__ == '__main__':
    # 构造预测置信度概率和GT值
    gt_label = np.random.randint(0, 2, 10).tolist()
    pred = np.random.rand(10).tolist()
    print(gt_label)
    print(pred)
    print(np.unique(np.array(pred)))
    # 根据预测结果计算precision和recall
    precision, recall, thresh = precision_recall_curve(gt_label, pred)
    print("precision------------------------------",len(precision))
    precision = precision[::-1]
    recall = recall[::-1]
    thresh = thresh[::-1]
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("thresh: {}".format(thresh))

    # 根据得到的precision列表和recall列表计算AP，mAP
    ap = voc_ap(recall, precision, use_07_metric=True)
    print(ap)

    ap = voc_ap(recall, precision, use_07_metric=False)
    print(ap)
    # 使用sklearn计算AP
    ap_sklearn = average_precision_score(gt_label, pred)
    print(ap_sklearn)

    """
    结果分析：
    输入：
    gt_label: [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    pred: [0.12156912078311422, 0.6707490847267786, 0.8258527551050476, 0.13670658968495297, 0.57509332942725, 0.891321954312264, 0.20920212211718958, 0.18532821955007506, 0.10837689046425514, 0.21969749262499216]

    # 基于gt_label和pred计算得到的precision
    precision: [1., 1., 1., 0.66666667, 0.75, 0.6, 0.5, 0.42857143, 0.5]

    # 基于gt_label和pred计算得到的recall
    recall: [0., 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1.]

    # 基于pred排序得到的thresh
    thresh: [0.89132195, 0.82585276, 0.67074908, 0.57509333, 0.21969749, 0.20920212, 0.18532822, 0.13670659]

    # 使用11-point方法计算得到的AP： 0.8181818181818181
    # 使用recall方法计算得到的AP：0.8125
    # 使用sklearn方法计算得到的AP：0.8125

    11-point方法AP计算过程：
    1、设定11个点的recall阈值：[0.0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    2、对于这11个recall阈值，在recall和precision列表里面找recall大于阈值对应的precision的最大值
    3、recall阈值列表[0.0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]找到的对应的precision如下：
    [ 1,   1,   1,   1,   1,   1,  0.75,0.75, 0.5,0.5, 0.5]
    4、累加找到的precision，然后除以11得到AP = (1 + 1 + 1 + 1 + 1 + 1 + 0.75 + 0.75 + 0.5 + 0.5 + 0.5) / 11 = 0.8181818181818181

    # recall区间方法AP计算过程(目前最常见的计算方式)：
    1、先根据recall列表划分recall区间[0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]
    2、计算各个recall区间内对应的precision的最大值[1, 1, 0.75, 0.5]
    3、recall区间长度乘以对应的precision，然后求和，0.25 * 1 + 0.25 * 1 + 0.25 * 0.75 + 0.25 * 0.5 = 0.8125
    """