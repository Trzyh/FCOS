import torch
import numpy as np
import cv2

def sort_by_score(pred_boxes, pred_labels, pred_scores):
    '''
    预测框
    预测类别
    预测分数
    '''
    # .argsort()：按数据从小到大排序，但我们想让分数最大的在前面，因此对score取了负数
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores

def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    '这里是做了一个拼接'
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 用积分求pr曲线面积也就是ap
    return ap

def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]:长度为你所有列表的数
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]：m和n是你针对不同样本预测出来的框数
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}

    # 这里循环并没有加背景:这是按类别分配了
    for label in range(num_cls)[1:]:
        # get samples with specific label
        #----------------------------------------------------------------------
        '''
        处理真实标签
        '''
        # 这里是将所有样本都循环迭代完成：true和false都有
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]  # 标签预测正确
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]
        #--------------------------------------------------------------------------
        '''
        处理预测标签
        '''
        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        #--------------------------------------------------------------------------
        '''
        预测对的预测的置信度也取出来
        '''
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        '''
        循环迭代样本:所以外循环是没问题的，外循环的索引范围为数据样本的长度
        真实bbox
        预测bbox
        预测scores
        '''
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)  # 求取所有测试样本中，每个类的真实框的数量
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            '这个是预测的'
            for index in range(len(sample_pred_box)):
                # 依次添加score
                scores = np.append(scores, sample_scores[index])  # 把预测出来的置信度也加进去
                #-------------------------------------------------------------------------------------------
                '如果当前样本中没有当前循环的类别，直接跳出当前样本，进入下一个样本'
                '假正例:实际是没有的，但是你预测出来了'
                'TP：真正例'
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    ''
                    fp = np.append(fp, 1)
                    ''
                    tp = np.append(tp, 0)
                    continue
                # -------------------------------------------------------------------------------------------
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)  # 升了一个维
                iou = iou_2d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)  # 取索引
                max_overlap = iou[gt_for_box, 0]
                #iou是一套约束，置信度又是一套约束
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        '这里其实得到的是每个fp和tp中在当前类别中的fp和tp，所以得到的是一个list'
        fp = np.cumsum(fp)   # 因为这个函数，所以其实我们完全可以只以iou阈值为基础来进行此运算，取最后一个即可
        tp = np.cumsum(tp)
        # compute recall and precision
        'total_gts:int'
        # 这里召回和预测都是当期类别的：这里其实是按不同的阈值计算出来的一系列的recall和precision
        recall = tp / total_gts  # 根据每个类别得到了一recall，这里的其内层徐娜混已经迭代完成，total_gts已经是在当前测试数据集中当前类别的总数
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)  # ap在这里返回的时候已经做了求和
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap

if __name__=="__main__":
    from model.fcos import FCOSDetector
    from dataset.VOC_dataset import VOCDataset
    from model.config import DefaultConfig 

    config = DefaultConfig
    # 处理数据
    eval_dataset = VOCDataset(root_dir='D:\jinhua\shenduzhiyan\image_object_detection\FCOS\data\VOCdevkit', resize_size=[640, 800],
                               split='my_test', use_difficult=False, is_train=False, augment=None, mean=config.mean, std=config.std)
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)

    # 模型
    model=FCOSDetector(mode="inference")
    model = torch.nn.DataParallel(model)
    # 加载训练完成以后fcos的权重
    # 这是我们自己训练出来的权重
    model.load_state_dict(torch.load(r"D:\jinhua\shenduzhiyan\image_object_detection\FCOS\training_dir\model_best.pth")['loader'])
    model=model.cuda().eval()
    print("===>success loading model")

    # 开始进行模型的评估
    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    # 将待评估的所有数据的预测框，预测类别，预测分数，和真实框，真实类别全部先放到一起
    num=0
    for img,boxes,classes,_ in eval_loader:
        # 不分配梯度
        with torch.no_grad():
            out=model(img.cuda())  # 推断的时候这是输出的是经过nms以后的框，我们刚开始训练一轮，所以结果差异很正常
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        print(num,end='\r')
    #--------------------------这一行代码只针对预测框，类别，分数
    # 按分数由大到小的调整一下
    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    # 得到每个类的ap并存储至一个字典中
    #----------------------------------------------------------------------------------------------
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME))
    print("all classes AP=====>\n")
    for key,value in all_AP.items():
        print('ap for {} is {}'.format(eval_dataset.id2name[int(key)],value))
    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(eval_dataset.CLASSES_NAME)-1)  # 求mAP
    print("mAP=====>%.3f\n"%mAP)

