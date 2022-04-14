import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

from .head import ClsCntRegHead
from .fpn_neck import FPN
from .backbone.resnet import resnet50
from .loss import GenTargets,LOSS,coords_fmap2orig
from .config import DefaultConfig
from .backbone.darknet19 import Darknet19

class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig # 可以看到，如果不输入配置文件，则默认调用上面的配置
        if config.backbone == "resnet50":
            self.backbone = resnet50(pretrained=config.pretrained)
        elif config.backbone == "darknet19":
            self.backbone = Darknet19(pretrained=config.pretrained)

        self.fpn = FPN(config.fpn_out_channels,
                       use_p5=config.use_p5,
                      backbone=config.backbone)

        self.head = ClsCntRegHead(config.fpn_out_channels,
                                  config.class_num,
                                  config.use_GN_head,
                                  config.cnt_on_reg,
                                  config.prior)
        self.config = config

    def train(self, mode=True):
        """
        set module training mode, and frozen bn
        """
        super().train(mode=mode)

    def forward(self, x):
        """
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        # C3缩小8倍，C4缩小16倍，C5缩小32倍
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        五个分支一次行全传进来了
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        # 5个分支的cls预测处理完成
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
        # 5个分支cnt处理完成
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        # 5个分支的reg处理完成
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]

        # cls和cnt走sigmoid
        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()

        coords =coords.cuda() if torch.cuda.is_available() else coords

        # 值，索引，索引就是我们所需要的类别数
        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))#[batch_size,sum(_h*_w)]  ：每一个anchor point的最终预测分数是这么来的
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]  # torch索引是从0开始，所以这里+1才是我们对应的类别
        #--------------------------------------------------------------------------------------------------------------

        # 得到框的左上角和右下角坐标
        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])  # 150个框
        # 分类分数最大的150个框的索引
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        # 后处理传150个框进去
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            # 先设一个阈值，满足要求得框取出来
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold) #nms到这里结束
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)

        return scores,classes,boxes

    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)  # 150个框，每一个框的面积
        order=scores.sort(0,descending=True)[1]  # 对框的分数进行排序并返回其对应的索引
        keep=[]
        # 开始nms了
        # order.numel()看列表中有多少个元素（在所有维度上）
        while order.numel()>0:
            # 如果只有一个框那就不过滤了
            if order.numel()==1:
                i=order.item()  # 这种取值精度更高
                keep.append(i)
                break
            else:
                # 先取第0个加进去，加索引，这里返回的是分数最大的那个框的索引
                i=order[0].item()  # 我这里要得到的是他在这150个中的索引
                keep.append(i)

            # 这里限制是直接求了交集
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)  # 计算取出来的框和目前这个基准框的iou交并比
            #
            idx=(iou<=thr).nonzero().squeeze()  # 取出所有是true的索引,满足要求的，这里取得是满足要求的框的索引，你这里求索引的时候，没考虑当前元素了
            if idx.numel()==0:
                break
            # print("idx",idx)
            order=order[idx+1]  #  这里是为了补和x的索引差
        #     print("order",order)
        # print("keep",keep)
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):

        # 如果这个张量里面没有内容，直接返回
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()  # 没给维度，求这个张量里面所有值得那个最大值
        # print("idxs",idxs)  # 这150个框得类别数
        # print("idxs_to",idxs.to(boxes))  # 跟上面是一样的
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]  # 框的4个坐标值每一个都加上150个框中最大的那个值
        # 开始过滤
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes



# 这里才是整个网络的流程
class FCOSDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        if mode=="training":
            self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range)
            self.loss_layer=LOSS()  # 训连的话没有什么好说的，让loss最小即可
        elif mode=="inference":
            self.detection_head=DetectHead(config.score_threshold,config.nms_iou_threshold,
                                            config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes=ClipBoxes()

    def forward(self,inputs):
        '''
        inputs:dataset打包好的
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode=="training":
            batch_imgs,batch_boxes,batch_classes=inputs
            out=self.fcos_body(batch_imgs)  # 这里是出来了每个尺度的三个分支：cnt，cls，reg
            # 正负样本从这里开始匹配
            targets=self.target_layer([out,batch_boxes,batch_classes])  # cnt是拿anchor point 和真实框以及公式得来的，它的作用是让目标回归的好
            # print("targets",targets)
            losses=self.loss_layer([out,targets])
            return losses
        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net
            '''
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)  # 走网络架构
            scores,classes,boxes=self.detection_head(out)  # 把对应索引里面的东西取出来
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes

        elif self.mode=="deploy":
            out = self.fcos_body(inputs)
            # Step 1. Concat Cls Output, cls_out shape [1,20,#anchor points]
            cls_out = F.sigmoid(torch.cat([out[0][i].view(1,20,-1) for i in range(len(out[0]))], -1))
            # Step 2. Concat Cnt Output, cnt_out shape [1,1,#anchor points]
            cnt_out = F.sigmoid(torch.cat([out[1][i].view(1,1,-1) for i in range(len(out[1]))], -1))
            # Step 3. Concat Reg Output, reg_out shape [1,4,#anchor points]
            reg_out = torch.cat([out[2][i].view(1,4,-1) for i in range(len(out[2]))], -1)
            return cls_out,cnt_out,reg_out

