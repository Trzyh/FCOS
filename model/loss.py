import torch
import torch.nn as nn
from .config import DefaultConfig
import pywt
import sys
sys.path.append("..")

def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    # print("feature",feature.shape)
    h,w=feature.shape[1:3]
    # print("h,w",h,w)
    # print("str",stride)
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    # print("shifts",shifts_x)
    # print("shifts", shifts_x.shape)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
    # print("shifts", shifts_y)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    # print("shift-x", shift_x)
    # print("shift-y", shift_y)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    # print("shift-x", shift_x)
    # print("shift-y", shift_y)
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    # print("coords",coords.shape)  # 得到h*w里面每个格子的偏移点
    return coords

class GenTargets(nn.Module):
    def __init__(self,strides,limit_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        assert len(strides)==len(limit_range)

    def forward(self,inputs):
        '''
        [out,batch_boxes,batch_classes]
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits,cnt_logits,reg_preds=inputs[0]
        gt_boxes=inputs[1]
        classes=inputs[2]
        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        assert len(self.strides)==len(cls_logits)
        for level in range(len(cls_logits)):
            level_out=[cls_logits[level],cnt_logits[level],reg_preds[level]]
            # 这里是做匹配的
            level_targets=self._gen_level_targets(level_out,gt_boxes,classes,self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            
        return torch.cat(cls_targets_all_level,dim=1),torch.cat(cnt_targets_all_level,dim=1),torch.cat(reg_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''

        # out是我们得到的参数
        cls_logits,cnt_logits,reg_preds=out # 分类参数，是否有物体参数，回归参数
        # print("cls_logits",cls_logits.shape)
        # print("cnt_logits",cnt_logits.shape)
        # print("reg_preds",reg_preds.shape)
        batch_size=cls_logits.shape[0]
        # print("batch",batch_size)
        class_num=cls_logits.shape[1]
        # print("class_num",class_num)
        m=gt_boxes.shape[1]
        # print("m",m)

        cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]
        # print("cls_logits", cls_logits.shape)
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2],得到每一个格子的中心点
        # print("coords",coords)

        cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]
        # print("cls_logits",cls_logits.shape)
        cnt_logits=cnt_logits.permute(0,2,3,1)
        cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        # print("cnt_logits",cnt_logits.shape)
        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))
        # print("reg_preds",reg_preds.shape)

        h_mul_w=cls_logits.shape[1]
        # print("h_mul",h_mul_w)

        x=coords[:,0]
        # print("x",x)
        y=coords[:,1]
        # print("y",y)
        # print("gt",gt_boxes)
        # print("gt_boxes[...,0]",gt_boxes[...,0])
        # print("x[None,:,None]",x[None,:,None])
        # print("gt_boxes[...,0][:,None,:]",gt_boxes[...,0][:,None,:])
        #--------------------------------------------------------------这一步算所有目标有的位置误差还是很大的
        # 每一个点和每个目标左上角x的差异
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m],m个差
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        #----------------------------
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]

        # areas永远是目标框的面积，利用anchor point求取目标框的面积
        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]
        #--------------------------------------------------------------------------------------------------
        c = torch.min(ltrb_off, dim=-1)
        # print("0",c)
        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]  # 获取最小值的值
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]  # 获取最大值的值

        # 判断anchor-point框内还是框外：true/false
        mask_in_gtboxes=off_min>0  # 判断这个anchorpoint是在目标框的内部还是外部
        # 分配anchor-point为正负样本：true/false
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])  # 用目标框的角标来分配正负样本
        # --------------------------------------------------------------------------------------------

        radiu=stride*sample_radiu_ratio  # 这个1.5的分布是为什么
        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2  # gt的中心点的横坐标
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2  # gt的中心点的纵坐标
        # 这里是我们的分布差异
        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        # print("c_loff", c_l_off.shape)
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu
        # print("mask",mask_center.shape)

        # 3层约束:分配正负样本
        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]
        # ----------------------------------------------------------------------------------------------------
        areas[~mask_pos]=99999999  #张量可以这么做：把不符合要求的anchor point计算的面积，给一个很大的值（无效值）

        #-----------------------------------------------------回归标签制作
        # print("areas",areas.shape)
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]  # 当前尺度的anchor point去和回归和分类属于他的最小面积的框
        # print("areas_min",areas_min_ind.shape)
        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]
        #---------------------------------------------------------制作完成

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]
        #-----------------------------------------------------------------class制作完成

        # 在每一个anchor point分配的最小框的面积上进行计算
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]
        #----------------------------------------------------------------cnt制作完成
        assert reg_targets.shape==(batch_size,h_mul_w,4)
        assert cls_targets.shape==(batch_size,h_mul_w,1)
        assert cnt_targets.shape==(batch_size,h_mul_w,1)

        #process neg coords
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2=mask_pos_2>=1  # 一个anchor point和一个或多个框都满足
        assert mask_pos_2.shape==(batch_size,h_mul_w)
        # 当前分支拉平了
        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]，anchor point没匹配上的给0
        cnt_targets[~mask_pos_2]=-1  # anchor point没匹配上的给-1：有的话就正常计算即可，没的话就-1
        reg_targets[~mask_pos_2]=-1  # anchor point没匹配上的给-1
        
        return cls_targets,cnt_targets,reg_targets

def compute_cls_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    preds_reshape=[]
    class_num=preds[0].shape[1]
    mask=mask.unsqueeze(dim=-1)
    # print(mask)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    # print("mask_shape",mask.shape)
    # torch.sum(mask, dim=[1, 2])看有多少个正样本
    # .clamp_(min=1).float()下限设置为1#即每个bs里面正样本数量最少为1，用来是的loss可以正常计算
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,class_num]) # 这是调整以后的维度
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)#[batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2]==targets.shape[:2]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index]#[sum(_h*_w),class_num]
        target_pos=targets[batch_index]#[sum(_h*_w),1]
        # 将target转为one-hot编码
        target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_cnt_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    for batch_index in range(batch_size):
        # print("preds[batch_index]",preds.shape,preds[batch_index].shape)
        # print("mask",mask.shape,mask[batch_index].shape)
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,]
        # print("pos",pred_pos.shape)
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,]
        assert len(pred_pos.shape)==1
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]


# 外部传参，内部调用
def compute_reg_loss(preds,targets,mask,mode='iou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss(pred_pos,target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def iou_loss(preds,targets):
    '''
    在回归里，我们只对有目标的进行位置进行loss的计算，因此这里的维度是：【目标数量，4】
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])  # 取出l+t（选择的是预测框和真实框小的那一个）
    rb=torch.min(preds[:,2:],targets[:,2:])  # 取出r+b（选择的是预测框和真实框小的那一个）
    wh=(rb+lt).clamp(min=0)  # 按小的取值

    overlap=wh[:,0]*wh[:,1]#[n]，计算两者的交集
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])  # 预测框的面积
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])  # 真实框的面积
    iou=overlap/(area1+area2-overlap)  # 计算iou，但iou是越大越好（0-1），我们的loss是越小越好，一旦优化的范围为（0-1），就可以转为ln
    loss=-iou.clamp(min=1e-6).log()  # 这里是log,pytorch里面都是以e为底，也就是ln，这是一个好函数，
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    # 在两个维度上各论各地
    lt = torch.min(preds[:, :2], targets[:, :2])  # 取出l+t（选择的是预测框和真实框小的那一个）
    rb = torch.min(preds[:, 2:], targets[:, 2:])  # 取出r+b（选择的是预测框和真实框小的那一个）
    wh = (rb + lt).clamp(min=0)  # 按小的取值

    overlap = wh[:, 0] * wh[:, 1]  # [n]，计算两者的交集
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])  # 预测框的面积
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])  # 真实框的面积
    iou = overlap / (area1 + area2 - overlap)

    # 计算包围框的面积
    lt_2 = torch.max(preds[:, :2], targets[:, :2])  # 取出l+t（选择的是预测框和真实框小的那一个）
    rb_2 = torch.max(preds[:, 2:], targets[:, 2:])  # 取出r+b（选择的是预测框和真实框小的那一个）
    wh_2 = (rb_2 + lt_2).clamp(min=0)  # 按小的取值

    c_area = wh_2[:, 0] * wh_2[:, 1]  # 包围框的面积
    giou = iou - (c_area - area1 - area2 + overlap) / c_area
    loss = -giou.clamp(min=1e-6).log()
    return loss


def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds=preds.sigmoid()
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*targets+(1.0-alpha)*(1.0-targets)
    loss=-w*torch.pow((1.0-pt),gamma)*pt.log()
    return loss.sum()


# 进入loss函数
class LOSS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        # out target
        preds,targets=inputs
        cls_logits,cnt_logits,reg_preds=preds  # 预测出来的
        cls_targets,cnt_targets,reg_targets=targets  # 制作的标签
        mask_pos=(cnt_targets>-1).squeeze(dim=-1)# [batch_size,sum(_h*_w)]
        # 分类loss
        cls_loss=compute_cls_loss(cls_logits,cls_targets,mask_pos).mean()  # 所有点都参与了分类的计算，没配上的对应位置是0
        # cnt loss
        cnt_loss=compute_cnt_loss(cnt_logits,cnt_targets,mask_pos).mean()  # 仅计算有分配样本的anchor point
        # 回归loss
        reg_loss=compute_reg_loss(reg_preds,reg_targets,mask_pos).mean()  # 仅计算有分配样本的anchor point
        if self.config.add_centerness:
            total_loss=cls_loss+cnt_loss+reg_loss
            return cls_loss,cnt_loss,reg_loss,total_loss
        else:
            total_loss=cls_loss+reg_loss+cnt_loss*0.0
            return cls_loss,cnt_loss,reg_loss,total_loss

if __name__=="__main__":
    loss=compute_cnt_loss([torch.ones([2,1,4,4])]*5,torch.ones([2,80,1]),torch.ones([2,80],dtype=torch.bool))
    print(loss)

