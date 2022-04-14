import yaml
from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.config import DefaultConfig
from eval import sort_by_score,iou_2d,_compute_ap,eval_ap_2d





# 获取参数
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=r'E:\example\zhao\FCOS\config\base.yaml', help='specify config file')
    args = parser.parse_args()
    print(args)
    with open(args.config_file, 'r') as stream:
        opts = yaml.safe_load(stream)
    print(opts)
    return opts

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['n_gpu']
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    transform = Transforms()
    config = DefaultConfig
    train_dataset = VOCDataset(root_dir=opt['data_root_dir'], resize_size=[416,416], split='my_train',use_difficult=False,is_train=True,augment=transform, mean=config.mean,std=config.std)

    model = FCOSDetector(mode="training").cuda()
    model = torch.nn.DataParallel(model)
    output_dir = 'training_dir'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']
    #-----------------------------------------处理数据-------------------------------------------------
    #WARMPUP_STEPS_RATIO = 0.12
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt['n_cpu'], worker_init_fn=np.random.seed(0))
    print("total_images : {}".format(len(train_dataset)))
    # 验证迭代
    eval_dataset = VOCDataset(root_dir=r'E:\example\zhao\FCOS\data\VOCdevkit',
                              resize_size=[640, 800],
                              split='my_test', use_difficult=False, is_train=False, augment=None, mean=config.mean,
                              std=config.std)
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)
    print("total_images : {}".format(len(eval_dataset)))
    #------------------------------------------------------------------------------------------------


    steps_per_epoch = len(train_dataset) // BATCH_SIZE  # 每一个epoch迭代次数
    TOTAL_STEPS = steps_per_epoch * EPOCHS  # 训练完成以后，总共的小迭代次数
    WARMPUP_STEPS = 501

    GLOBAL_STEPS = 1
    LR_INIT = 1e-4
    optimizer = torch.optim.SGD(model.parameters(),lr=LR_INIT,momentum=0.9,weight_decay=1e-4)

    model.train()

    map = 0
    w_path = []
    for epoch in range(EPOCHS):
        # 全部训练迭代完成，再验证
        for epoch_step, data in enumerate(train_loader):
            batch_imgs, batch_boxes, batch_classes, _= data
            # print("_",_)
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            #lr = lr_func()------------------------------------------学习率调整
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
            for param in optimizer.param_groups:
                param['lr'] = lr
            if GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
                lr = LR_INIT * 0.1
            for param in optimizer.param_groups:
                param['lr'] = lr
            if GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
                lr = LR_INIT * 0.01
            for param in optimizer.param_groups:
                param['lr'] = lr
            #------------------------------------------------------
            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])  # 我的网络需要三个输入
            loss = losses[-1]  # 取总体损失
            loss.mean().backward()

            optimizer.step()

            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            # 每50小轮打印一轮
            if GLOBAL_STEPS%50 == 0:
                print(
                    "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                    (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                    losses[2].mean(), cost_time, lr, loss.mean()))
            GLOBAL_STEPS += 1
        save_path = os.path.join(output_dir, "model_last.pth")
        torch.save(model.state_dict(),
                save_path)  # epoch是从0开始的
        print("epoch",epoch)
        # 开始验证
        # 模型
        val_model = FCOSDetector(mode="inference")
        

        val_model = torch.nn.DataParallel(val_model)
        # 加载训练完成以后fcos的权重
        # 这是我们自己训练出来的权重

        torch.cuda.empty_cache()
        if epoch != 0:
            val_path = os.path.join(output_dir, "model_best.pth")
            load = torch.load(val_path)
            val_model.load_state_dict(
                load['loader'])
            best_map = load['map']
            val_model = val_model.cuda().eval()
            print("===>success loading model")
            # 开始进行模型的评估
            gt_boxes = []
            gt_classes = []
            pred_boxes = []
            pred_classes = []
            pred_scores = []
            # 将待评估的所有数据的预测框，预测类别，预测分数，和真实框，真实类别全部先放到一起
            num = 0
            for img, boxes, classes, _ in eval_loader:
                # 不分配梯度
                with torch.no_grad():
                    out = val_model(img.cuda())  # 推断的时候这是输出的是经过nms以后的框，我们刚开始训练一轮，所以结果差异很正常
                    pred_boxes.append(out[2][0].cpu().numpy())
                    pred_classes.append(out[1][0].cpu().numpy())
                    pred_scores.append(out[0][0].cpu().numpy())
                gt_boxes.append(boxes[0].numpy())
                gt_classes.append(classes[0].numpy())
                num += 1
                print(num, end='\r')
            # --------------------------这一行代码只针对预测框，类别，分数
            # 按分数由大到小的调整一下
            pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
            # 得到每个类的ap并存储至一个字典中
            # ----------------------------------------------------------------------------------------------
            all_AP, recalls, precisions= eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5,
                                len(eval_dataset.CLASSES_NAME))
            print("all classes AP=====>\n")
            for key, value in all_AP.items():
                print('ap for {} is {}'.format(eval_dataset.id2name[int(key)], value))
            mAP = 0.
            for class_id, class_mAP in all_AP.items():
                mAP += float(class_mAP)
            mAP /= (len(eval_dataset.CLASSES_NAME) - 1)  # 求mAP
            print("mAP=====>%.3f\n" % mAP)
            w_path.append(save_path)
            if epoch==0:
                # map.append(mAP)
                # 第一轮保存当前权重
                pass
            elif epoch>0:
                if mAP>best_map:
                    # os.remove(w_path[-2])
                    # map.append(mAP)
                    # 加载权重
                    data = {'loader': model.state_dict(), 'map': mAP}
                    torch.save(data, os.path.join(output_dir, "model_best.pth"))
                else:
                    # os.remove(w_path[-1])
                    pass
        else:
            data = {'loader' : model.state_dict(),'map':0}
            torch.save(data,os.path.join(output_dir, "model_best.pth"))








        
if __name__ == '__main__':
    opt = parse_config()  # 得参
    main(opt)  # 传参














