from utils.teacher_model import LanGuideMedSeg_teacher
from utils.student_model import LanGuideMedSeg_student
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
from einops import rearrange, repeat
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class LIA(nn.Module):
    """
    Cross-attention-based Token-wise Alignment
    """
    def __init__(self,temperature=0.07):
        super(LIA, self).__init__()
        self.temperature = temperature

    def forward(self, original, after_cross_attention):
        """
        original: [B, S, C]
        after_cross_attention: [B, S, C]
        attention_scores:[B, S, L]
        sim_matrix:[B,S,S]
        """
        S = original.size(1)
        original_transposed = original.transpose(1, 2)  # 转置后形状为 [B, C, S]
        sim_matrix = torch.matmul(F.normalize(after_cross_attention, dim=2), F.normalize(original_transposed, dim=1)) # [B, S, S]
        # 计算Cross-attention前后的对比损失（图像）
        loss_original2after = self.contrastive_loss(sim_matrix, S)
        loss_after2original = self.contrastive_loss(sim_matrix.transpose(1, 2), S)
        # 计算最终的LIA损失
        loss_LIA = (loss_original2after + loss_after2original) / 2
        return loss_LIA
    def contrastive_loss(self, sim_matrix, S):
        """
        最大化对应token之间的相似度
        """
        B = sim_matrix.size(0)  # 获取批量大小
        labels = torch.arange(S).repeat(B, 1).to(sim_matrix.device)
        log_prob = F.log_softmax(sim_matrix, dim=2)
        loss = -log_prob.gather(2, labels.unsqueeze(2)).squeeze(2)

        return loss.mean()
class LTA(nn.Module):
    """
    Cross-attention-based Token-wise Alignment
    """
    def __init__(self,temperature=0.07):
        super(LTA, self).__init__()
        self.temperature = temperature

    def forward(self, original, after_cross_attention):
        """
        original: [B, L, C]
        after_cross_attention: [B, L, C]
        attention_scores:[B, L, S]
        sim_matrix:[B, L, L]
        """
        L = original.size(1)
        original_transposed = original.transpose(1, 2)  # 转置后形状为 [B, C, L]
        sim_matrix = torch.matmul(F.normalize(after_cross_attention, dim=2), F.normalize(original_transposed, dim=1)) #[B, L, L]
        # 计算Cross-attention前后的对比损失（文本）
        loss_original2after = self.contrastive_loss(sim_matrix, L)
        loss_after2original = self.contrastive_loss(sim_matrix.transpose(1, 2), L)
        # 计算最终的LTA损失
        loss_LTA = (loss_original2after + loss_after2original) / 2
        return loss_LTA
    def contrastive_loss(self, sim_matrix, L):
        """
        最大化对应token之间的相似度
        """
        B = sim_matrix.size(0)  # 获取批量大小
        labels = torch.arange(L).repeat(B, 1).to(sim_matrix.device)
        log_prob = F.log_softmax(sim_matrix, dim=2)
        loss = -log_prob.gather(2, labels.unsqueeze(2)).squeeze(2)

        return loss.mean()

class CTA(nn.Module):
    """
    (LIA+LTA)/2
    """
    def __init__(self,temperature=0.07):
        super(CTA,self).__init__()
        self.LIA = LIA(temperature=temperature)
        self.LTA = LTA(temperature=temperature)

    def forward(self, img_original, img_after, txt_original, txt_after):
        loss_LIA = self.LIA(img_original, img_after)
        loss_LTA = self.LTA(txt_original, txt_after)
        return (loss_LTA + loss_LIA)/2


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1):
        super(CriterionKD, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft):
        B, C, h, w = soft.size()
        scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return loss

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap

class CriterionCWD(nn.Module):

    def __init__(self, norm_type='channel', divergence='mse', temperature=1.0):

        super(CriterionCWD, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self, preds_S, preds_T):
        if len(preds_S.shape) == 3:
            n, wh, c = preds_S.shape
            preds_S = preds_S.reshape(n, c, int(torch.sqrt(torch.tensor(wh)).item()),
                                      int(torch.sqrt(torch.tensor(wh)).item()))
            preds_T = preds_T.reshape(n, c, int(torch.sqrt(torch.tensor(wh)).item()),
                                      int(torch.sqrt(torch.tensor(wh)).item()))
        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S
            norm_t = preds_T.detach()

        if self.divergence == 'kl':
            norm_s = (norm_s + 1e-8).log()
        loss = self.criterion(norm_s, norm_t)

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)

class CriterionSKD(nn.Module):
    """
    preds: [B, N, C]
    targets: [B, N, C]
    """
    def __init__(self, temperature=1):
        super(CriterionSKD, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        preds_norm = F.normalize(preds, p=2, dim=2)
        targets_norm = F.normalize(targets, p=2, dim=2)

        model_similarity = torch.bmm(preds_norm, preds_norm.transpose(1, 2))
        target_similarity = torch.bmm(targets_norm, targets_norm.transpose(1, 2))

        loss = torch.norm((model_similarity-target_similarity), p=2, dim=(1,2)).mean() * (self.temperature**2)#/ preds_norm.shape[1]
        return loss

class CriterionL2(nn.Module):
    """
    preds: [B, N, C]
    targets: [B, N, C]
    """
    def __init__(self, temperature=1):
        super(CriterionL2, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        preds_norm = F.normalize(preds, p=2, dim=2)
        targets_norm = F.normalize(targets, p=2, dim=2)

        loss = torch.norm((preds_norm-targets_norm), p=2, dim=(1,2)).mean() * (self.temperature**2) #/ preds_norm.shape[1]
        return loss
class CriterionL1(nn.Module):
    """
    preds: [B, N, C]
    targets: [B, N, C]
    """
    def __init__(self, temperature=1):
        super(CriterionL1, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        preds_norm = F.normalize(preds, p=1, dim=2)
        targets_norm = F.normalize(targets, p=1, dim=2)

        loss = torch.norm((preds_norm-targets_norm), p=1, dim=(1,2)).mean() * (self.temperature**2) #/ preds_norm.shape[1]
        return loss
class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(LanGuideMedSegWrapper, self).__init__()
        
        self.model_teacher = LanGuideMedSeg_teacher(args.bert_type, args.vision_type, args.project_dim)
        self.model_student = LanGuideMedSeg_student(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}

        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.loss_kl = CriterionKD(temperature=1.0) # nn.KLDivLoss(reduction='none')
        self.loss_CKD = CriterionCWD('channel', 'kl', 1.0) # channel Kl
        self.loss_SKD = CriterionSKD(temperature=1.0) # Best set is 10.0
        self.loss_L2 = CriterionL2(temperature=1.0) # Best set is 1.0
        self.loss_L1 = CriterionL1(temperature=1.0)

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        # 定义教师网络的训练、验证和测试指标
        self.train_metrics_teacher = nn.ModuleDict(metrics_dict)
        self.val_metrics_teacher = deepcopy(self.train_metrics_teacher)
        self.test_metrics_teacher = deepcopy(self.train_metrics_teacher)
        # 定义学生网络的训练、验证和测试指标
        self.train_metrics_student = deepcopy(self.train_metrics_teacher)
        self.val_metrics_student = deepcopy(self.train_metrics_teacher)
        self.test_metrics_student = deepcopy(self.train_metrics_teacher)
        
        self.save_hyperparameters()


    def loss_sd(self, student_feature_maps, teacher_feature_maps):
        """
        IMD蒸馏
        """
        # 确保学生和导师的特征列表长度相同
        assert len(student_feature_maps) == len(teacher_feature_maps)
        # 初始化蒸馏损失
        distillation_loss = 0.0
        # 迭代每一对特征图
        for student_feature_map, teacher_feature_map in zip(student_feature_maps, teacher_feature_maps):
            student_feature_map_sum = student_feature_map.sum(dim=2) # student/teacher_feature_map: [B,N,C] -> [B, N]
            teacher_feature_map_sum = teacher_feature_map.sum(dim=2)
            student_norm = F.normalize(student_feature_map_sum, p=2, dim=1)
            teacher_norm = F.normalize(teacher_feature_map_sum, p=2, dim=1)
            distillation_loss += torch.norm((student_norm - teacher_norm), p=2, dim=1) # distillation_loss: [B,N] -> [B,]
        return torch.mean(distillation_loss) / len(student_feature_maps) # 对批量求平均


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model_student.parameters(), lr = self.lr) #lr = self.hparams.lr
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model_student.forward(x)


    def shared_step(self,batch,batch_idx):
        x, y, name = batch
        outputs_teacher, *alignment_teacher = self.model_teacher(x)
        outputs_student, *alignment_student = self.model_student(x)
        preds_student = outputs_student
        preds_teacher = outputs_teacher
        student_feature_maps = alignment_student
        teacher_feature_maps = alignment_teacher
        loss = self.loss_fn(preds_student['out_logit'], y) \
               + 0.5*(self.loss_L1(student_feature_maps[0], teacher_feature_maps[0])\
               + self.loss_L1(student_feature_maps[1], teacher_feature_maps[1])\
               + self.loss_L1(student_feature_maps[2], teacher_feature_maps[2]))\
               # + 0.5*(self.loss_L2(student_feature_maps[0], teacher_feature_maps[0])\
               # + self.loss_L2(student_feature_maps[1], teacher_feature_maps[1])\
               # + self.loss_L2(student_feature_maps[2], teacher_feature_maps[2]))\
               # + self.loss_kl(preds_student['out_logit'], preds_teacher['out_logit']) \
               # + self.loss_CKD(student_feature_maps[0], student_feature_maps[0]) \
               # + self.loss_CKD(student_feature_maps[1], teacher_feature_maps[1]) \
               # + self.loss_CKD(student_feature_maps[2], teacher_feature_maps[2])\
               # + self.loss_SKD(student_feature_maps[0], teacher_feature_maps[0])\
               # + self.loss_SKD(student_feature_maps[1], teacher_feature_maps[1])\
               # + self.loss_SKD(student_feature_maps[2], teacher_feature_maps[2])\
               # + self.loss_sd(student_feature_maps, teacher_feature_maps)\
               # + self.loss_kl(F.log_softmax(preds_student['out_logit'], dim=1),
               #                  F.softmax(preds_teacher['out_logit'].detach(), dim=1))\


        return {'loss': loss, 'preds_student': preds_student['out'].detach(),
                'preds_teacher': preds_teacher['out'].detach(), 'mask': y.detach()}
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        if stage == "train":
            metrics_teacher = self.train_metrics_teacher
            metrics_student = self.train_metrics_student
        elif stage == "val":
            metrics_teacher = self.val_metrics_teacher
            metrics_student = self.val_metrics_student
        else:
            metrics_teacher = self.test_metrics_teacher
            metrics_student = self.test_metrics_student

        preds_teacher = outputs['preds_teacher'][:, 1, :, :].unsqueeze(dim=1)
        preds_student = outputs['preds_student'][:, 1, :, :].unsqueeze(dim=1)

        # 用于记录教师网络的精度指标
        for name in metrics_teacher:
            step_metric = metrics_teacher[name](preds_teacher, outputs['mask']).item()
        # 用于记录学生网络的精度指标
        for name in metrics_student:
            step_metric = metrics_student[name](preds_student, outputs['mask']).item()
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        if stage == "train":
            metrics_teacher = self.train_metrics_teacher
            metrics_student = self.train_metrics_student
        elif stage == "val":
            metrics_teacher = self.val_metrics_teacher
            metrics_student = self.val_metrics_student
        else:
            metrics_teacher = self.test_metrics_teacher
            metrics_student = self.test_metrics_student

        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage + "_loss").replace('train_', '')] for t in outputs])).item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics_student:
            epoch_metric_student = metrics_student[name].compute().item()
            metrics_student[name].reset()
            dic["student" + "_" + stage + "_" + name] = epoch_metric_student

        for name in metrics_teacher:
            epoch_metric_teacher = metrics_teacher[name].compute().item()
            metrics_teacher[name].reset()
            dic["teacher" + "_" + stage + "_" + name] = epoch_metric_teacher
        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)