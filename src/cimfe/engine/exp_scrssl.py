import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

from cimfe.data.datasets import data_provider
from cimfe.engine.exp_basic import Exp_Basic
from cimfe.models.layers.revin import RevIN
from cimfe.models.scrssl import SCRSSL
from cimfe.utils.tools import create_patch, random_masking, transfer_weights

warnings.filterwarnings('ignore')


class Exp_SCRSSL(Exp_Basic):
    """Self-supervised pretraining of the SCRSSL patch encoder (masked reconstruction
    + optional PSI auxiliary task). Finetuning the encoder directly is out of scope here;
    the pretrained encoder is instead used to extract embeddings (see baselines.tables)."""

    def __init__(self, args):
        super(Exp_SCRSSL, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        normalizer = RevIN(6, affine=False)

        model = SCRSSL(
            c_in=6, patch_len=self.args.patch_len, num_patch=self.args.num_patch,
            n_layers=self.args.e_layers, n_heads=self.args.n_heads, d_model=self.args.d_model, shared_embedding=True, d_ff=self.args.d_ff,
            dropout=self.args.dropout, head_dropout=self.args.head_dropout, y_range=(0, 2),
            pre_norm=self.args.pre_norm, norm=self.args.norm
        ).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model, normalizer

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.pretrain_learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def criterion_mask_reconstruct(self, preds, target, mask=None):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def pretrain(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, setting, str(self.args.seed))
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                      T_max=self.args.pretrain_epochs * len(train_loader))
        self.criterion = self._select_criterion()

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.pretrain_epochs):
            start_time = time.time()

            train_loss, train_recloss, train_contloss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler, train=True)
            vali_loss, vali_recloss, vali_contloss = self.pretrain_one_epoch(vali_loader, model_optim, model_scheduler, train=False)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} ({4:.4f}, {5:.4f}) Val Loss: {6:.4f} ({7:.4f}, {8:.4f})"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time,
                        train_loss, train_recloss, train_contloss, vali_loss, vali_recloss, vali_contloss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'vali_loss': vali_loss,
            }

            self.writer.add_scalars(f"pretrain_loss/{setting}/", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'W_' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'W_' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, data_loader, model_optim, model_scheduler, train=True):

        total_loss = []
        total_recloss = []
        total_taskloss = []

        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train):
            for i, (batch_x, psi, subj) in enumerate(data_loader):

                batch_x = batch_x.float().to(self.device)
                normed_x = self.normalizer(batch_x, 'norm')
                normed_x = create_patch(normed_x, self.args.patch_len, self.args.stride)
                masked_x, _, mask, _ = random_masking(normed_x, self.args.mask_ratio)
                psi = psi.float().to(self.device)

                z, pred, pred_psi = self.model(masked_x)
                recloss = self.criterion_mask_reconstruct(pred, normed_x, mask)

                if self.args.pretrain_task == 'rec':
                    task_loss = torch.tensor([0]).to(self.device)
                else:
                    task_loss = self.criterion(psi, pred_psi)
                loss = recloss + self.args.alpha * task_loss

                if train:
                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()
                    model_scheduler.step()

                # record
                total_loss.append(loss.item())
                total_recloss.append(recloss.item())
                total_taskloss.append(task_loss.item())

        total_loss = np.average(total_loss)
        total_recloss = np.average(total_recloss)
        total_taskloss = np.average(total_taskloss)

        self.model.train()
        return total_loss, total_recloss, total_taskloss
