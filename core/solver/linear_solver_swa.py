import argparse
from asyncio.log import logger
from logging import raiseExceptions
import os
import pprint
import json
import time
import datetime

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from core.utils import dist as link
import random
import numpy as np

from tensorboardX import SummaryWriter
from easydict import EasyDict
from core.solver.ssl_solver import SSLSolver
from core.model import model_entry
from core.utils.dist import dist_init, broadcast_object
from core.utils.misc import (count_params, count_flops, load_state_model, AverageMeter, load_state_optimizer,
                             accuracy, makedir, create_logger, get_logger, modify_state)
from core.optimizer import optim_entry
from core.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

class LinearImageNetSolverSWA(SSLSolver):
    def setup_env(self):
        # # >>> dist
        # self.dist = EasyDict()
        self.dist = None
        # self.dist.rank, self.dist.world_size, self.dist.local_id = link.get_rank(), link.get_world_size(), link.get_local_rank()
        # self.prototype_info.world_size = self.dist.world_size

        # >>> directories
        self.path = EasyDict()
        # self.path.root_path = os.path.dirname(self.config_file)
        self.path.root_path = self.config.get('root_path' ,os.path.dirname(self.config_file))
        data_split = os.path.basename(self.config['data']['train']['meta_file']).split('.txt')[0]
        prune_strategy = os.path.basename(os.path.abspath(os.path.join(self.config['data']['train']['meta_file'],"..")))
        trunk_name = self.config['saver']['pretrain']['path']

        trunk_name = os.path.basename(os.path.abspath(os.path.join(trunk_name,"../..")))
        self.path.root_path = os.path.join(self.path.root_path, f"{prune_strategy}_{data_split}_{trunk_name}")

        ## REMOVE assertion only for SWA case
        # assert not os.path.exists(self.path.root_path), f'{self.path.root_path} path already exist'


        os.path.dirname(self.config_file) != self.path.root_path and os.makedirs(self.path.root_path, exist_ok=True)

        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints_finetune_swa')
        self.path.event_path = os.path.join(self.path.root_path, 'events_finetune_swa')
        self.path.result_path = os.path.join(self.path.root_path, 'results_finetune_swa')
        
        os.makedirs(self.path.save_path, exist_ok=True)
        os.makedirs(self.path.event_path, exist_ok=True)
        os.makedirs(self.path.result_path, exist_ok=True)

        
        self.tb_logger = SummaryWriter(self.path.event_path)

        # >>> logger
        create_logger(os.path.join(self.path.root_path, 'log_finetune_swa.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        self.logger.info(f'root path {self.path}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")

        # >>> load pretrain checkpoint
        try:
            self.logger.info('======= Looking for local finetune pretrain... =======')
            local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints_finetune', 'ckpt.pth')
            self.state = torch.load(local_ft_ckpt, 'cpu')
            self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
        except:
            self.logger.info('======= Local finetune pretrain NOT FOUND =======')
            try:
                self.logger.info('======= Looking for local pretrain... =======')
                self.logger.info(f'======= Looking for local pretrain at {self.path.root_path}... =======')
                local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints', 'ckpt.pth')
                self.state = torch.load(local_ft_ckpt, 'cpu')
                self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
            except:
                self.logger.info('======= local pretrain NOT FOUND =======')
                self.logger.info('======= Looking for pretrain... =======')
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)

            state_dict = self.state['model']

            for k in list(state_dict.keys()):
                if 'backbone' in k and ('fc' not in k or self.config.model.type.startswith("vit")
                                        or self.config.model.type.startswith("swin")):  # rename & clean loaded keys
                    state_dict[k[len("module.backbone."):]] = state_dict[k]  # remove module.backbone.
                del state_dict[k]
            self.state = {'model': state_dict, 'last_iter': 0}

        # >>> seed initialization
        seed = self.config.get('seed', 233)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # >>> reproducibility config    note: deterministic would slow down training
        if self.config.get('strict_reproduceable', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # init the fc layer

        if self.config.model.type.startswith("vit") or self.config.model.type.startswith("swin"):
            self.model.head.weight.data.normal_(mean=0.0, std=0.01)
            self.model.head.bias.data.zero_()
        elif self.config.model.type.startswith("resnet"):
            self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.model.fc.bias.data.zero_()
        else:
            raise NotImplementedError

        self.model.cuda()
        self.prototype_info.model = self.config.model.type

        count_params(self.model)
        count_flops(self.model, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

        # self.model = torch.nn.parallel.DistributedDataParallel(self.model,
        #                                                        device_ids=[self.dist.local_id],
        #                                                        output_device=self.dist.local_id,
        #                                                        find_unused_parameters=True)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        list_model_state_dict_keys = []
        for key in list(self.model.module.state_dict().keys()):
            if 'tracked' not in key:
                list_model_state_dict_keys.append(key)

        list_pretrain_state_dict_keys = []
        for key in list(self.state['model'].keys()):
            if 'tracked' not in key:
                list_pretrain_state_dict_keys.append(key)

        for k_state, k_model in zip(list_pretrain_state_dict_keys, list_model_state_dict_keys):
            if k_state != k_model:
                self.state['model'][k_model] = self.state['model'][k_state]
                del self.state['model'][k_state]
                self.logger.info(f"{k_state} ==> {k_model}, del {k_state}")

        load_state_model(self.model.module, self.state['model'])

    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        opt_config.kwargs.params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.config.get('bn_fc', False):
            assert len(opt_config.kwargs.params) == 4  # fc.weight, fc.bias, bn.weight, bn.bias
        else:
            assert len(opt_config.kwargs.params) == 2  # fc.weight, fc.bias

        self.optimizer = optim_entry(opt_config)

        ## IMPORTANT TO SET IT 0; SO WE DON"T CONTINUE ANY TRAINING
        self.state['last_iter'] = 0
        print("optimizer builded!")

        # WE DONT NEED PREVIOUS STATE FOR SWA
        # if 'optimizer' in self.state:
        #     load_state_optimizer(self.optimizer, self.state['optimizer'])

    def get_dataloader(self, dataset:torch.utils.data.Dataset, batch_size:int=64, n_workers:int=1, isShuffle:bool=True):

        assert isinstance(dataset, torch.utils.data.Dataset),f"Expected dataset to be of type `torch.utils.data.Dataset` but got {type(dataset)}"
        assert isinstance(batch_size, int),f"Expected batch_size to be of type `int` but got {type(batch_size)}"
        assert isinstance(n_workers, int),f"Expected n_workers to be of type `int` but got {type(n_workers)}"
        assert isinstance(isShuffle, bool),f"Expected isShuffle to be of type `bool` but got {type(isShuffle)}"
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle, num_workers=n_workers, drop_last=isShuffle)
        
        return dataloader

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.type == 'imagenet':
            train_dataset = build_imagenet_train_dataloader(self.config.data, return_dataset=True)
            val_dataset   = build_imagenet_test_dataloader(self.config.data, return_dataset=True)

            train_dataloader = self.get_dataloader(train_dataset, batch_size=self.config.data.batch_size, n_workers=self.config.data.num_workers, isShuffle=True)
            val_dataloader   = self.get_dataloader(val_dataset, batch_size=self.config.data.batch_size, n_workers=self.config.data.num_workers, isShuffle=False)

            self.train_data = {'type': 'train', 'loader': train_dataloader}
            self.val_data   = {'type': 'test', 'loader': val_dataloader}

            # self.train_data = build_imagenet_train_dataloader(self.config.data)
            # self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            raise NotImplementedError

    def pre_train(self):
        # super(LinearImageNetSolverSWA, self).pre_train() ## donot call super().pre_train as it only supports ddp
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)
        self.meters.losses = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)

        self.model.train()

        self.criterion = None
        self.meters.top1 = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq, isdistributed=self.dist is not None)

        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes

        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.pre_train()
        best_prec1 = 0
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        self.model.train()
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.model.module.fc[0].train()
        if self.config.get('bn_112_stat', False):
            self.model.module.bn1.change_current_res(112)

        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            target = batch['label']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda()

            logits = self.model(input)

            loss = self.criterion(logits, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone() / self.dist.world_size
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            self.optimizer.zero_grad()

            loss.backward()
            dist.barrier()
            self.optimizer.step()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                curr_epoch = (curr_step - 1) // self.config.data.iter_per_epoch + 1
                log_msg = f'Epoch: [{curr_epoch}/{self.config.data.max_epoch}]\t' \
                          f'Iter: [{curr_step}/{total_step}]|[{curr_step - (curr_epoch - 1) * self.config.data.iter_per_epoch}/{self.config.data.iter_per_epoch}]\t' \
                          f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                          f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                          f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                          f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                          f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                          f'LR {current_lr:.4f}\t' \
                          f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            if (curr_step % self.config.saver.val_freq == 0 or curr_step == total_step) and curr_step > 0:
                metrics = self.evaluate()
                best_prec1 = max(metrics.metric['top1'], best_prec1)
                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)
                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    torch.save(self.state, ckpt_name)
                    self.sanity_check(self.model.module.state_dict(), self.state['model'])

            end = time.time()
        self.logger.info('best acc:' + str(best_prec1) + '\n')

    @torch.no_grad()
    def custom_update_bn(self,loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be transferred to
                :attr:`device` before being passed into :attr:`model`.
        Example:
            >>> # xdoctest: +SKIP("Undefined variables")
            >>> loader, model = ...
            >>> torch.optim.swa_utils.update_bn(loader, model)
        .. note::
            The `update_bn` utility assumes that each data batch in :attr:`loader`
            is either a tensor or a list or tuple of tensors; in the latter case it
            is assumed that :meth:`model.forward()` should be called on the first
            element of the list or tuple corresponding to the data batch.
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0


        for input in tqdm(loader, desc="BN Stats"):
           
            # if isinstance(input, (list, tuple)):
            #     input = input[0]
            # if device is not None:
            #     input = input.to(device)

            model(input["image"].cuda())

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    def swa_train(self):
        self.pre_train()
        best_prec1 = 0
        total_step = len(self.train_data['loader'])
        start_step = 0
        end = time.time()

        self.model.train()

        # ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        # 0.1 * averaged_model_parameter + 0.9 * model_parameter
        # self.swa_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)
        self.swa_model = AveragedModel(self.model)
        self.swa_lr    = self.config.lr_scheduler.kwargs.base_lr
        self.logger.info(f"SWALR scheduler initialized with LR: {self.swa_lr}")
        # default annealing strategy is cosine
        swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)

        
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.model.module.fc[0].train()
        if self.config.get('bn_112_stat', False):
            self.model.module.bn1.change_current_res(112)

        total_epochs = self.config.data.max_epoch
        curr_step = start_step
        for curr_epoch in range(total_epochs):
            for i, batch in enumerate(tqdm(self.train_data['loader'], desc=f"Training Epoch #{curr_epoch}")):
                input = batch['image']
                target = batch['label']

                curr_step += 1
                
                # self.lr_scheduler.step(curr_step)
                # lr_scheduler.get_lr()[0] is the main lr

                current_lr = swa_scheduler.get_last_lr()[0]
                # measure data loading time
                self.meters.data_time.update(time.time() - end)
                # transfer input to gpu
                target = target.squeeze().cuda().long()
                input = input.cuda()

                logits = self.model(input)

                loss = self.criterion(logits, target)
                # # measure accuracy and record loss
                prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

                # reduced_loss = loss.clone() / self.dist.world_size
                # reduced_prec1 = prec1.clone() / self.dist.world_size
                # reduced_prec5 = prec5.clone() / self.dist.world_size

                self.meters.losses.reduce_update(loss.clone())
                self.meters.top1.reduce_update(prec1.clone())
                self.meters.top5.reduce_update(prec5.clone())

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()


                self.swa_model.update_parameters(self.model)
                swa_scheduler.step()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)
                if curr_step % self.config.saver.print_freq == 0:# and self.dist.rank == 0:
                    self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                    self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)
                    remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    # curr_epoch = (curr_step - 1) // self.config.data.iter_per_epoch + 1
                    log_msg = f'Epoch: [{curr_epoch}/{self.config.data.max_epoch}]\t' \
                            f'Iter: [{curr_step}/{total_step}]\t' \
                            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                            f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                            f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                            f'LR {current_lr:.4f}\t' \
                            f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.info(log_msg)

                

                end = time.time()
        
            # if (curr_step % self.config.saver.val_freq == 0 or curr_step == total_step) and curr_step > 0:
            ##########################################
            ### PERFORM EVALUATION AFTER EVERY EPOCH
            #########################################
            ## Update SWA model BN
            # self.logger.info("Calculating Batchnorm stats of SWA model")
            # self.custom_update_bn(self.train_data['loader'], self.swa_model)
            # self.logger.info("BN stats updated for SWA model successfully..!!")

            metrics = self.swa_evaluate()
            best_prec1 = max(metrics.metric['top1'], best_prec1)
            # testing logger
            # if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
            if self.config.data.test.evaluator.type == 'imagenet':
                metric_key = 'top{}'.format(self.topk)
                self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)
            # save ckpt
            # if self.dist.rank == 0:
            if self.config.saver.save_many:
                ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth'
            else:
                ckpt_name = f'{self.path.save_path}/ckpt.pth'
            self.state['swa_model'] = self.swa_model.state_dict()
            self.state['optimizer'] = self.optimizer.state_dict()
            self.state['last_iter'] = curr_step
            torch.save(self.state, ckpt_name)
            
            self.sanity_check(self.model.module.state_dict(), self.state['model'])
            self.sanity_check(self.swa_model.module.state_dict(), self.state['model'])
        self.logger.info('best acc:' + str(best_prec1) + '\n')

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        res_dict = {}
        res_dict['score'] = []
        res_dict['label'] = []
        for batch_idx, batch in enumerate(tqdm(self.val_data['loader'], desc="Evaluating over validation set")):
            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)

            curr_preds = self.val_data['loader'].dataset.tensor2numpy(preds)
            curr_scores = self.val_data['loader'].dataset.tensor2numpy(scores)
            curr_labels = self.val_data['loader'].dataset.tensor2numpy(batch['label'])

            for _idx in range(curr_preds.shape[0]):
                res_dict['score'].append([float('%.8f' % s) for s in curr_scores[_idx]])
                res_dict['label'] = res_dict['label'] + [int(curr_labels[_idx])]

        metrics = self.val_data['loader'].dataset.evaluator.eval_from_dict(res_dict)

        self.logger.info(json.dumps(metrics.metric, indent=2))
        
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.model.module.fc[0].train()
        self.model.train()

        res_dict = None # Free memory

        return metrics

    @torch.no_grad()
    def swa_evaluate(self):
        self.swa_model.eval()
        res_dict = {}
        res_dict['score'] = []
        res_dict['label'] = []
        for batch_idx, batch in enumerate(tqdm(self.val_data['loader'], desc="Evaluating over validation set")):
            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits = self.swa_model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)

            curr_preds = self.val_data['loader'].dataset.tensor2numpy(preds)
            curr_scores = self.val_data['loader'].dataset.tensor2numpy(scores)
            curr_labels = self.val_data['loader'].dataset.tensor2numpy(batch['label'])

            for _idx in range(curr_preds.shape[0]):
                res_dict['score'].append([float('%.8f' % s) for s in curr_scores[_idx]])
                res_dict['label'] = res_dict['label'] + [int(curr_labels[_idx])]

        metrics = self.val_data['loader'].dataset.evaluator.eval_from_dict(res_dict)

        self.logger.info(json.dumps(metrics.metric, indent=2))
        
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.swa_model.module.fc[0].train()
        self.swa_model.train()

        res_dict = None # Free memory

        return metrics


    def clean_up_files(self):
        import glob, os
        paths = glob.glob(f"{self.path.result_path}/*")
        print(paths)
        for path in paths:
            print(path)
            os.remove(path)

    def test(self):
        self.pre_train()
        metrics = self.evaluate()
        self.logger.info('acc:' + str(metrics.metric['top1']) + '\n')
        
    def sanity_check(self, state_dict, pretrained_weights):
        """
        Linear classifier should not change any weights other than the linear layer.
        This sanity check asserts nothing wrong happens (e.g., BN stats updated).
        """
        # print("=> loading '{}' for sanity check".format(pretrained_weights))
        # checkpoint = torch.load(pretrained_weights, map_location="cpu")
        # state_dict_pre = checkpoint['model']

        list_model_state_dict_keys = []
        for key in list(state_dict.keys()):
            if 'tracked' not in key:
                list_model_state_dict_keys.append(key)

        list_pretrain_state_dict_keys = []
        for key in list(pretrained_weights.keys()):
            if 'tracked' not in key:
                list_pretrain_state_dict_keys.append(key)

        for k, k_pre in zip(list_model_state_dict_keys, list_pretrain_state_dict_keys):
            # only ignore fc layer
            if 'fc.weight' in k or 'fc.bias' in k:
                continue
            if 'num_batches_tracked' not in k:
                continue
            # if (self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False)) and 'fc.' in k:
            if self.config.get('bn_fc', False) and 'fc.' in k:
                continue
            # name in pretrained model
            # k_pre = 'module.features.' + k[len('module.'):] \
            #     if k.startswith('module.') else 'module.features.' + k
            assert ((state_dict[k].cpu() == pretrained_weights[k_pre]).all()), \
                '{} is changed in linear classifier training.'.format(k)

        print("=> sanity check passed.")

from core.utils.misc import parse_config

def main():
    parser = argparse.ArgumentParser(description='linear solver')
    parser.add_argument('--config', required=True, type=str)
    # parser.add_argument("--tcp_port", type=str, default="5671")

    args = parser.parse_args()

    # dist_init(port=str(args.tcp_port))
    # build solver
    solver = LinearImageNetSolverSWA(args.config)

    config_file = args.config
    prototype_info = EasyDict()
    config = parse_config(config_file)

    # if solver.config.data.last_iter < solver.config.data.max_iter:
    #     solver.train()
    # else:
    #     solver.logger.info('Training has been completed to max_iter!')
    
    solver.test()
    solver.swa_train()
    print()


if __name__ == '__main__':
    main()
