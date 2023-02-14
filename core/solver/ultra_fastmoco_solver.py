from asyncio.log import logger
import copy
import argparse
import os
import pprint

import time
import datetime
from .prune_utils import *
# import torch.distributed.ReduceOp  # delete me
import random


import torch
import torch.nn as nn
import torch.distributed as dist
from core.utils import dist as link

from core.lr_scheduler import scheduler_entry
from core.solver.ssl_solver import SSLSolver
from core.model import model_entry
from core.utils import unsupervised_entry
from core.utils.dist import dist_init
from core.utils.misc import count_params, parse_config, count_flops, load_state_model, AverageMeter, load_state_optimizer, makedir, create_logger, get_logger, modify_state, removedir
# from core.utils.misc import AverageMeter, parse_config, makedir, create_logger, get_logger, modify_state
from easydict import EasyDict
from tensorboardX import SummaryWriter

from core.optimizer import optim_entry
from core.data import build_imagenet_train_dataloader
from core.loss_functions import loss_entry
import numpy as np

class ImageNetSolver(SSLSolver):
    def __init__(self, config_file):
        self.config_file = config_file
        

        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)

        self.prune_epochs = self.config.prune.prune_epochs
        self.max_epochs = self.config.prune.max_epochs
        self.min_k = self.config.prune.min_k
        self.prune_style = self.config.prune.prune_style
        self.prune_strategy = self.config.prune.strategy
        self.skip_epochs = self.config.prune.skip_epochs
        # skip_epochs
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        # self.build_data()
        load_data_for_epochs = self.max_epochs if self.prune_style.lower()=="weighted_loss" else self.prune_epochs+self.config.prune.skip_epochs

        self.build_pruned_data(load_data_for_epochs)
        self.total_step = (self.max_epochs//(load_data_for_epochs))*len(self.train_data['loader'])

        self.build_lr_scheduler(self.total_step)
        self.weights = None

    def build_lr_scheduler(self, max_iter):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        # if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
        self.config.lr_scheduler.kwargs.max_iter = max_iter#self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = 0#self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def setup_env(self):
        # >>> dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size, self.dist.local_id = link.get_rank(), link.get_world_size(), link.get_local_rank()
        self.prototype_info.world_size = self.dist.world_size

        # >>> directories
        self.path = EasyDict()
        # self.path.root_path = os.path.dirname(self.config_file)
        self.path.root_path = self.config.get('root_path' ,os.path.dirname(self.config_file))

        self.path.root_path = os.path.join(self.path.root_path, f"{self.config.prune.prune_style}_{self.config.prune.prune_epochs}_{self.config.prune.strategy}_{self.config.prune.min_k}_{self.config.prune.weight_easy}_{self.config.prune.weight_hard}")

        os.path.dirname(self.config_file) != self.path.root_path and removedir(self.path.root_path)
        os.path.dirname(self.config_file) != self.path.root_path and makedir(self.path.root_path)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        # >>> tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)

        # >>> logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")

        # >>> load pretrain checkpoint
        try:
            self.logger.info('======= Looking for local pretrain... =======')
            local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints', 'ckpt.pth')
            self.state = torch.load(local_ft_ckpt, 'cpu')
            self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        except:
            self.logger.info('======= local pretrain NOT FOUND =======')
            try:
                self.logger.info('======= Looking for pretrain... =======')
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
                if hasattr(self.config.saver.pretrain, 'ignore'):
                    self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            except:
                self.logger.info('======= pretrain NOT FOUND =======')
                self.state = {}
                self.state['last_iter'] = 0

        # >>> seed initialization
        seed = self.config.get('seed', 233)
        if self.config.get('dist_seed', False):
            seed += self.dist.rank
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

        if not self.config.model.kwargs.get('bn', False):
            self.config.model.kwargs.bn = copy.deepcopy(
                self.config.backbone.kwargs.bn)  # note: prevent anticipated inplace edit in model_entry.
            if self.config.model.kwargs.get('ema', False):
                bb_config_back = copy.deepcopy(self.config.backbone)
        else:
            if self.config.model.kwargs.get('ema', False):
                bb_config_back = copy.deepcopy(self.config.backbone)

        self.config.model.kwargs.backbone = model_entry(self.config.backbone)
        if self.config.model.kwargs.get('ema', False):
            if bb_config_back.kwargs.get('img_size', False):
                bb_config_back.kwargs.img_size = 224
            if bb_config_back.type in ['swin_t'] or bb_config_back.type.startswith("vit"):
                bb_config_back.kwargs.is_teacher = True

            self.config.model.kwargs.backbone_ema = model_entry(bb_config_back)

        self.model = unsupervised_entry(self.config.model)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        self.prototype_info.model = f"{self.config.model.type} + {self.config.backbone.type}"

        count_params(self.model.encoder)
        count_flops(self.model.encoder, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                               device_ids=[self.dist.local_id],
                                                               output_device=self.dist.local_id,
                                                               find_unused_parameters=True)

        
        # import pdb; pdb.set_trace()
        self.logger.info(self.dist.local_id)
        self.logger.info("ids")
        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])
        
        # import pdb; 

    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        param_group = [
            {
                "params": [p for n, p in self.model.named_parameters() if not ("predictor" in n) and p.requires_grad],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if ("predictor" in n) and p.requires_grad],
            }
        ]

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

    def build_pruned_data(self, prune_epochs=None, prune_iter=None, meta_file=None):

        if self.config.data.get('max_epoch', False):
            self.config.data.pop('max_epoch')
        
        if self.config.data.get('max_iter', False):
            self.config.data.pop('max_iter')
        # self.config.data.pop('max_iter')

        self.config.data.last_iter = 0#self.state['last_iter']
        # if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
        if prune_epochs:
            self.config.data.max_epoch = prune_epochs#self.config.lr_scheduler.kwargs.max_iter
        
        if prune_iter:
            self.config.data.max_iter = prune_iter#self.config.lr_scheduler.kwargs.max_iter
        
        if meta_file:
            self.config.data.train.meta_file = meta_file#self.config.lr_scheduler.kwargs.max_iter
        # else:
        #     self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        self.logger.info(f"loading new data from {self.config.data}")
        if self.config.data.type == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            raise NotImplementedError
        


    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.type == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            raise NotImplementedError

    def pre_train(self):
        super(ImageNetSolver, self).pre_train()
        self.meters.distance_z = AverageMeter(self.config.saver.print_freq)
        self.criterion = loss_entry(self.config.criterion)

    # def get_total_epochs(self):
    def save_ckpt(self, curr_step, curr_epoch):
        ckpt_name = f'{self.path.save_path}/ckpt_{curr_epoch}.pth'
        ckpt_name_latest = f'{self.path.save_path}/ckpt_latest.pth'
        self.state['model'] = self.model.state_dict()
        self.state['optimizer'] = self.optimizer.state_dict()
        self.state['last_iter'] = curr_step
        torch.save(self.state, ckpt_name)
        torch.save(self.state, ckpt_name_latest)
    
    def get_weights(self, keys):
        batch_weight = torch.ones(len(keys))
        if self.weights == None:
            return batch_weight
        
        # import pdb; pdb.set_trace()

        for i, key in enumerate(keys):
            key = key.replace(f'{self.config.data.train.root_dir}', '')
            batch_weight[i] = self.weights[key]
        

        return batch_weight

    def update_weights(self, selected_keys_path=None):
        # self.weights = self.config.data.train.meta_file
        # with 
        filetrain = open(f'{self.config.data.train.meta_file}', 'r')
        lines = filetrain.readlines()
        filetrain.close()

        self.weights = {line.split(' ')[0]: self.config.prune.weight_easy  for line in lines}
        if selected_keys_path:
            selected_keys_file = open(f'{selected_keys_path}', 'r')
            selected_lines = selected_keys_file.readlines()
            selected_keys_file.close()
            for key in selected_lines:
                self.weights[key.strip()] = self.config.prune.weight_hard

        
        self.logger.info(f"total hard exm ({len(selected_lines)/len(lines) * 100 }) {len(selected_lines)} out of {len(lines)}")


    def gather_prune(self, curr_epoch):
        self.logger.info("merging results over ranks")
        merge_ranks(prefix=self.path.result_path, world_size=dist.get_world_size(), start_epoch=curr_epoch-self.prune_epochs+1, end_epoch=curr_epoch)
        self.logger.info("merging results over epoch")
        merge_epochs(prefix=self.path.result_path, start_epoch=curr_epoch-self.prune_epochs+1, end_epoch=curr_epoch)
        prune_keys(strategy=self.prune_strategy, prefix=self.path.result_path, start_epoch=curr_epoch-self.prune_epochs+1, end_epoch=curr_epoch, min_k=self.min_k)
        modify_train_txt(prefix=self.path.result_path, start_epoch=curr_epoch-self.prune_epochs+1, end_epoch=curr_epoch, min_k=self.min_k)

    def train(self):
        self.pre_train()
        curr_step = 1#
        # self.state['last_iter'] + 1
        end = time.time()
        self.prune_stage = 0
        prune_at = ((self.prune_stage + 1) * self.prune_epochs) + self.skip_epochs

        self.skip_epochs = 0 if self.prune_style.lower()=="weighted_loss" else self.skip_epochs
        # next_prune_epoch = self.skip_epochs + self.prune_epochs
        
        # curr_step = 1
        # image_ids = []
        # scores = []
        # cls_preds = []

        output_dict = {
            'filename': [],
            'image_id': [],#int(image_id[_idx]),
            'prediction': [],#int(prediction[_idx]),
            'score': [],#[float('%.8f' % s) for s in score[_idx]],
        }
        while curr_step < self.total_step:
            # curr_stage_steps = 0#len(self.train_data['loader'])
            # start_step = curr_step
            for i, batch in enumerate(self.train_data['loader']):   
                input = batch['image']  # [bs, #channel * 2, h, w]
                # curr_stage_steps = 
                self.lr_scheduler.step(curr_step)

                # lr_scheduler.get_lr()[0] is the main lr
                current_lr, head_lr = self.lr_scheduler.get_lr()[0], self.lr_scheduler.get_lr()[1]
                # measure data loading time
                self.meters.data_time.update(time.time() - end)
                # transfer input to gpu
                input = input.cuda()
                # input -> p1, z1, p2, z2
                output = self.model(input)
                weights = self.get_weights(batch['filename']) if self.prune_style.lower()=="weighted_loss" else None
                loss, p_z_m = self.criterion.forward(*output, weights=weights)
                # --------- positive cls and scores
        # (i - 1) // 5005 + 1
                curr_epoch = ((self.prune_stage * self.prune_epochs) + (0 if self.prune_stage == 0 else self.skip_epochs)) + (i // self.config.data.iter_per_epoch + 1)
                # index_pos = torch.arange(0, input.size(0))
                offset = link.get_rank() * input.size(0)
                index_pos = torch.arange(offset, offset + input.size(0), dtype=torch.long)
                _, preds = p_z_m.data.topk(k=1, dim=1)
                preds = torch.eq(preds.squeeze().data.cpu(), index_pos).type(torch.int)

                scores = torch.diagonal(p_z_m[:, offset: offset +input.size(0)])
                
                output_dict = {'prediction': preds.data.cpu().numpy().tolist(),
                'score': scores.data.cpu().numpy().tolist(),
                'image_id': batch['image_id'],
                'filename': batch['filename'],
                }
                res_file = os.path.join(self.path.result_path, f'train_results.txt.rank_{self.dist.rank}_{curr_epoch}')
                writer = open(res_file, 'a')
                self.train_data['loader'].dataset.dump_train(writer, output_dict)
                writer.close()

                # --------- end of positive cls and scores
            
                loss = loss

                with torch.no_grad():
                    distance_z = self.criterion.cosine_similarity(output[1], output[3])
                reduced_loss, reduced_dist_z = loss.clone() / self.dist.world_size, distance_z.clone() / self.dist.world_size
                self.meters.losses.reduce_update(reduced_loss)
                self.meters.distance_z.reduce_update(reduced_dist_z)

                self.optimizer.zero_grad()
                loss.backward()
                dist.barrier()

                if self.config.get('clip_grad_norm', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)
                if (curr_epoch - 1) == prune_at:
                    # import pdb; pdb.set_trace()
                # and (curr_epoch - 1) % self.prune_epochs == 0:
                    # self.prune_stage+=1
                    if self.dist.rank == 0: self.save_ckpt(curr_step, curr_epoch)
                    if self.prune_style.lower()=="weighted_loss":
                        if self.dist.rank == 0: self.gather_prune(curr_epoch=prune_at)
                        dist.barrier()
                        self.update_weights(f'{self.path.result_path}/selected_keys_{prune_at-self.prune_epochs+1}_{prune_at}_{self.min_k}.txt')
                        # self.build_pruned_data(prune_iter=self.total_step-curr_step, meta_file=f'{self.path.result_path}/train_pruned_{curr_epoch-self.prune_epochs+1}_{curr_epoch}_{self.min_k}.txt')
                    # pass    
                    prune_at += self.prune_epochs 

                    
                if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                    self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar('distance_z_train', self.meters.distance_z.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)
                    self.tb_logger.add_scalar('lr_head', head_lr, curr_step)
                    remain_secs = (self.total_step - curr_step) * self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    # curr_epoch = (curr_step - 1) // self.config.data.iter_per_epoch + 1

                    log_msg = f'Epoch: [{curr_epoch}/100]\t' \
                            f'Iter: [{curr_step}/{self.total_step}]|[{i - (((i // self.config.data.iter_per_epoch + 1) - 1) * self.config.data.iter_per_epoch) + 1}/{self.config.data.iter_per_epoch}]\t' \
                            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                            f'Distance_Z {self.meters.distance_z.val:.4f} ({self.meters.distance_z.avg:.4f})\t' \
                            f'LR {current_lr:.4f}\t' \
                            f'Head LR {head_lr:.4f}\t' \
                            f'Remaining Time {remain_time} ({finish_time})'

                    self.logger.info(log_msg)
                    
                curr_step +=1# start_step + i
                if curr_step >= self.total_step:
                    break
                end = time.time()
                # break
            # curr_step+=len(self.train_data['loader'])
            # self.logger.info(f"out of inner loop at rank {self.dist.rank}")
            
            if curr_step >= self.total_step:
                if self.dist.rank == 0: self.save_ckpt(curr_step, curr_epoch)


                break
            dist.barrier()

            # if 
            # curr_step=9999
            # total_step=10000
            
            if self.dist.rank == 0: self.save_ckpt(curr_step, curr_epoch)
            if self.prune_style.lower()=="once" and curr_epoch >= self.prune_epochs:
                if self.dist.rank == 0: self.gather_prune(curr_epoch=curr_epoch)
                dist.barrier()
                self.build_pruned_data(prune_iter=self.total_step-curr_step, meta_file=f'{self.path.result_path}/train_pruned_{curr_epoch-self.prune_epochs+1}_{curr_epoch}_{self.min_k}.txt')
                # pass    
            elif self.prune_style.lower()=="step":
                # self.gather_prune(curr_epoch=curr_epoch)
                if self.dist.rank == 0: self.gather_prune(curr_epoch=curr_epoch)
                dist.barrier()
                self.build_pruned_data(prune_epochs=self.prune_epochs, meta_file=f'{self.path.result_path}/train_pruned_{curr_epoch-self.prune_epochs+1}_{curr_epoch}_{self.min_k}.txt')

            self.prune_stage+=1
            # self.logger.info(f"out of loops at rank {self.dist.rank}")
            # dist.barrier()
            # self.build_pruned_data(self.prune_epochs)
    
    @torch.no_grad()
    def dump_scores(self, writer, batch):
        self.train_data['loader'].dataset.dump(writer, batch)
        writer.close()

def main():
    parser = argparse.ArgumentParser(description='ssl solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument("--tcp_port", type=str, default="5671")

    args = parser.parse_args()

    dist_init(port=str(args.tcp_port))
    # build solver
    solver = ImageNetSolver(args.config)

    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
