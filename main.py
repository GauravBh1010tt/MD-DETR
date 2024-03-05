
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.utils.data import DataLoader
#from models import build_model
from datetime import timedelta

import utils
import pytorch_lightning as pl
from datasets.coco_eval import CocoEvaluator
from datasets.create_voc_instance import task_info_voc_10
from engine import local_trainer, Evaluator
# from transformers import AutoImageProcessor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything
# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.coco_hug import CocoDetection, task_info_coco, create_task_json
# from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
# from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection 
from transformers_local.models.deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor 

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--new_params', default="", type=str)
    parser.add_argument('--freeze', default="", type=str)
    parser.add_argument('--lr_old', default=2e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--n_classes', default=80, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--eval_epochs', default=2, type=int)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--repo_name', default="SenseTime/deformable-detr", type=str)
   
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--n_gpus', default=4, type=int,
                        help="Number of GPUs available")

    parser.add_argument("--num_imgs_viz", type=int, default=10, help="length of prompt")

    # prompt memory
    parser.add_argument("--use_prompts", type=int, default=1, help="use prompt memory")
    parser.add_argument("--prompt_len", type=int, default=10, help="length of prompt")
    parser.add_argument("--num_prompts", type=int, default=10, help="number of prompts in the pool")
    parser.add_argument("--num_prompt_layers", type=int, default=1, help="num of layers for adding prompts")
    parser.add_argument("--prompt_key", type=int, default=1, help="use learnable prompt key")
    parser.add_argument("--prompt_pool_sz", type=int, default=10, help="size of prompts")

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--prompt_loss_coef', default=1, type=float)
    
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    ## Cont Setup
    parser.add_argument('--n_tasks', default=4, type=int)
    parser.add_argument('--lambda_query', default=0, type=float)
    parser.add_argument('--local_query', default=0, type=int)
    parser.add_argument('--start_task', default=1, type=int)
    parser.add_argument('--task_id', default=0, type=int)
    parser.add_argument('--reset_optim', default=1, type=int)
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--mask_gradients', default=1, type=int)

    parser.add_argument('--checkpoint_dir', default='', help='initialized from the pre-training model')
    parser.add_argument('--checkpoint_base', default='', help='initialized from the pre-training model')
    parser.add_argument('--checkpoint_next', default='', help='initialized from the pre-training model')
    parser.add_argument('--num_classes', default=81, type=int)
    
    parser.add_argument('--train_img_dir', default='/ubc/cs/research/shield/datasets/MSCOCO/2017/train2017', type=str)
    parser.add_argument('--test_img_dir', default='/ubc/cs/research/shield/datasets/MSCOCO/2017/val2017', type=str)
    parser.add_argument('--load_task_model', default='/ubc/cs/home/g/gbhatt/borg/cont_learn/runs/hug_demo/checkpoint00.pth', type=str)
    parser.add_argument('--task_ann_dir', default='', type=str)
    parser.add_argument('--split_point',default=0, type=int)
    #parser.add_argument('--test_ann_dir', default='/ubc/cs/research/shield/datasets/MSCOCO/2017/annotations/instances_val2017.json', type=str)
    parser.add_argument('--bbox_thresh', default=0.3, type=float)
    parser.add_argument('--bg_thres', default=0.7, type=float)
    parser.add_argument('--bg_thres_topk', default=5, type=int)
    parser.add_argument('--big_pretrained', default="", type=str)
    return parser

def main(args):

    # fix the seed for reproducibility
    seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #Trainer = pl.Trainer(args)
    seed_everything(seed, workers=True)
    
    args.iou_types = ['bbox']
    out_dir_root = args.output_dir
    
    args.task_map, args.task_label2name =  task_info_voc_10(split_point=args.split_point)
    args.task_label2name[args.n_classes-1] = "BG"

    if args.repo_name:
        processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name)
    else:
        processor = DeformableDetrImageProcessor()
    #print('set up processor ...')

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}')
    #logger = TensorBoardLogger(save_dir=args.output_dir, version=1, name="lightning_logs")
    logger = CSVLogger(save_dir=args.output_dir, name="lightning_logs")
    
    # devices=list(range(8))

    for task_id in range(args.start_task, args.n_tasks+1):
        args.output_dir = os.path.join(out_dir_root, 'Task_'+str(task_id))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_file = open(out_dir_root+'/Task_'+str(task_id)+'_log.out', 'a')
        print('Logging: args ', args, file=args.log_file)

        if task_id == 1:
            args.epochs = 11
        else:
            args.epochs = 6

        #args.switch = True
        args.task = str(task_id)

        #### automatic training schedule
        pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
                    gradient_clip_val=0.1, accumulate_grad_batches=int(32/(args.n_gpus*args.batch_size)), \
                    check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
                    log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0)
        
        #### manual training schedule
        # pyl_trainer = pl.Trainer(devices=list(range(args.n_gpus)), accelerator="gpu", max_epochs=args.epochs, 
        #             check_val_every_n_epoch=args.eval_epochs, callbacks=[checkpoint_callback],
        #             log_every_n_steps=args.print_freq, logger=logger, num_sanity_val_steps=0)
        
        tr_ann = os.path.join(args.task_ann_dir,'train_task_'+str(task_id)+'.json')
        tst_ann = os.path.join(args.task_ann_dir,'test_task_'+str(task_id)+'.json')

        # tr_ann = "/ubc/cs/home/g/gbhatt/borg/cont_learn/data/data/temp_train.json"
        # tst_ann = "/ubc/cs/home/g/gbhatt/borg/cont_learn/data/data/temp_val.json"

        train_dataset = CocoDetection(img_folder=args.train_img_dir, 
                            ann_file=tr_ann, processor=processor)

        test_dataset = CocoDetection(img_folder=args.test_img_dir, 
                                    ann_file=tst_ann, processor=processor)
        
        train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        
        test_dataloader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        
        coco_evaluator = CocoEvaluator(test_dataset.coco, args.iou_types)
        local_evaluator = Evaluator(processor=processor, test_dataset=test_dataset,test_dataloader=test_dataloader,
                                    coco_evaluator=coco_evaluator,args=args,task_label2name=args.task_label2name, task_name='cur')
        
        trainer = local_trainer(train_loader=train_dataloader,val_loader=test_dataloader,
                                      test_dataset=test_dataset,args=args,local_evaluator=local_evaluator,task_id=task_id)
        
        if args.use_prompts:
            print ('previous task : ', trainer.model.model.prompts.task_count, file=args.log_file)
            trainer.model.model.prompts.set_task_id(task_id-1)
            print ('current task : ', trainer.model.model.prompts.task_count, file=args.log_file)

        if task_id>1:
            #prev_task =  os.path.join(args.out_dir_root, 'Task_'+str(task_id-1), 'checkpoint'+str(args.epochs-1)+'.pth')
            if not args.eval:
                if args.resume:
                    prev_task = args.checkpoint_dir.replace('Task_1','Task_'+str(task_id-1))
                    args.resume=0
                else:
                    prev_task = args.output_dir.replace('Task_'+str(task_id),'Task_'+str(task_id-1))
            else:
                prev_task = args.checkpoint_dir.replace('Task_1','Task_'+str(task_id))
            
            if task_id == args.start_task:
                trainer.resume(os.path.join(prev_task,args.checkpoint_base))
            else:
                trainer.resume(os.path.join(prev_task,args.checkpoint_next))
        else:
            if not args.eval:
                if args.repo_name:
                    trainer.resume()
                args.resume=0
            else:
                trainer.resume(os.path.join(args.checkpoint_dir,args.checkpoint_base))

        ####################### Training/Evaluating on Current classes ################################################
        if args.eval:
            trainer.evaluator.local_eval = 1
            # trainer.evaluator.evaluate()
            pyl_trainer.validate(trainer,test_dataloader)
        else:
            pyl_trainer.fit(trainer, train_dataloader, test_dataloader)
        #trainer.train_task(task_id=task_id, task_info=cur_task, task_map=task_label2name)
        #############################################################################################################
        
        if task_id>1:

            ####################### Evaluating on previous classes ###################################################
            prev_task_ids = ''.join(str(i) for i in range(1,task_id))
            tst_ann_prev = os.path.join(args.task_ann_dir,'test_task_'+str(prev_task_ids)+'.json')
            test_dataset_prev = CocoDetection(img_folder=args.test_img_dir, 
                                        ann_file=tst_ann_prev, processor=processor)
            test_dataloader_prev = DataLoader(test_dataset_prev, collate_fn=test_dataset_prev.collate_fn, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
            args.task = prev_task_ids
        
            if len(args.task) == 1:
                args.task = '01'
            coco_evaluator = CocoEvaluator(test_dataset_prev.coco, args.iou_types)
            local_evaluator = Evaluator(processor=processor, test_dataset=test_dataset_prev,test_dataloader=test_dataloader_prev,
                                        coco_evaluator=coco_evaluator,args=args,task_label2name=args.task_label2name,
                                        local_trainer=trainer, local_eval=1, task_name='prev')
            PREV_INTRODUCED_CLS = args.task_map[task_id][1]
            CUR_INTRODUCED_CLS = args.task_map[task_id][2]

            seen_classes = PREV_INTRODUCED_CLS + CUR_INTRODUCED_CLS
            invalid_cls_logits = list(range(seen_classes, args.n_classes-1))
            local_evaluator.invalid_cls_logits = invalid_cls_logits

            trainer.evaluator = local_evaluator
            trainer.evaluator.model = trainer.model
            trainer.eval_mode = True
            pyl_trainer.validate(trainer,test_dataloader_prev)
            #local_evaluator.evaluate()
            ##########################################################################################################

            ####################### Evaluating on all known classes ###################################################
            known_task_ids = ''.join(str(i) for i in range(1,task_id+1))
            tst_ann_known = os.path.join(args.task_ann_dir,'test_task_'+str(known_task_ids)+'.json')
            test_dataset_known = CocoDetection(img_folder=args.test_img_dir, 
                                        ann_file=tst_ann_known, processor=processor)
            test_dataloader_known = DataLoader(test_dataset_known, collate_fn=test_dataset_known.collate_fn, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
            
            args.task = known_task_ids
            coco_evaluator = CocoEvaluator(test_dataset_known.coco, args.iou_types)
            local_evaluator = Evaluator(processor=processor, test_dataset=test_dataset_known,test_dataloader=test_dataloader_known,
                                        coco_evaluator=coco_evaluator,args=args,task_label2name=args.task_label2name,
                                        local_trainer=trainer, local_eval=1, task_name='all')
            PREV_INTRODUCED_CLS = args.task_map[task_id][1]
            CUR_INTRODUCED_CLS = args.task_map[task_id][2]

            seen_classes = PREV_INTRODUCED_CLS + CUR_INTRODUCED_CLS
            invalid_cls_logits = list(range(seen_classes, args.n_classes-1))
            local_evaluator.invalid_cls_logits = invalid_cls_logits

            trainer.evaluator = local_evaluator
            trainer.evaluator.model = trainer.model
            trainer.eval_mode = True
            pyl_trainer.validate(trainer,test_dataloader_known)
            #local_evaluator.evaluate()
            ##########################################################################################################

        args.log_file.close()

    #utils.print_final(out_dir=args.output_dir, n_tasks=args.n_tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    out_dir = args.output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    #if args.eval:
    utils.print_final(out_dir=out_dir, start_task=args.start_task, n_tasks=args.n_tasks)