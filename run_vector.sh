#!/bin/bash
#SBATCH --job-name=oak_4task
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --partition='a40'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --output=slurm-%j.out


export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64:/pkgs/cudnn-10.2-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/pkgs/cuda-11.3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1


use_prompts=0
num_prompts=100
plen=10
train=1
exp_name="OAK_4Task"
# repo_name="facebook/deformable-detr-detic"
repo_name="SenseTime/deformable-detr"
tr_dir="/scratch/ssd004/scratch/jross/OAK/OAK_FRAME"
val_dir="/scratch/ssd004/scratch/jross/OAK/OAK_FRAME"
task_ann_dir="/scratch/ssd004/scratch/jross/OAK/coco_labels/"${split_point}
# big_pretrained="SenseTime/deformable-detr"
# freeze='backbone,encoder,decoder,bbox_embed,reference_points,input_proj,level_embed'
freeze='backbone,encoder,decoder'
new_params='class_embed,prompts'

EXP_DIR=/scratch/ssd004/scratch/jross/iod_exps/{exp_name}

if [[ $train -gt 0 ]]
then
echo "Training ..."
LD_DIR=/scratch/ssd004/scratch/jross/iod_exps/PT_NEW
/h/jross/cont_det/env_pyl/bin/python3 main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 1 --batch_size 1 --epochs 11 --lr 1e-4 --lr_old 1e-5 --n_classes=104 --num_workers=2 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --viz --new_params=${new_params} \
    --repo_name=${repo_name} --n_tasks=4 --save_epochs=1 --eval_epochs=1 --split_point=0 \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint10.pth' --resume=0 --start_task=1
else
echo "Evaluating ..."
exp_name="eval_OAK_4Task"
EXP_DIR=/scratch/ssd004/scratch/jross/iod_exps/${exp_name}
LD_DIR=/scratch/ssd004/scratch/jross/iod_exps/coco_40_PT

/h/jross/cont_det/env_pyl/bin/python3 main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 2 --batch_size 1 --epochs 16 --lr 1e-4 --lr_old 1e-5 --save_epochs=5 --eval_epochs=2 --n_classes=103 --num_workers=2 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --new_params=${new_params} \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint10.pth' --checkpoint_next 'checkpoint10.pth' --eval --viz \
    --start_task=2 --n_tasks=2
fi

# #### Resume from a checlpoint for given task and then train
# python main.py \
#     --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
#     --n_gpus 8 --batch_size 1 --epochs 26 --lr 1e-4 --lr_old 1e-5 --n_tasks=4 \
#     --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len $plen --save_epochs=5 --eval_epochs=2 \
#     --freeze=${freeze} --viz --new_params=${new_params} --n_classes=81 --num_workers=2 \
#     --start_task=2 --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint 'checkpoint10.pth' --resume=1