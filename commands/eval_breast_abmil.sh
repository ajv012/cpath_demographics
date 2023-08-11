TARGET_COL="oncotree_code"

SPLIT_DIR="/path/to/eval/splits"
TASK="BRCA_subtyping"
DATA_SOURCE="/path/to/features" 
EXP_CODE="experiment_code"
CKPT_PATH="/results/dir/with/model/checkpoints"
MODEL_TYPE="clam_sb"

CUDA_VISIBLE_DEVICES=0 python eval.py \
--data_source $DATA_SOURCE \
--in_dim 1024 \
--ckpt_path $CKPT_PATH \
--exp_code $EXP_CODE \
--target_col $TARGET_COL \
--task $TASK \
--model_type $MODEL_TYPE \
--split_dir $SPLIT_DIR

