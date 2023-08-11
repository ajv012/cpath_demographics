# breast, resnet, abmil, no mitigation
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_source "/path/to/features" \
--in_dim 1024 \
--exp_code BRCA_ABMIL_example \
--target_col OncoTreeCode \
--task BRCA_subtyping \
--k 20 \
--max_epochs 50 \
--es_min_epochs 25 \
--es_patience 20 \
--split_dir TCGA_BRCA_subtyping \
--subtyping \
--model_type clam_sb \
--no_inst_cluster \
--results_dir results \
--early_stopping



