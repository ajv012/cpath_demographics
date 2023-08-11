DATA_DIRECTORY="/path/to/WSI"
RESULTS_DIRECTORY="/path/to/results/storage"
PRESET=bwh_biopsy.csv
python create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY --patch_size 256 --preset $PRESET --seg --patch --stitch
