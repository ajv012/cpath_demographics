DIR_TO_COORDS="/path/to/patch/corrdinates"
DATA_DIRECTORY="/path/to/WSI"
CSV_FILE_NAME="path/to/csv/with/file/names"
FEATURES_DIRECTORY="/path/to/features/storage"
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
