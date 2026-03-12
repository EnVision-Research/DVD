CKPT='ckpt'
FRAME_NUM=300 # 200,300
NUM=10
VIDEO_BASE_DATA_DIR='/hpc2hdd/home/hongfeizhang/dataset'

python test_script/test_from_trained_all_vid.py --ckpt $CKPT --frame $FRAME_NUM --num $NUM --base_data_dir $VIDEO_BASE_DATA_DIR;