CKPT='ckpt'
IMAGE_BASE_DATA_DIR='/hpc2hdd/JH_DATA/share/jhe812/PrivateShareGroup/jhe812_d4p_datasets/d4p_datasets/eval/depth'

python test_script/test_from_trained_all_img.py --ckpt $CKPT --base_data_dir $IMAGE_BASE_DATA_DIR