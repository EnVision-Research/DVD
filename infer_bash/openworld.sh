CKPT='ckpt/DVD'

INPUT_VIDEO='demo/raw_run_cropped_src.mp4'

# You could increase the resolution here but expect slower inference speed.
HEIGHT=480
WIDTH=640

# You might increase these two number to ensure more stable scale variation but expect slower inference speed
WINDOW_SIZE=81
OVERLAP=21

python test_script/test_single_video.py --ckpt $CKPT --input_video $INPUT_VIDEO  --height $HEIGHT --width $WIDTH --window_size $WINDOW_SIZE --overlap $OVERLAP