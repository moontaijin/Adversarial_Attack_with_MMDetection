CONFIG=""
CHECKPOINT=""
IMG_PATH=""
DEVICE="cuda:0"

python ./adv_make.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --img-dir ${IMG_PATH} \
    --device ${DEVICE} \
    --steps 200