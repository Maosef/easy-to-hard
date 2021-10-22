OUTPUT="test"
# RNN_MODEL_PATH="checkpoints/recur/recur_net_sgd_depth=84_width=8_lr=0.01_batchsize=300_at109_epoch=119_1.pth"
# RNN_MODEL_PATH="checkpoints/recur_resnet_adam_depth=28_width=2_lr=0.1_batchsize=64_at0_epoch=0_log.pth"
RNN_MODEL_PATH="check_default/recur_resnet_adam_depth=28_width=2_lr=0.01_batchsize=64_at1_epoch=1_log.pth"

# DEPTH=84
# FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at119_epoch=119_1.pth"
# python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000

# DEPTH=88
# FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at99_epoch=139_1.pth"
# python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000

# DEPTH=92
# FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at99_epoch=139_1.pth"
# python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000

python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 5
python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 10
python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 15
python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 20
python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 25
python train.py --save_json --output $OUTPUT --model recur_resnet --epochs 0 --width 2 --depth 28 --train_batch_size 64 --test_batch_size 64 --model_path $RNN_MODEL_PATH --test_iterations 30

# python train.py \
# --model recur_resnet \
# --width 2 \
# --depth 28 \
# --epochs 1 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# # --model_path recur_28_1.pth \
# --quick_test \
# --test_mode max_conf \
# --test_iterations 8



