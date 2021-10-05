OUTPUT="test"
RNN_MODEL_PATH="checkpoints/recur/recur_net_sgd_depth=84_width=8_lr=0.01_batchsize=300_at109_epoch=119_1.pth"

DEPTH=84
# FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at99_epoch=139_1.pth"
FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at119_epoch=119_1.pth"
python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000

DEPTH=88
FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at99_epoch=139_1.pth"
python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000

DEPTH=92
FF_MODEL_PATH="checkpoints/ff/ff_net_sgd_depth=${DEPTH}_width=8_lr=0.01_batchsize=300_at99_epoch=139_1.pth"
python train.py --save_json --output $OUTPUT --model ff_net --epochs 0 --width 8 --depth $DEPTH --train_batch_size 3000 --test_batch_size 3000 --model_path $FF_MODEL_PATH --eval_start 600000 --eval_end 700000


python train.py --save_json --output $OUTPUT --model recur_net --epochs 0 --width 8 --depth 84 --train_batch_size 3000 --test_batch_size 3000 --test_mode max_conf --model_path $RNN_MODEL_PATH --test_iterations 20 --eval_start 600000 --eval_end 700000
python train.py --save_json --output $OUTPUT --model recur_net --epochs 0 --width 8 --depth 84 --train_batch_size 3000 --test_batch_size 3000 --test_mode max_conf --model_path $RNN_MODEL_PATH --test_iterations 21 --eval_start 600000 --eval_end 700000
python train.py --save_json --output $OUTPUT --model recur_net --epochs 0 --width 8 --depth 84 --train_batch_size 3000 --test_batch_size 3000 --test_mode max_conf --model_path $RNN_MODEL_PATH --test_iterations 22 --eval_start 600000 --eval_end 700000
python train.py --save_json --output $OUTPUT --model recur_net --epochs 0 --width 8 --depth 84 --train_batch_size 3000 --test_batch_size 3000 --test_mode max_conf --model_path $RNN_MODEL_PATH --test_iterations 23 --eval_start 600000 --eval_end 700000
python train.py --save_json --output $OUTPUT --model recur_net --epochs 0 --width 8 --depth 84 --train_batch_size 3000 --test_batch_size 3000 --test_mode max_conf --model_path $RNN_MODEL_PATH --test_iterations 24 --eval_start 600000 --eval_end 700000

