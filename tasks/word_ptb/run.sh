# export CUDA_VISIBLE_DEVICES=0; nohup python -u train.py --use_lstm --batch_size 256 --task char > lstm.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python -u train.py --batch_size 256 --task char > tcn.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0; nohup python -u train.py --use_lstm --batch_size 256 --task char > lstm_no_recurrent_dropout.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python -u train.py --batch_size 256 --task char > tcn_boost.log 2>&1 &