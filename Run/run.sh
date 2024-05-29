#!/bin/sh

python -u S_update.py --d Yelp -m LightGCN_1 --tt 1 --rl --US --UP --ab 50 --ss 1 --ps 0 --sw 1.0 --pw 0.0 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 1 --rl --UCL --US --UP --ab 100 --ss 5 --ps 0 --cs 5 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 1 --rl --UCL --US --UP --ab 100 --ss 5 --ps 0 --cs 5 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_2 --tt 1 --rl --UCL --US --UP --ab 100 --ss 5 --ps 0 --cs 5 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_3 --tt 1 --rl --UCL --US --UP --ab 100 --ss 5 --ps 0 --cs 5 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_4 --tt 1 --rl --UCL --US --UP --ab 100 --ss 5 --ps 0 --cs 5 --max_epoch 10
wait
python -u Ensemble.py --d Yelp --tt 1 
python -u KD.py --d Yelp -m LightGCN_1 --tt 1 --max_epoch 10
python -u S_update.py --d Yelp -m LightGCN_1 --tt 2 --rl --US --UP --ab 100 --ss 3 --ps 5 --sw 1.0 --pw 0.1 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_2 --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_3 --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_4 --tt 2 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 5  --max_epoch 10
wait
python -u Ensemble.py --d Yelp --tt 2 
python -u KD.py --d Yelp -m LightGCN_1 --tt 2 --max_epoch 10
python -u S_update.py --d Yelp -m LightGCN_1 --tt 3 --rl --US --UP --ab 100 --ss 5 --ps 5 --sw 0.9 --pw 0.0 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 3 --rl --UCL --US --UP --ab 50 --ss 3 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 3 --rl --UCL --US --UP --ab 50 --ss 3 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_2 --tt 3 --rl --UCL --US --UP --ab 50 --ss 3 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_3 --tt 3 --rl --UCL --US --UP --ab 50 --ss 3 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_4 --tt 3 --rl --UCL --US --UP --ab 50 --ss 3 --ps 5 --cs 1  --max_epoch 10
wait
python -u Ensemble.py --d Yelp --tt 3 
python -u KD.py --d Yelp -m LightGCN_1 --tt 3 --max_epoch 10
python -u S_update.py --d Yelp -m LightGCN_1 --tt 4 --rl --US --UP --ab 50 --ss 1 --ps 5 --sw 0.9 --pw 0.0  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_2 --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_3 --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_4 --tt 4 --rl --UCL --US --UP --ab 100 --ss 1 --ps 3 --cs 1  --max_epoch 10
wait
python -u Ensemble.py --d Yelp --tt 4 
python -u KD.py --d Yelp -m LightGCN_1 --tt 4 --max_epoch 10
python -u S_update.py --d Yelp -m LightGCN_1 --tt 5 --rl --US --UP --ab 50 --ss 3 --ps 1 --sw 0.9 --pw 0.0 --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_2 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_3 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1  --max_epoch 10
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_4 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1  --max_epoch 10
wait
python -u Ensemble.py --d Yelp --tt 5 
python -u KD.py --d Yelp -m LightGCN_1 --tt 5 --max_epoch 10

# date
# squeue --job $SLURM_JOBID
# echo "##### END ####"
