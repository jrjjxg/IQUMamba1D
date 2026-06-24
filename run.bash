python main.py  --data_choice 2016 --source_names BPSK QPSK
python main.py --data_choice 2018 --source_names BPSK QPSK
python main.py --data_choice TorchSig --source_names BPSK QPSK  --multiple_runs --num_runs 5 --start_seed 42
python main.py --mode train --data_choice 8PSK-A --source_names S1 S2 --stage 41 --loss_fun PIT-SI-SNR+Huber --synthetic_root /path/to/synthetic --train_mix_enable --train_mix_prob 0.35 --train_mix_sir_min_db -4 --train_mix_sir_max_db 4

