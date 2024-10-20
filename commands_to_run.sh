# latent, neural_process, mlp 
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component MLP --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-np-mlp

# latent, neural process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-np-rnn

# latent, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1  --data_dim 58 --model VAE_SEQ2SEQ --component MLP --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-vae-mlp

# latent, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-vae-rnn

# latent, social process, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component MLP --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-sp-mlp

# latent, social process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/latent-sp-rnn



# uniform, neural_process, mlp 
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component MLP --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-np-mlp

# uniform, neural process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component RNN --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-np-rnn

# uniform, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component MLP --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-vae-mlp

# uniform, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component RNN --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-vae-rnn

# uniform, social process, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component MLP --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-sp-mlp

# uniform, social process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component RNN --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/uniform-sp-rnn



# dot, neural_process, mlp 
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component MLP --use_deterministic_path --attention_type DOT --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-np-mlp

# dot, neural process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component RNN --use_deterministic_path --attention_type DOT --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-np-rnn

# dot, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component MLP --use_deterministic_path --attention_type DOT --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-vae-mlp

# dot, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component RNN --use_deterministic_path --attention_type DOT --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-vae-rnn

# dot, social process, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component MLP --attention_type DOT --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-sp-mlp

# dot, social process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component RNN --attention_type DOT --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/dot-sp-rnn



# mh, neural_process, mlp 
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component MLP --use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-np-mlp

# mh, neural process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model NEURAL_PROCESS --component RNN --use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-np-rnn

# mh, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component MLP --use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-vae-mlp

# mh, vae, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model VAE_SEQ2SEQ --component RNN --use_deterministic_path --attention_type MULTIHEAD --attention_rep RNN --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-vae-rnn

# mh, social process, mlp
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component MLP --attention_type MULTIHEAD --attention_rep RNN --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-sp-mlp

# mh, social process, rnn
python -m run.train_dataset --dataset_root panoptic-haggling-full-features --data_file haggling-full.h5 --feature_set FULL --gpus 1 --observed_len 60 --future_len 60 --max_future_offset 150 --time_stride 1 --data_dim 58 --model SOCIAL_PROCESS --component RNN --attention_type MULTIHEAD --attention_rep RNN --use_deterministic_path --lr 1e-5 --weight_decay 5e-4 --max_epochs 500 --train_all --dropout 0.25 --ndata_workers 4  --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --out_dir results/mh-sp-rnn




# considerations: 1. k-fold cross-validation; 2. sampling rate is 20 rn (so approx 1 second), maybe change that? 3. ...