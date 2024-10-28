# Experiments when predicting a Normal distribution with softmaxed mean

## Training on Dual
Trained on: FFA dual. z: 1-dimensional. Target size and context size is 4. Future length is 2. Predicting Normal distribution with a softmax on the mean.
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 1 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=27-monitored_nll=-13.958.ckpt"
```

Very similar as above. Trained on: FFA dual. z: 1-dimensional. Target size and context size is 3. Future length is 3. Predicting Normal distribution with a softmax on the mean.
```bash
 python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 3 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 1 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 3 --my_target_size 3 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=031-v0.ckpt
 ```

 Similar as above. Trained on: FFA dual. z: 5-dimensional. Target size and context size is 3. Future length is 4. Predicting Normal distribution with a softmax on the mean. Got total posterior collapse:
 ```bash
 python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 3 --my_target_size 3 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=26-monitored_nll=141.463.ckpt"
 ```

However, if I give much more data (same as above just with more data), it starts to work. z 5-dimensional, 10 context, 10 target, 1 observed, 3 future, 7 people, 200 meta samples per case, predicting Normal distribution with a softmax on the mean:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 200  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=014.ckpt
```


## Training on Dominating to see if it is even possible to learn

Training on the dominating dataset. Predicting Normal distribution with a softmax on the mean. It learns to identify the dominating person, but not the clockwise/anticlocwse order.
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_dominating --my_meta_sample_count_per_case 200  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=64.517.ckpt
```

## Training on Dual-random

All results with different data sizes, different z and hidden dimension values are quite similar - we get a lot of averaging behaviour. Here too we are always predicting a Normal distribution with a softmax on the mean.

5-dimensional z:
```bash
 python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=76.608.ckpt
 ```
 16-dimensional z:
 ```bash
 python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 16 --z_dim 16 --override_hid_dims --hid_dim 16 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=19-monitored_nll=109.736.ckpt
```

## Training on Full-random
Full posterior collapse, we get full averaging behaviour, doesn't even learn to find the dominating person.
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 6 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 6  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=2-monitored_nll=154.783.ckpt
```

After tuning, I got remembering to work quite well on VERY SMALL datasets:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 4 --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 2 --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=100.509.ckpt"
```

Then on a slightly bigger dataset, but even then, there are like 7 metasamples (for some reason):
```bash
 python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 2 --my_use_dominating --my_meta_sample_count_per_case 1  --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 3 --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=24-monitored_nll=122.775.ckpt" --my_how_many_z_samples_to_plot 4
```
On even larger, we get counting behaviour:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 4 --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=25-monitored_nll=124.882.ckpt"
```



# Predicting softmax on the mean and then doing categorical cross-entropy on the vector

Trained on Dual dataset. Note that there are no special flags here, as I just hardcoded the loss.
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=30-monitored_nll=0.035.ckpt"
```

# Predicting a Categorical distribution

## A lot of smaller experiments
Before doing the controlled setting with the same architecture but different datasets, I did a lot of other experiments. I will past the most serious ones here so that they are not lost, however they were mostly exploratory.

Trained on the Dual dataset. z 1-dimensional, 7 people, small hidden dimensions: 
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical  --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=0.038.ckpt"
```

Trained on the Dual dataset. z 5-dimensional, 7 people, small hidden dimensions. Does not work as I had kind of a bug in the Categorical loss: 
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 200  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=33-monitored_nll=0.877.ckpt"
```

Trained on the Dual-random dataset. Learns to completely average, however there still was a kind of a bug in the Categorical loss:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 16 --z_dim 16 --override_hid_dims --hid_dim 16 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_predict_categorical --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=17-monitored_nll=1.069.ckpt"
```

Then we run 3 experiments in a controlled setting with 1-dimensional z. We get sever posterior collapses:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --my_analyse_datasets --my_ffa_dual_model  "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=0.038.ckpt" --my_ffa_dual_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=191-monitored_nll=0.900.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=8-monitored_nll=1.984.ckpt" --my_plot_posteriors
``` 

Then I kind of fixed the bug and got some remembering to work for the a small full-random dataset:
```
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_predict_categorical --fix_variance --nz_samples 1 --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 2 --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=58-monitored_nll=7.509.ckpt
``` 

## Final experiment with 1-dimensional z
Everything stays the same for all models, except the training dataset changes. 5 people, 8 context interactions, 3 target interactions, repeat_count=2.

Model trained on the Dual dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=0.006.ckpt"
```
Model trained on the Dual-random dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=62-monitored_nll=38.022.ckpt"
```
Model trained on the Full-random dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random  --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=22-monitored_nll=70.348.ckpt"
```

Dataset analysis of all 3 models:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 10 --my_how_many_random_permutations 2000  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --my_analyse_datasets --my_ffa_dual_model "out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=0.006.ckpt" --my_ffa_dual_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=62-monitored_nll=38.022.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=22-monitored_nll=70.348.ckpt" --my_plot_posteriors
```






## Final experiment with 64-dimensional z
Everything stays the same for all models, except the training dataset changes. 5 people, 8 context interactions, 3 target interactions, repeat_count=2.

Model trained on the Dual dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-6.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=28-monitored_nll=0.001.ckpt"
```
Model trained on the Dual-random dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-7.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=103-monitored_nll=38.794-v0.ckpt"
```
Model trained on the Full-random dataset:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-8.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random  --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=76-monitored_nll=69.785.ckpt"
```

Dataset analysis of all 3 models:
```bash
python -m run.ordering.test_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 10 --my_how_many_random_permutations 2000  --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --my_analyse_datasets --my_ffa_dual_model "out-3\logs\checkpoints-synthetic\mon-epoch=28-monitored_nll=0.001.ckpt" --my_ffa_dual_random_model "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=103-monitored_nll=38.794-v0.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=76-monitored_nll=69.785.ckpt" --my_plot_posteriors
```

