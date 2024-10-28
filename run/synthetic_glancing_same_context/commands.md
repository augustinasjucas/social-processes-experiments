

## Training with fixed variance

This is what happens when we train with mixed context (i.e., the same experiment as in the SP paper). We get posterior collapse:
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=13-monitored_nll=23.385.ckpt" --my_train_with_mixed_context --test --my_test_with_mixed_context --my_dont_plot_reds --my_plot_nice_mixed_context_summary
```

This is the result when we train with fixed variance and test with different contexts. It is effectively all that we care about.
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --fix_variance --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_test_with_mixed_context --my_dont_plot_reds --my_plot_nice_mixed_context_summary
```
Then with the same model we can also run a few other visualizations/experiments:
```bash
# diffent z analsysis, used for paper figure
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --fix_variance --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_plot_z_analysis --my_z_anal_left 0.21 --my_z_anal_right 1.75

# meta sample visualizations, used for paper figure
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --fix_variance --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_test_with_mixed_context --my_dont_plot_reds

# command for training this model. Same command could be used for training unfixed variacne model just without --fix-variance
python -m run.synthetic_glancing_same_context.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-69txt --fix_variance --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future
```
## Training with unfixed variance
Trained on unfixed variance, here is a z-analysis plot of that. We see that all looks well.
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=038-v7.ckpt" --test --my_plot_z_analysis --my_z_anal_left -2.5 --my_z_anal_right -0.9
```

We can also replace `--my_plot_z_analysis --my_z_anal_left -2.5 --my_z_anal_right -0.9` with `--my_plot_q_of_z` to see the actual q(z|C) for meta samples.

To see how different (also mixed) contexts affect the output, we can plot full analysis (`my_plot_nice_mixed_context_summary`):
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=038-v7.ckpt" --test --my_plot_nice_mixed_context_summary --my_test_with_mixed_context
```

We see a move of the mean as the context changes. However, the transition is not as smooth as it was with the fixed variance. 

## Testing what happens when we draw multiple samples from q(z|C)
*Note: you can append all of these commands with `my_test_with_mixed_context` to see what happens.*

**With fixed variance**. 
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context  --my_multi_plot_reds --my_multi_plot_blacks --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_dont_plot_reds --my_test_multiple_samples_of_z --my_multi_plot_greens
```

**With not fixed variance**
```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --my_multi_plot_greens --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=038-v7.ckpt" --test --my_test_multiple_samples_of_z
```

**Observations** We see that different samples from q(z|C) lead to different outputs. We also see that the mean of the outputs matches the output of the mean z.

Also, here we can see that the predicted output variance does not match the actual variance when we sample from multiple q(z|C)s.

```
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --my_multi_plot_reds --my_multi_plot_blacks --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=038-v7.ckpt" --test --my_test_multiple_samples_of_z
```



## Outputting a mixture of 2 Gaussians
For this, you need to checkout to the GM branch.

The commands to run are:
```bash
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file FFINAAALL-with-proper-val-9.txt --fix_variance --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_proper_validation --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=005-v14.ckpt" --test --my_use_GM --my_dont_plot_reds
```
```bash
python -m run.synthetic_glancing_same_context.test_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file FFINAAALL-with-proper-val-9.txt --fix_variance --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_GM --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=4-monitored_nll=-1.100.ckpt" --test --my_dont_plot_reds
 
 ```


 # Other remarks
 If you want to use proper validation sets, there are flags `--my_use_proper_validation` for training and `--my_plot_validation_set` for plotting the validation set instead of the training set.

Every single model can be tested in different ways, by generating figures for it. So even though the commands above all generate some specific figures, all of them can be changed to generate different figures for the same model specified in the command. The following flags denote different figures:
- `--my_plot_q_of_z` will plot the context and the q(z|c) for meta-samples.
- `--my_plot_z_analysis` will create a figure with different meta samples and examples of what happens to their predictions as z values change.
- `--my_plot_tight_z_analysis` will create a different figure for showing effects of changing `z` on the outputs.
- `--my_plot_nice_mixed_context_summary` - will create pretty much a figure that combines everything that we need: it will show how different context mixtures map to different z values and how different z values map to different outputs for different meta-samples. 
- `--my_test_multiple_samples_of_z` will create a figure showing how different samples taken from q(z|C) of some meta sample results in different predictions. And how these predictions relate to each other, to the mean prediction and to the prediction of the mean. Also, with this one can see how the actual standard deviation of the predictions differs from the predicted standard deviation of the predictions. 
- `[no flags]` in case neither of these flags are set, a plot will be generated for a meta-sample, showing its context, its q(z|C) and the predictions. 
  
Note that combining some of these flags together (for instance, having both `--my_plot_q_of_z` and `--my_plot_z_analysis` in the same command) is not allowed. Also, most of these flags come with other arguments that need to be specified - for that, check the `--help`.
