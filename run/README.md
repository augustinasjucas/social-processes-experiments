# Ran experiments

## 0. First experiments before it started working
The main issue why nothing was working was that I had to encode the full context for calculation of q(z | C). I.e., for calculation of q(z | C)
I had to change the context into fully-observed context with no corresponding futures, since the crucial information about the context is in the future sequences
but the model encodes only the observed parts of the context.

Before that, there were also other drawbacks. For instance, choosing too small hid_dim for encoders/decoders (e.g., 2) messes up back-propagation, since
by default the ReLU activation function is used and by randomness the two activations go below zero, and the activation vector results to a constant value of 0
and as a result, no back-propagation is done. 

Also, to enforce an exact value for `z_n_hid` I had to hackily change the model code (since it was calculating `z_n_hid` using some smart averaging), 
to use the exact value that I provide in the arguments. A separate commit for this was made to make this clear and easily revertable.

In this process I also tried enforcing one of the KL terms to be some constant predefined value (based on what type of context is given) to make 
sure q(z | C) is learned correctly. That did not help, but it was the final step before I realized that I was not providing enough data in the context for
proper calculation of q(z | C) (i.e., as mentioned - I was not providing the future sequences into the context, but only the observed parts of the context). Since
that was extremely hacky, I had to revert this long time ago and this posterior forcing experiment is not reproducible. But running the z analysis
experiments **is** done and that part does even more advanced posterior forcing, so all is good. 

I will not post the commands that did not work, since they are not very useful and would require retraining due to changes made to the code.

## 1. First time it worked
After a lot of debugging and changing things (in code), this was the first **training** command that worked:
```bash
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future 
```
The corresponding test command:
```bash
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=016-v7.ckpt" 
```
In there, it can be clearly seen that the future predictions are learned, and they do not match the averaged (red) line as in all previous experiments
Also, Adding a validation set could be tried, but at least it does not underfit, which was a challenge for long!

## 2. Trying to see what different z values do
To test how different z values affect the predictions, I trained the model:
```bash
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future
```
And then tested it with:
```bash
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=016-v7.ckpt" --my_plot_z_analysis 
```
This was essentially the main result of the experiments. It showed that the model learns a nice mapping to/from the latent space.

**TO BE CONTINUED**