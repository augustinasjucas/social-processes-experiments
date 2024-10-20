# --data_file haggling-full.h5
# --feature_set FULL
# --observed_len 60
# --future_len 60
# --max_future_offset 150
# --time_stride 1
# --data_dim 58

import hashlib

data_flags = {
    "dataset_root": "panoptic-haggling-full-features",
    "data_file": "haggling-full.h5",
    "feature_set": "FULL",
    "observed_len": 60,
    "future_len": 60,
    "max_future_offset": 150,
    "time_stride": 1,
    "data_dim": 58,
}

additional_flags = {
    "dataset_root": "panoptic-haggling-full-features",
    "feature_set": "FULL",
    "batch_size": 64,
    "lr": "1e-5",
    "max_epochs": 1,
    "dropout": 0.25
}

gpu_flags = {
    "ndata_workers": 4,
    "gpus": 1,
}

def VED_MLP(nlayers, enc_nhid, dec_nhid, z_dim, r_dim):
    return {
        "model": "VAE_SEQ2SEQ",
        "component": "MLP",
        "nlayers": nlayers,
        "enc_nhid": enc_nhid,
        "dec_nhid": dec_nhid,
        "z_dim": z_dim,
        "r_dim": r_dim,
        "VERSION": "plain"
    }

def VED_RNN(nlayers, enc_nhid, dec_nhid, z_dim, r_dim):
    return {
        "model": "VAE_SEQ2SEQ",
        "component": "RNN",
        "nlayers": nlayers,
        "enc_nhid": enc_nhid,
        "dec_nhid": dec_nhid,
        "z_dim": z_dim,
        "r_dim": r_dim,
        "VERSION": "plain"
    }

def NP(nlayers, enc_nhid, dec_nhid, z_dim, r_dim, attention_qk_dim, attention_nheads):
    return {
        "model": "NEURAL_PROCESS",
        "nlayers": nlayers,
        "enc_nhid": enc_nhid,
        "dec_nhid": dec_nhid,
        "z_dim": z_dim,
        "r_dim": r_dim,
        "attention_qk_dim": attention_qk_dim,
        "attention_nheads": attention_nheads
    }

def SP_MLP(nlayers, enc_nhid, dec_nhid, z_dim, r_dim, attention_qk_dim, attention_nheads):
    return {
        "model": "SOCIAL_PROCESS",
        "component": "MLP",
        "nlayers": nlayers,
        "enc_nhid": enc_nhid,
        "dec_nhid": dec_nhid,
        "z_dim": z_dim,
        "r_dim": r_dim,
        "attention_qk_dim": attention_qk_dim,
        "attention_nheads": attention_nheads
    }

def SP_RNN(nlayers, enc_nhid, dec_nhid, z_dim, r_dim, attention_qk_dim, attention_nheads):
    return {
        "model": "SOCIAL_PROCESS",
        "component": "RNN",
        "nlayers": nlayers,
        "enc_nhid": enc_nhid,
        "dec_nhid": dec_nhid,
        "z_dim": z_dim,
        "r_dim": r_dim,
        "attention_qk_dim": attention_qk_dim,
        "attention_nheads": attention_nheads
    }

def latent(x):
    x_copy = x.copy()
    # remove attention parameters
    x_copy.pop("attention_qk_dim", None)
    x_copy.pop("attention_nheads", None)
    x_copy["VERSION"] = "latent"
    return x_copy

def uniform(x):
    # remove attention parameters
    x = x.copy()
    x["use_deterministic_path"] = True
    x.pop("attention_qk_dim", None)
    x.pop("attention_nheads", None)
    x["VERSION"] = "uniform"
    return x

def dot(x):
    x = x.copy()
    x["use_deterministic_path"] = True
    x["attention_type"] = "DOT"
    x.pop("attention_qk_dim", None)
    x.pop("attention_nheads", None)
    x["VERSION"] = "dot"
    return x

def mh(x):
    x = x.copy()
    x["use_deterministic_path"] = True
    x["attention_type"] = "MULTIHEAD"
    x["attention_rep"] = "MLP" if x["model"] == "NEURAL_PROCESS" else x["component"]
    if x["attention_rep"] == "MLP":
        assert x["attention_qk_dim"] is not None
    else:
        x.pop("attention_qk_dim", None)
    x["VERSION"] = "multihead"
    return x

configs_to_run = [
    VED_MLP(nlayers=2, enc_nhid=64, dec_nhid=64, z_dim=32, r_dim=32),
    # VED_RNN(nlayers=2, enc_nhid=512, dec_nhid=512, z_dim=128, r_dim=128),
    # NP(nlayers=2, enc_nhid=512, dec_nhid=512, z_dim=128, r_dim=128, attention_qk_dim=64, attention_nheads=4),
    # SP_MLP(nlayers=2, enc_nhid=512, dec_nhid=512, z_dim=128, r_dim=128, attention_qk_dim=64, attention_nheads=4),
    # SP_RNN(nlayers=2, enc_nhid=512, dec_nhid=512, z_dim=128, r_dim=128, attention_qk_dim=64, attention_nheads=4)
    VED_RNN(nlayers=2, enc_nhid=64, dec_nhid=64, z_dim=32, r_dim=32),
    NP(nlayers=2, enc_nhid=64, dec_nhid=64, z_dim=32, r_dim=32, attention_qk_dim=64, attention_nheads=4),
    SP_MLP(nlayers=2, enc_nhid=64, dec_nhid=64, z_dim=32, r_dim=32, attention_qk_dim=64, attention_nheads=4),
    SP_RNN(nlayers=2, enc_nhid=64, dec_nhid=64, z_dim=32, r_dim=32, attention_qk_dim=64, attention_nheads=4)

]

def is_metalearning_model(config):
    return config["model"] in ["NEURAL_PROCESS", "SOCIAL_PROCESS"]


def serialize_flags(flags):
    # remove "VERSION" from the flags
    flags = flags.copy()
    flags.pop("VERSION", None)

    # dont forget to NOT write "--flag_name True" and instead just "--flag_name"
    return " ".join([f"--{k} {v}" if type(v) is not bool else f"--{k}" for k, v in flags.items()])

def main():

    meta_variants = [latent, uniform, dot, mh]

    paramsets = []
    folded_paramsets = []
    for config in configs_to_run:

        # add a hash flag to the config, so that later we can group the results from the same type
        relevant_params = {**data_flags, **additional_flags, **config}
        strigified_params = serialize_flags(relevant_params)
        component = ("-" + config["component"]) if "component" in config else ""
        hash = config["model"] + component + "-" + str(hashlib.md5(strigified_params.encode()).hexdigest())[:10]
        config["paramset_id"] = hash

        
        if is_metalearning_model(config):
            paramsets.extend([variant(config) for variant in meta_variants])
        else:
            paramsets.append(config)

    for paramset in paramsets:
        flags = {**data_flags, **additional_flags, **gpu_flags, **paramset}
        
        for i in range(5):
            new_flags = flags.copy()
            new_flags["fold_file"] = "fold" + str(i) + ".pkl"
            new_flags["out_dir"] = "results/" + flags["paramset_id"] + "/" + new_flags["VERSION"] + "/fold" + str(i) + "/" 
            folded_paramsets.append(new_flags)

    for paramset in folded_paramsets:
        flags = {**data_flags, **additional_flags, **gpu_flags, **paramset}
        print("python -m run.train_dataset " + serialize_flags(flags))


if __name__ == "__main__":
    main()