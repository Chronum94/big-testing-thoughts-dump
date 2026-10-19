"""molcrys v2: B4 vs no-c4/c6 trained properly on the GPU (Muon, small batch, long schedule)."""
import sys, yaml
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
for arm, src in [("B4", "mcB4_s1.yaml"), ("nc", "mcnc_s1.yaml")]:
    for seed in (1, 2, 3):
        c = yaml.safe_load(open("/home/chronum/scratch/molcrys/" + src))
        name = f"v2{arm}_s{seed}" if EPOCHS >= 100 else f"t{arm}_s{seed}"
        c["data"]["experiment"] = name
        c["data"]["directory"] = "/home/chronum/scratch/molcrys/v2/models/"
        c["data"]["batch_size"] = 4
        c["n_epochs"] = EPOCHS
        c["optimizer"]["schedule"]["period"] = EPOCHS
        c["ckpt_interval"] = 50
        c["seed"] = seed
        yaml.safe_dump(c, open(f"/home/chronum/scratch/molcrys/v2/{name}.yaml", "w"), sort_keys=False)
