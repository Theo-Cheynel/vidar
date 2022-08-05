git pull
python3 -m scripts.inference.inference ./configs/papers/selfcalib/ucm_thomas.yaml /data/vidar/checkpoints/hardy-lion-67/models/032.ckpt $1 /data/output/hardy-lion-67/walk_person_idle --export_type=npy
