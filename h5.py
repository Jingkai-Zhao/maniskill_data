import h5py

with h5py.File("/home/teleai/miniconda3/envs/maniskill/lib/python3.10/site-packages/mani_skill/demos/aaa-OpenCabinetDrawer-v1/teleop/merged_trajectory.h5","r") as f:
    print(f.keys())