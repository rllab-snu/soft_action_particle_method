################## OPTIONS ##################
# N_layers : 2(2layer-Q_Network), 3(3layer-Q_Network)
# DPP      : 0(Random Sampling), 1(DPP Sampling)
# Ablation : 0(Fixed Particles), 1(Grad + No Resampling),
#	     2(Grad + Resampling), 3(SAPM)
#############################################

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_5e-1_128 --act_size 128 --min_d 0.3 --scale 0.5 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-1_128 --act_size 128 --min_d 0.3 --scale 0.1 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &
 
export CUDA_VISIBLE_DEVICES=2
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-2_128 --act_size 128 --min_d 0.3 --scale 0.01 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 

export CUDA_VISIBLE_DEVICES=3
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_5e-1_64 --act_size 64 --min_d 0.3 --scale 0.5 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-1_64 --act_size 64 --min_d 0.3 --scale 0.1 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &
 
export CUDA_VISIBLE_DEVICES=1
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-2_64 --act_size 64 --min_d 0.3 --scale 0.01 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 

export CUDA_VISIBLE_DEVICES=2
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_5e-1_16 --act_size 16 --min_d 0.3 --scale 0.5 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &

export CUDA_VISIBLE_DEVICES=3
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-1_16 --act_size 16 --min_d 0.3 --scale 0.1 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 &
 
export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env HalfCheetah-v2 --exp_name Half-sapm_1e-2_16 --act_size 16 --min_d 0.3 --scale 0.01 --N_layers 3 --DPP 0 --seed 0 10 20 30 40 
