################## OPTIONS ##################
# N_layers : 2(2layer-Q_Network), 3(3layer-Q_Network)
# DPP      : 0(Random Sampling), 1(DPP Sampling)
# Ablation : 0(Fixed Particles), 1(Grad + No Resampling),
#	     2(Grad + Resampling), 3(SAPM)
#############################################

#export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env Swimmer-v2 --exp_name Swimmer-sapm0.5 --act_size 32 --min_d 0.1 --scale 0.5 --N_layers 2 --DPP 1 --seed 0 10 20 30 40 &

#export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env Swimmer-v2 --exp_name Swimmer-sapm0.2 --act_size 32 --min_d 0.1 --scale 0.2 --N_layers 2 --DPP 1 --seed 0 10 20 30 40 &
 
#export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sapm --env Swimmer-v2 --exp_name Swimmer-sapm0.1 --act_size 32 --min_d 0.1 --scale 0.1 --N_layers 2 --DPP 1 --seed 0 10 20 30 40 &
 
