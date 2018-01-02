# rl-mapping

Code to reproduce [Active Robotic Mapping through Deep Reinforcement Learning](https://arxiv.org/abs/1712.10069).

Commands to run:
```
# MLP
nohup python main.py --experiment runs/mlp --network mlp --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python main.py --experiment runs/mlp-prims --network mlp --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &

nohup python main.py --experiment runs/mlp-range --network mlp --N 25 --map_p .1 --sensor_type range --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &

# CNN
nohup python main.py --experiment runs/cnn --cuda --network cnn --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python main.py --experiment runs/cnn-prims --cuda --network cnn --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &

# Resnet
nohup python main.py --experiment runs/resnet --cuda --network resnet --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 2000 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python main.py --experiment runs/resnet-prims --cuda --network resnet --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 300 --N_episodes 20000 --max_steps 20 --anneal_step_size 5000 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &

# Myopic
nohup python myopic.py --experiment runs/myopic --N 25 --map_p .1 --episode_length 300 --sensor_type local --sensor_span 1 --sensor_p .8 --seed 0 &
nohup python myopic.py --experiment runs/myopic-prims --N 25 --prims --map_p .1 --episode_length 300 --sensor_type local --sensor_span 1 --sensor_p .8 --seed 7 &
```
