# rl-mapping


```
# MLP
nohup python q.py --experiment runs/run-mlp --network mlp --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 500 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python q.py --experiment runs/run-mlp-prims --network mlp --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 500 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &

# CNN
nohup python q.py --experiment runs/run-cnn --cuda --network cnn --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 500 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python q.py --experiment runs/run-cnn-prims --cuda --network cnn --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 500 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
# Resnet
nohup python q.py --experiment runs/run-resnet --cuda --network resnet --N 25 --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 2000 --anneal_gamma .9 --optimizer adam --lr 3e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
nohup python q.py --experiment runs/run-resnet-prims --cuda --network resnet --N 25 --prims --map_p .1 --sensor_type local --sensor_span 1 --sensor_p .8 --gamma .99 --episode_length 200 --N_episodes 10000 --max_steps 20 --anneal_step_size 500 --anneal_gamma .5 --optimizer adam --lr 1e-4 --lambda_entropy .001 --max_grad_norm 50 --seed 7 &
# Myopic
nohup python myopic.py --experiment runs/myopic --N 25 --map_p .1 --episode_length 200 --sensor_type local --sensor_span 1 --sensor_p .8 --seed 7 &
nohup python myopic.py --experiment runs/myopic-prims --N 25 --prims --map_p .1 --episode_length 200 --sensor_type local --sensor_span 1 --sensor_p .8 --seed 7 &
```