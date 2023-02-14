export LC_ALL=en_US

k=config_forget_3_sgd_swa
PORT=10101
echo "seleceted $PORT"

date # when we start

PYTHONPATH=$PYTHONPATH:../../../ \
srun --mpi=pmi2 --job-name=linear_eval -n1 -c 20 --gres=gpu:4 --ntasks-per-node=1   \
python -u -m core.solver.linear_solver_swa --config linear_configs/$k.yaml 2>&1|tee logs/swa_linear_$k-now.log &

date # when we end
