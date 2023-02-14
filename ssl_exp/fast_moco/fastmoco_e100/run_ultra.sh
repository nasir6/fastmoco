export LC_ALL=en_US


PORT=10104
k=online_sum
echo "seleceted $PORT"

PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 --job-name=$k -n16 -c 10 --gres=gpu:8 --ntasks-per-node=8  \
python -u -m core.solver.ultra_fastmoco_solver --config ssl_configs/$k.yaml --tcp_port $PORT 2>&1|tee logs/train-$k-epoch-sampler.log &

# while true 
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     echo "seleceted $PORT"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
#  python -u -m core.solver.fastmoco_solver --config ssl_exp/fast_moco/fastmoco_e100/config.yaml --tcp_port 10111

