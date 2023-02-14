export LC_ALL=en_US

k=online_sum
PORT=10101
echo "seleceted $PORT"
PYTHONPATH=$PYTHONPATH:../../../ \
srun --mpi=pmi2 --job-name=$k -n8 -c 10 --gres=gpu:8 --ntasks-per-node=8  \
python -u -m core.solver.linear_solver --config linear_configs/$k.yaml 2>&1|tee logs/linear_$k-epochsampler.log &
# for k in 3 4 5 6 7 8 9 10
# do
# done
# srun --nodelist=dgx1-r120-1.g42cloud.net --mpi=pmi2 --job-name=linear_eval -n8 -c 10 --gres=gpu:8 --ntasks-per-node=8   \

# while true # find unused tcp port
# do GLOG_vmodule=MemcachedClient=-1 
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done

# conda activate fast-moco
# cd ssl_exp/fast_moco/fastmoco_e100/