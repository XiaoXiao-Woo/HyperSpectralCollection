# mkdir run_folders
# run_folder= run_folders/$(date +20%y-%m-%d-%H-%M-%S)
# mkdir run_folders
# mkdir $run_folder
# run_file= run-file.sh

# cuda = 7
# # mkdir $run_folder
# bash ./copyfile $run_folder
# cd $run_folder
# bash $run_file $cuda
cuda_num=$1
run_folder=$2
cuda_count=$3
echo $cuda_num
CUDA_VISIBLE_DEVICES=$cuda_num python3 -m torch.distributed.launch --master_port 3213 --nproc_per_node=1 main.py --envname $run_folder
# cuda_num=$1
# run_folder=$2
# echo $cuda_num
# CUDA_VISIBLE_DEVICES=$cuda_num python3 Overall.py --envname $run_folder