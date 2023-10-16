#!/bin/bash
# 
#CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=wandb_s2s_comp
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --mem=48G
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --time=12:00:00
#SBATCH --output=jupyter.log

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/10.2
module load cudnn/8.1.1/cuda-10.2
echo "======================="


# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i 6000-6999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "

 For more info and how to connect from windows, 
   see http://login.kuacc.ku.edu.tr/#h.p3tmxkpdxjsz

 Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: login.kuacc.ku.edu.tr
SSH login: $user
SSH port: 22

====================================================================================
 MacOS or linux terminal command to create your ssh tunnel on your local machine:

ssh -N -L ${port}:${node}:${port} ${user}@login.kuacc.ku.edu.tr
====================================================================================

WAIT 1 MINUTE, WILL BE CONNECT ADDRESS APPEARS!

"

# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
jupyter-lab --no-browser --port=${port} --ip="*"

