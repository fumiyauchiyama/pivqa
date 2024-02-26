~~~bash
qrsh -g grpname -l rt_G.small=1 -l h_rt=5:00:00
module load singularitypro
export SINGULARITY_TMPDIR=$SGE_LOCALDIR
singularity run --bind /groups/gaf51265/fumiyau/pivqa:/groups/gaf51265/fumiyau/pivqa --nv docker://nvcr.io/nvidia/pytorch:23.11-py3
cd /groups/gaf51265/fumiyau/pivqa
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/groups/gaf51265/fumiyau/pivqa/src:$PYTHONPATH"
~~~
# Review
## GOAL: 物理的VQAで
## 昨日行ったこと

## 今日やること

jupyter notebook --no-browser --ip=`hostname` >> jupyter.log 2>&1 &
jupyter notebook list

ssh -L 18888:g0007:8888 -l acc13097es -i C:\Users\tomom\.ssh\id_rsa_abci -p 10022 localhost
ssh -N -L 18888:g0007:8888 abci