~~~bash
[username@es1 ~]$ qrsh -g grpname -l rt_G.small=1 -l h_rt=2:00:00
[username@g0001 ~]$ module load singularitypro
[username@g0001 ~]$ export SINGULARITY_TMPDIR=$SGE_LOCALDIR
[username@g0001 ~]$ singularity run --bind /groups/gaf51265/fumiyau/pivqa:/groups/gaf51265/fumiyau/pivqa --nv docker://nvcr.io/nvidia/pytorch:23.11-py3
cd fumiyau/pivqa/environments/singularity
singularity build --fakeroot env.sif env.def
singularity run --bind /groups/gaf51265/fumiyau/pivqa:/groups/gaf51265/fumiyau/pivqa --nv env.sif
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/groups/gaf51265/fumiyau/pivqa/src:$PYTHONPATH"
cd ..
cd ..
poetry install
poetry add torch deepspeed
poetry add transformers datasets accelerate
cd data
mkdir clevrer
cd clevrer
mkdir video_train
mkdir video_valid
mkdir video_test
mkdir annotation_train
mkdir annotation_valid
cd video_train
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
unzip -
http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip
http://data.csail.mit.edu/clevrer/questions/validation.json
http://data.csail.mit.edu/clevrer/questions/test.json


export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/groups/gaf51265/fumiyau/pivqa/src:$PYTHONPATH"
~~~


今日行ったこと
・環境の作り直し
    ascenderをもとに
     - singularityのenv.def作成
     - poetryのインストール、pythonライブラリ整備
    を行った。逐一PATHを設定する必要があるがまぁこのままで
・アノテーションから速度と位置のnumpy.arrayを抽出
    シーンインデックスもつけたし、train, validでわけてある
・gitのpush環境構築
    aist-abciでgitのアクセストークン作成、devブランチにpush
    https://qiita.com/riita10069/items/e875ae6b96756abd0956

明日やること
・質問回答データセットの作成
    定型文を作成する。
    - 位置の質問
    - 速度の質問
    - 累積移動ベクトルの質問
    - （同じフレームでの２物体の相対位置、速度の質問）
    でデータセットを作成。累積移動ベクトルは適宜ことなる時間スケール、オブジェクトで実行
・video-llavaの訓練
    （出来なくてもいい、とりあえず実行してエラーが出るところまでいく）