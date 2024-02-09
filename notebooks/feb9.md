~~~bash
qrsh -g grpname -l rt_G.small=1 -l h_rt=2:00:00
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
- アノテーションから速度と位置のnumpy.arrayを抽出
  - シーンインデックスもつけたし、train, validでわけてある。

## 今日やること
- 質問回答データセットの作成
  - 以下の定型文データセットを作成。累積移動ベクトルは適宜異なる時間スケール、オブジェクトで実行。
    - 位置の質問
      - Where is the {color} {shape} at {time}s? Please provide the coordinates in three dimensions.
      - [{x}, {y}, {z}]
    - 速度の質問
      - What is the velocity of the {color} {shape} at {time}s? Please provide the vector components in three dimensions.
      - [{vx}, {vy}, {vz}]
    - 累積移動ベクトルの質問
      - シンプルな回答
        - How much has the {color} {shape} moved between {start_time}s and {end_time}s? Please provide the vector components in three dimensions.
        - [{dx}, {dy}, {dz}]
      - 誘導付き回答
        - How much has the {color} {shape} moved between {start_time}s and {end_time}s? Please infer with its velocity, consider collision if it happened, and provide the vector components in three dimensions.
        - There are {number_of_collisions} collision(s) related to the {color} {shape} in the following times: {timestamps}.
        - At timestamp {collision_time}, the {color} {shape} collided with {object}, altering its velocity vector to [{vx_after_collision}, {vy_after_collision}, {vz_after_collision}].
        - The motion pattern continued for {end_time} - {collision_time} = {time_interval}[s] with velocity vectors [{vx}, {vy}, {vz}]. During this time, the {color} {shape} moves [{dx_interval}, {dy_interval}, {dz_interval}].
        - Therefore, the cumulative movement is [{dx_previous} + {dx_interval}, {dy_previous} + {dy_interval}, {dz_previous} + {dz_interval}] = [{dx_total}, {dy_total}, {dz_total}].
    - 同じフレームでの２物体の相対位置、速度の質問
      - シンプルな回答
        - What is the relative position of the {color1} {shape1} with respect to the {color2} {shape2} at {time}s?
        - [{dx_relative}, {dy_relative}, {dz_relative}]
        - What is the relative velocity of the {color1} {shape1} with respect to the {color2} {shape2} at {time}s?
        - [{dvx_relative}, {dvy_relative}, {dvz_relative}]
      - 明示的な計算を行う回答
        - What is the relative position of the {color1} {shape1} with respect to the {color2} {shape2} at {time}s?
        - At {time}s, the position of the {color1} {shape1} is [{x1}, {y1}, {z1}] and that of the {color2} {shape2} is [{x2}, {y2}, {z2}]. The relative position with respect to the {color2} {shape2} is [{x1} - {x2}, {y1} - {y2}, {z1} - {z2}] = [{dx_relative}, {dy_relative}, {dz_relative}].
        - What is the relative velocity of the {color1} {shape1} with respect to the {color2} {shape2} at {time}s?
        - At {time}s, the velocity of the {color1} {shape1} is [{vx1}, {vy1}, {vz1}] and that of the {color2} {shape2} is [{vx2}, {vy2}, {vz2}]. The relative velocity with respect to the {color2} {shape2} is [{vx1} - {vx2}, {vy1} - {vy2}, {vz1} - {vz2}] = [{dvx_relative}, {dvy_relative}, {dvz_relative}].

- video-llavaの訓練
  - 必須ではないが、とりあえず実行してエラーが出るところまで進める。

# レポート
 - そもそもどうやって評価するんだっけ
   - 自作データセットでの評価
   - どのくらいバイアス・バリアンスのある回答をするか？
   - clevrerのpredictiveに転移する？
   - counter factrualには、衝突による速度変化の明示が必要？
   - とりあえず位置と速度は終わった
   - 累積はmypyでのチェックがまだ（questionの方はできてる。本体がまだ）
   - 相対位置・速度はすぐできるでしょう、👆ができれば

# Next Action
 - データセット自作完了
 - Video-LLaVA動かす
 - 自作データセットで訓練後にテストデータでのテスト
 - CLEVRERでFTしたあと、質問のタイプごとに対象学習