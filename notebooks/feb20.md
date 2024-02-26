~~~bash
qrsh -g grpname -l rt_AF=1 -l USE_SSH=1 -l h_rt=1:00:00
acc13097es@a0036
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.9/8.9.7 nccl/2.15/2.15.5-1 hpcx/2.12
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


video_10000
"question_id": 6,
"question": "How much has the yellow cube moved between 0.96s and 4.32s? Please provide the vector components in three dimensions.",
"question_type": "cumulative",
"question_subtype": "elaborative",
"program": [],
"square_error": 16.734104719008002,
"answer": 
"There are 2 collisions related to the yellow cube in the following times: 1.36s, 3.36s. 

from 0.96s to 1.36s, the motion pattern continued with velocity vectors [1.7746, -1.3562, 0.0108]. During this time, the yellow cube moves [0.70984, -0.5424800000000001, 0.00432].

At timestamp 1.36, the yellow cube collided with the brown cube, altering its velocity vector to [1.5673, -1.1967, 0.0068]. The motion pattern continued for 3.36 - 1.36 = 2.0[s] with velocity vectors [1.422, -1.0841, 0.0041]. During this time, the yellow cube moves [2.844, -2.1682, 0.0082].

At timestamp 3.36, the yellow cube collided with the blue cylinder, altering its velocity vector to [1.0557, -0.7995, 0.0082]. The motion pattern continued for 4.32 - 3.36 = 0.96[s] with velocity vectors [1.0557, -0.7995, 0.0082]. During this time, the yellow cube moves [1.0134720000000002, -0.76752, 0.007872].

Therefore, the cumulative movement is [0.70984 + 2.844 + 1.0134720000000002, -0.5424800000000001 + -2.1682 + -0.76752, 0.00432 + 0.0082 + 0.007872] = [4.567312, -3.4782, 0.020392]."





"question": "How much has the yellow cube moved between 0.96s and 4.32s? Please provide the vector components in three dimensions."
"square_error": 16.734104719008002, 
"answer": "There are 2 collisions related to the yellow cube in the following times: 1.36s, 3.36s.\n
from 0.96s to 1.36s, the motion pattern continued with velocity vectors [1.7746, -1.3562, 0.0108]. During this time, the yellow cube moves [0.70984, -0.5424800000000001, 0.00432]\nAt timestamp 1.36, the yellow cube collided with the brown cube, altering its velocity vector to [1.5673, -1.1967, 0.0068].\n
The motion pattern continued for 3.36 - 1.36 = 2.0[s] with velocity vectors [1.422, -1.0841, 0.0041]. During this time, the yellow cube moves [2.844, -2.1682, 0.0082].\n
At timestamp 3.36, the yellow cube collided with the blue cylinder, altering its velocity vector to [1.0557, -0.7995, 0.0082].\n
The motion pattern continued for 4.32 - 3.36 = 0.96[s] with velocity vectors [1.0557, -0.7995, 0.0082]. During this time, the yellow cube moves [1.0134720000000002, -0.76752, 0.007872].\n
Therefore, the cumulative movement is [0.70984 + 2.844 + 1.0134720000000002, -0.5424800000000001 + -2.1682 + -0.76752, 0.00432 + 0.0082 + 0.007872] = [4.567312, -3.4782, 0.020392]."

{"question_id": 7, "question": "How much has the yellow cube moved between 0.96s and 2.4s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "elaborative", "program": [], "square_error": 4.375450213552004, "answer": "There are one collision related to the yellow cube in the following times: 1.36s.\nfrom 0.96s to 1.36s, the motion pattern continued with velocity vectors [1.7746, -1.3562, 0.0108]. During this time, the yellow cube moves [0.70984, -0.5424800000000001, 0.00432]\nAt timestamp 1.36, the yellow cube collided with the brown cube, altering its velocity vector to [1.5673, -1.1967, 0.0068].\nThe motion pattern continued for 2.4 - 1.36 = 1.04[s] with velocity vectors [1.5564, -1.188, -0.0169]. During this time, the yellow cube moves [1.618656, -1.23552, -0.017575999999999998].\nTherefore, the cumulative movement is [0.70984 + 1.618656, -0.5424800000000001 + -1.23552, 0.00432 + -0.017575999999999998] = [2.3284960000000003, -1.778, -0.013255999999999997]."}, 


{"question_id": 8, "question": "How much has the brown cube moved between 2.88s and 3.84s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 0.02513240009599998, "answer": "[0.20788799999999996, -0.076224, 0.005376]"}, 


{"question_id": 9, "question": "How much has the yellow cube moved between 0.0s and 4.32s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 33.80238221412801, "answer": "[6.4888, -4.947568, 0.018248]"}, 

{"question_id": 10, "question": "Where is the gray sphere at 0.8s? Please provide the coordinates in three dimensions.", "question_type": "physical", "question_subtype": "location", "program": [], "square_error": null, "answer": "[-3.6565, -3.1991, 0.2008]"}, {"question_id": 11, "question": "How much has the yellow cube moved between 0.48s and 1.44s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 2.5204507372319993, "answer": "[1.767064, -1.350256, 0.0020399999999999997]"}, {"question_id": 12, "question": "What is the velocity of the yellow cube at 0.8s? Please provide the coordinates in three dimensions.", "question_type": "physical", "question_subtype": "velocity", "program": [], "square_error": null, "answer": "[1.8939, -1.4475, 0.0004]"}, {"question_id": 13, "question": "What is the velocity of the brown cube at 3.88s? Please provide the coordinates in three dimensions.", "question_type": "physical", "question_subtype": "velocity", "program": [], "square_error": null, "answer": "[0.1152, -0.0401, 0.0029]"}, {"question_id": 14, "question": "How much has the gray sphere moved between 0.96s and 2.88s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 11.188569717488, "answer": "[4.012432, 2.424192, 0.0]"}, {"question_id": 15, "question": "How much has the blue cylinder moved between 0.48s and 2.4s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 0.7677757458240005, "answer": "[0.134888, -1.2091040000000002, -0.020592000000000003]"}, {"question_id": 16, "question": "What is the velocity of the blue cylinder at 4.72s? Please provide the coordinates in three dimensions.", "question_type": "physical", "question_subtype": "velocity", "program": [], "square_error": null, "answer": "[0.9267, 0.3478, 0.0082]"}, 

{"question_id": 17, "question": "How much has the yellow cube moved between 2.4s and 3.36s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "elaborative", "program": [], "square_error": 1.2285392456799988, "answer": "There are no collisions related to the yellow cube in between 2.4s and 3.36s.\nfrom 2.4s to 3.36s, the motion pattern continued with velocity vectors [1.2865, -0.9789, -0.0147]. During this time, the yellow cube moves [1.23504, -0.9397439999999999, -0.014112]\nTherefore, the cumulative movement is [1.23504, -0.9397439999999999, -0.014112] = [1.23504, -0.9397439999999999, -0.014112]."}, 

{"question_id": 18, "question": "What is the velocity of the brown cube at 3.2s? Please provide the coordinates in three dimensions.", "question_type": "physical", "question_subtype": "velocity", "program": [], "square_error": null, "answer": "[0.2466, -0.0946, 0.0091]"}, {"question_id": 19, "question": "How much has the yellow cube moved between 1.44s and 4.32s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "elaborative", "program": [], "square_error": 11.106625804992001, "answer": "There are one collision related to the yellow cube in the following times: 3.36s.\nfrom 1.44s to 3.36s, the motion pattern continued with velocity vectors [1.4113, -1.0754, 0.0088]. During this time, the yellow cube moves [2.709696, -2.064768, 0.016896]\nAt timestamp 3.36, the yellow cube collided with the blue cylinder, altering its velocity vector to [1.0557, -0.7995, 0.0082].\nThe motion pattern continued for 4.32 - 3.36 = 0.96[s] with velocity vectors [1.0557, -0.7995, 0.0082]. During this time, the yellow cube moves [1.0134720000000002, -0.76752, 0.007872].\nTherefore, the cumulative movement is [2.709696 + 1.0134720000000002, -2.064768 + -0.76752, 0.016896 + 0.007872] = [3.7231680000000003, -2.832288, 0.024768000000000002]."}, {"question_id": 20, "question": "How much has the blue cylinder moved between 2.88s and 4.8s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "simple", "program": [], "square_error": 1.3618597807839996, "answer": "[1.6103519999999998, 0.18475200000000003, -0.012624]"}, {"question_id": 21, "question": "How much has the blue cylinder moved between 0.96s and 2.4s? Please provide the vector components in three dimensions.", "question_type": "cumulative", "question_subtype": "elaborative", "program": [], "square_error": 0.7677757458240005, "answer": "There are one collision related to the blue cylinder in the following times: 1.36s.\nfrom 0.96s to 1.36s, the motion pattern continued with velocity vectors [-0.0, -0.0, 0.0]. During this time, the blue cylinder moves [-0.0, -0.0, 0.0]\nAt timestamp 1.36, the blue cylinder collided with the brown cube, altering its velocity vector to [0.1311, -1.1734, -0.0114].\nThe motion pattern continued for 2.4 - 1.36 = 1.04[s] with velocity vectors [0.1297, -1.1626, -0.0198]. During this time, the blue cylinder moves [0.134888, -1.2091040000000002, -0.020592000000000003].\nTherefore, the cumulative movement is [-0.0 + 0.134888, -0.0 + -1.2091040000000002, 0.0 + -0.020592000000000003] = [0.134888, -1.2091040000000002, -0.020592000000000003]."}, {"question_id": 22,