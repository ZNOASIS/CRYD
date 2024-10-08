# CRYD  
## 简述
首先用四个模型分别对测试集进行处理，先运行DeGCN，HDBN，SkateFormer中的ensemble文件，最后再运行CTR-GCN-main中的ensemble文件实现对不同模型和模态的合成。
## 用四个模型分别对测试集测试并得到各自的权重
**四个模型分别是：**
* DeGCN
* former
* SkateFormer
* CTR-GCN<br>
### DeGCN
* DeGCN_joint:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/joint.yaml --work-dir CRYD-main/DeGCN/work_dir_test --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/joint/epoch_67_34371.pt --device 0  
* DeGCN_bone:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/bone.yaml --work-dir CRYD-main/DeGCN/DeGCN/work_dir_test_bone --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/bone/epoch_76_38988.pt --device 0  
* DeGCN_joint_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/joint_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_joint_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/joint_motion/epoch_74_37962.pt --device 0  
* DeGCN_bone_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/bone_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_bone_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/bone_motion/epoch_74_37962.pt --device 0  
### former  
* former_joint:
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_J.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_joint --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/joint/runs-62-15872.pt --device 0
* former_bone
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_B.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_bone --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/bone/runs-56-7168.pt --device 0
* former_joint_motion:
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JM.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_joint_motion --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/joint_motion/runs-57-7296.pt --device 0
* former_bone_motion:
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BM.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_bone_motion --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/bone_motion/runs-59-7552.pt --device 0
* former_k2:
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_k2 --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/k2/runs-56-7168.pt --device 0
* former_k2_motion:
python CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2M.yaml --work-dir CRYD-main/HDBN/ICMEW2024-Track10-main/work_dir_test_former_k2_motioin --phase test --save-score True --weights CRYD-main/HDBN/ICMEW2024-Track10-main/weights/k2_motion/runs-59-7552.pt --device 0
### SkateFormer
* SkateFormer_joint：
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_j.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test_joint --phase test --save-score True --weights CRYD-main/SkateFormer-main/weights/joint/runs-468-119808.pt --device 0
* SkateFormer_bone:
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_b.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test_bone --phase test --save-score True --weights CRYD-main/SkateFormer-main/weights/bone/runs-500-256500.pt --device 0
* SkateFormer_joint_motion:
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_jm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test_joint_motion --phase test --save-score True --weights CRYD-main/SkateFormer-main/weights/joint_motion/runs-467-119552.pt --device 0
* SkateFormer_bone_motion:
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_bm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test_bone_motion --phase test --save-score True --weights CRYD-main/SkateFormer-main/weights/bone_motion/runs-451-115456.pt --device 0
### CTR-GCN
* CTR-GCN_joint:
python CRYD-main/CTR-GCN-main/main.py --config CRYD-main/CTR-GCN-main/config/nturgbd-cross-subject/default.yaml --work-dir CRYD-main/CTR-GCN-main/work_dir_test_joint --phase test --save-score True --weights CRYD-main/CTR-GCN-main/weights/joint/runs-98-25088.pt  --device 0
* CTR-GCN_bone:
python CRYD-main/CTR-GCN-main/main.py --config CRYD-main/CTR-GCN-main/config/nturgbd-cross-subject/bone.yaml --work-dir CRYD-main/CTR-GCN-main/work_dir_test_bone --phase test --save-score True --weights CRYD-main/CTR-GCN-main/weights/bone/runs-63-16128.pt  --device 0
* CTR-GCN_joint_motion:
python CRYD-main/CTR-GCN-main/main.py --config CRYD-main/CTR-GCN-main/config/nturgbd-cross-subject/joint_motion.yaml --work-dir CRYD-main/CTR-GCN-main/work_dir_test_joint_motion --phase test --save-score True --weights CRYD-main/CTR-GCN-main/weights/joint_motion/runs-84-21504.pt  --device 0
* CTR-GCN_bone_motion:
python CRYD-main/CTR-GCN-main/main.py --config CRYD-main/CTR-GCN-main/config/nturgbd-cross-subject/bone_motion.yaml --work-dir CRYD-main/CTR-GCN-main/work_dir_test_bone_motion --phase test --save-score True --weights CRYD-main/CTR-GCN-main/weights/bone_motion/runs-95-24320.pt  --device 0
## 权重融合
* 执行下面三个命令，对DeGCN，HDBN，SkateFormer的权重各自融合。
python CRYD-main/HDBN/ICMEW2024-Track10-main/ensemble.py  --output-dir CRYD-main/result1
python CRYD-main/DeGCN/DeGCN_pytorch-main/ensemble.py --output-dir CRYD-main/result2
python CRYD-main/SkateFormer-main/ensemble.py  --output-dir CRYD-main/result3
* 执行下面命令，融合CTRGCN各自模态，同时融合以上四个模型。
python CRYD-main/CTR-GCN-main/ensemble.py  --output-dir autodl-tmp/result4
* 执行下面命令，将格式转换为numpy并命名为pred.npy存放在eval.py的同一目录下。
python CTR-GCN-main/eval.py --pred_path autodl-tmp/result4/fused_scores.pkl
