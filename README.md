# CRYD  
首先用四个模型分别对测试集进行处理，先运行DeGCN，HDBN，SkateFormer中的ensemble文件，最后再运行CTR-GCN-main中的ensemble文件实现对不同模型和模态的合成，具体命令如下:  
## DeGCN
* DeGCN_joint:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/joint.yaml --work-dir CRYD-main/DeGCN/work_dir_test --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/joint/epoch_67_34371.pt --device 0  
* DeGCN_bone:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/bone.yaml --work-dir CRYD-main/DeGCN/DeGCN/work_dir_test_bone --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/bone/epoch_76_38988.pt --device 0  
* DeGCN_joint_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/joint_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_joint_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/joint_motion/epoch_74_37962.pt --device 0  
* DeGCN_bone_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/bone_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_bone_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/bone_motion/epoch_74_37962.pt --device 0  
## former  
* former_joint:
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_J.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_joint --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/joint/runs-62-15872.pt --device 0
* former_bone
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_B.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_bone --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/bone/runs-56-7168.pt --device 0
*former_joint_motion:
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JM.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_joint_motion --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/joint_motion/runs-57-7296.pt --device 0
*former_bone_motion:
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BM.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_bone_motion --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/bone_motion/runs-59-7552.pt --device 0
*former_k2:
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_k2 --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/k2/runs-56-7168.pt --device 0
*former_k2_motion:
python CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2M.yaml --work-dir CRYD/HDBN/ICMEW2024-Track10-main/work_dir_test_former_k2_motioin --phase test --save-score True --weights CRYD/HDBN/ICMEW2024-Track10-main/weights/k2_motion/runs-59-7552.pt --device 0
