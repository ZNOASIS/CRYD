# CRYD  
首先用四个模型分别对测试集进行处理，先运行DeGCN，HDBN，SkateFormer中的ensemble文件，最后再运行CTR-GCN-main中的ensemble文件实现对不同模型和模态的合成，具体命令如下:  
## DeGCN:
* DeGCN_joint:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/joint.yaml --work-dir CRYD-main/DeGCN/work_dir_test --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/joint/epoch_67_34371.pt --device 0  
* DeGCN_bone:  
python CRYD-main/DeGCN/DeGCN_pytorch-main/main.py --config CRYD-main /DeGCN/DeGCN_pytorch-main/config/bone.yaml --work-dir CRYD-main/DeGCN/DeGCN/work_dir_test_bone --phase test --save-score True --weights CRYD-main/DeGCN/DeGCN_pytorch-main/weights/bone/epoch_76_38988.pt --device 0  
* DeGCN_joint_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/joint_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_joint_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/joint_motion/epoch_74_37962.pt --device 0  
* DeGCN_bone_motion:  
python CRYD/DeGCN/DeGCN_pytorch-main/main.py --config CRYD /DeGCN/DeGCN_pytorch-main/config/bone_motion.yaml --work-dir CRYD/DeGCN/work_dir_test_bone_motion --phase test --save-score True --weights CRYD/DeGCN/DeGCN_pytorch-main/weights/bone_motion/epoch_74_37962.pt --device 0  
