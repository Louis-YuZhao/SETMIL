SETMIL: Spatial Encoding Transformer-based Multiple Instance Learning for Pathological Image Analysis

Code for paper titled "SETMIL: Spatial Encoding Transformer-based Multiple Instance Learning for Pathological Image Analysis"

Before you use this code, you need to configure default.yaml. Here we provide a reference. 

1、First, in order to split the WSI into patches, execute the following script .

python WSI_cropping.py 
  --dataset /folder/  
  --output /output_patch/
  --scale 20 --patch_size 1120 --num_threads 16

2、Then, extract features from each patch. It's worth noting that you need to pretrain a feature extractor before using it. 

python extract_feature.py 
- WSI id1
    - patch1 feat.pkl --> dict({'val': [1280] , 'tr': })
    - patch2 feat.pkl
- WSI id2
    - patch1 feat.pkl
    - patch2 feat.pkl


3、Next, combine features of one WSI. 

python merge_patch_feat.py --cfg configs/*.yaml


4、Finally, you can train with preprocessed data 

python main.py --cfg configs/*.yaml

