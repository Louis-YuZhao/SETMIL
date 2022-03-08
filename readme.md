## Spatial Encoding Transformer-based Multiple Instance Learning for Pathological Image Analysis ##

Code for paper titled "SETMIL: Spatial Encoding Transformer-based Multiple Instance Learning for Pathological Image Analysis" submitted to MICCAI2022. The basic method and applications are introduced as follows:

![avatar](./Figure1.png)

<center>The overall framework of the proposed spatial encoding transformer-based MIL (SETMIL). It consists of three main stages including (1) position-preserving encoding (PPE) to transform huge-size WSI to a small-size position encoded feature map, (2) transformer-based pyramid multi-scale fusion (TPMF) aiming at modifying the feature map and enriching representation with multi-scale context information, and (3) spatial encoding transformer (SET)-based bag embedding, which generates a high-level bag representation comprehensively considering all instance representations in a fully trainable way and leverages a joint absolute-relative position encoding mechanism to encode the position and context information. </center>


![avatar](./Figure2.png)

 Sub-figure (A) illustrates the transformer-based pyramid multi-scale fusion module, which consists of three tokens-to-token (T2T) modules \cite{yuan2021tokens} working in a pyramid arrangement to modify the feature map and enrich a representation (token) with multi-scale context information. Each tokens-to-token module has a soft-split and reshape process together with a transformer layer\cite{vaswani2017attention}. Sub-figure (B) shows a example heatmap for model interpretability. Colors reflect the prediction contribution of each local patch.

# Dependencies #
    python==3.7.11
    torch==1.7.0
    torchvision==0.8.1+cu110
    rich==10.16.1
    yacs==0.1.8
    einops==0.3.2
    openslide-python==1.1.2
    opencv-python==4.5.4.60
    setuptools==58.0.4
    matplotlib==3.5.1
    Pillow==8.4.0
    scikit-image==0.19.1
    scikit-learn==1.0.1
    scipy==1.7.3
    cffi==1.15.0
    numpy==1.20.3
    pandas==1.3.5
    pkgconfig==1.5.1
    pycparser==2.20
    python-dateutil==2.8.1
    pytz==2020.1
    pyvips==2.1.12
    six==1.16.0
# Pathological Image Analysis  #
This code uses the centralized configs. Before using this code, a config file needs to be edited to assign necessary parameters. A sample config file named 'default.yaml' is provided as the reference.
    
    ./trans/configs/default.yaml

1、First, in order to split the WSI into patches, execute the following script.

    python WSI_cropping.py 
      --dataset /folder/  
      --output /output_patch/
      --scale 20 --patch_size 1120 --num_threads 16

2、Then, extract features from each patch. a pre-trained feature extractor can be utilized here (e.g. EfficientNet-B0 trained on the ImageNet). 

    python extract_feature.py 
      -- WSI id1
        -- patch1 feat.pkl --> dict({'val': [1280] , 'tr': })
        -- patch2 feat.pkl
      -- WSI id2
        -- patch1 feat.pkl
        -- patch2 feat.pkl


3、Next, combine features of one WSI. 

    python merge_patch_feat.py --cfg configs/*.yaml


4、Finally, we can train and evaluate the model with preprocessed data 

    python ./trans/main.py --cfg ./trans/configs/*.yaml
 
