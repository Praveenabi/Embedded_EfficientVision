## Convolutional Neural Networks(CNN)
- CNNs have emerged as the predominant approach for computer vision tasks due to their natural inductive locality biases and translation equivalence.
- However, the extensive computation of standard CNNs renders them unsuitable for deployment on resource-constrained mobile devices.
- Many efficient design principles have been proposed to enhance the computational efficiency of CNNs for mobile devices. Lightweight CNN Design Principles are 
    - **Separable Convolutions**(MobileNets)
    - **Inverted Residual Bottlenecks**(MobileNetV2)
    - **Channel Shuffling**(ShuffleNet, ShuffleNetv2)
    - **Structural Re-parameterization**(RepVGG, ACNet)

## Vision Transformers (ViTs)
- ViT adapts the transformer architecture to achieve state-of-the-art performance on large-scale image recognition tasks, surpassing that of CNNs.
- Building on the competitive performance of ViTs, subsequent works have sought to incorporate spatial inductive biases to enhance their stability and performance(CoatNet, CMT), design more efficient self-attention operations (CSwin Transformer, Biformer), and adapt ViTs to a diverse range of computer vision tasks.
-  Heavy-weighted, requiring substantial computation and memory footprint [39, 58]. That makes them unsuitable for mobile devices with limited resources.
-  Many efficient design principles have been proposed to enhance the computational efficiency of ViTs for mobile devices. Lightweight ViT Design Principles are
    - **Hybrid Networks**(MobileFormer,MobileViT)
    - **Self Attention with Linear Complexity**(MobileViTv2)
    - **Dimension Consistent Design Principles**(EfficientFormer,EfficientFormerv2)


## CNN Models
- **RegNet** 
    - [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678)

## Lightweight CNN Models
- **ACNet**
    - [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/pdf/1908.03930)
- **MobileNets**
    - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)
- **MobileNet V2**
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)
- **MobileNet V3**
    - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244)
- **MobileOne**
    - [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/pdf/2206.04040)
- **ParCNet**
    - [ParC-Net: Position Aware Circular Convolution with Merits from ConvNets and Transformer](https://arxiv.org/pdf/2203.03952)
- **ShuffleNet**
    - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083)
- **ShuffleNet V2**
    - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164)
- **RepVGG**
    - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697)
- **ResNet**
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
- **QARepVGG**
    - [Make RepVGG Greater Again: A Quantization-aware Approach](https://arxiv.org/pdf/2212.01593) 





## ViT Models
- **Vanilla ViT** 
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
- **Biformer**
    - [BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/pdf/2303.08810)
- **CoAtNet** 
    - [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803)
- **CMT** 
    - [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263)
- **CSwin Transformer**
    - [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/pdf/2107.00652)
- **DeiT** 
    - [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877)
- **DETR**
    - [End-to-end object detection with Transformers](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
- **LeViT**
    - [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136)
- **LV-ViT**
    - [All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/pdf/2104.10858)
- **Mask2Former**
    - [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/pdf/2112.01527)
- **NViT** 
    - [NViT: Vision Transformer Compression and Parameter Redistribution](https://arxiv.org/pdf/2110.04869v1)
- **PoolFormer**
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/pdf/2111.11418)
- **Pyramid Vision Transformer**
    - [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122)
- **SaViT**
    - [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://papers.nips.cc/paper_files/paper/2022/file/3b11c5cc84b6da2838db348b37dbd1a2-Paper-Conference.pdf)
- **SegFormer**
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203)
- **Swin Transformer** 
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)
- **Swin Transformer V2**
    - [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883)
- **ViT-22B**
    - [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/pdf/2302.05442)


## Lightweight ViT Models
- **Conv2Former**
    - [Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition](https://arxiv.org/pdf/2211.11943)
- **EdgeViTs**
    - [EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers](https://arxiv.org/pdf/2205.03436)
- **EfficientViT**
    - [EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/pdf/2305.07027)
- **EfficientFormer**
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191)
- **EfficientFormerV2**
    - [Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/pdf/2212.08059)
- **FastViT**
    - [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/pdf/2303.14189)
- **Mobile-Former**
    - [Mobile-Former: Bridging MobileNet and Transformer](https://arxiv.org/pdf/2108.05895)
- **MobileViG**
    - [MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications](https://arxiv.org/pdf/2307.00395)
- **MobileViT**
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/pdf/2110.02178)
- **MobileViT V2**
    - [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/pdf/2206.02680)




## Training Recipe of Lightweight ViTs
- **AdamW Optimizer**
    - [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101)
- **Cosine Learning Rate Scheduler**
- **Mixup**
    - [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412)
- **Auto-augmentation**
    - [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/pdf/1805.09501)
- **Random Erasing**
    - [Random Erasing Data Augmentation](https://arxiv.org/pdf/1708.04896)
- **Label Smoothing**
    - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567)
- **Knowledge Distillation**


## Differences Between Lightweight ViTs and CNNs
- **Block Structure, Macro/Micro Designs**
- **Block Structure - MetaFormer (ViTs), Inverted Residual Bottleneck (CNNs)**

## Block Design
- **Separate token mixer and channel mixer**
    - The block structure of lightweight ViTs(EfficientFormerV2, EfficientFomer,MobileViTV2) incorporates an important design feature, namely the separate token mixer and channel mixer.
    - According to the recent research, the effectiveness of ViTs primarily originates
    from their general token mixer and channel mixer architecture, i.e., the MetaFormer architecture, rather than the equipped specific token mixer.
    - [MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452)
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/pdf/2111.11418)
- **Reducing Expansion Ratio and increasing width**
    -  In vanilla ViTs, the expansion ratio in the channel mixer is typically set to 4, making the hidden dimension of the Feed Forward Network (FFN) module 4× wider than the input dimension.
    -  It thus consumes a significant portion of the computation resource, thereby contributing substantially to the overall inference time(explained in SaViT).
    -  To alleviate this bottle neck, recent works(LV-ViT,LeViT) employ a narrower FFN.
    - NViT  point out that there exists a significant amount of channel redundancy in FFN. Therefore, it is reasonable to use a smaller expansion ratio.
    - Consequently, with the smaller expansion ratio, we can increase the network width to remedy the large parameter reduction. 
    - We double the channels after each stage, ending up with 48, 96, 192, and 384 channels for each stage, respectively

## Macro Design
- **Stem**
    - Early Convolution
- **Downsampling Layers**
    - Deeper Downsampling Layers
- **Classifier**
    - simple classifier - a global average pooling layer and a linear layer
- **Overall Stage ratio**
    - Stage ratio represents the ratio of the number of blocks in different stages, thereby indicating the distribution of computation across the stages.
    - we employ a stage ratio of 1:1:7:1 for the network
    - We then increase the network depth to 2:2:14:2, achieving a deeper layout

## Micro Design
- **Kernel Size Selection**
    - [**ConvNext](https://arxiv.org/pdf/2201.03545), [**RePLKNet](https://arxiv.org/pdf/2203.06717), employs Large kernel-sized convolution, exhibiting the performance gain.
    - However, large kernel-sized convolution is not friendly for mobile devices, due to its computation complexity and memory access costs
    - To ensure the inference efficiency on the mobile device, we prioritize the simple 3 × 3 convolutions
- **SE Layer Placement**
    - Advantage of self-attention module compared with convolution is the ability to adapt weights according to input, known as the data-driven attribute
        - [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025)
        - [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
    - As a channel wise attention module, SE layers can compensate for the limitation of convolutions in lacking data-driven attributes, bringing better performance
        - [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
        - [ParC-Net: Position Aware Circular Convolution with Merits from ConvNets and Transformer](https://arxiv.org/pdf/2203.03952)
    - However, as shown in TresNet, stages with low-resolution feature maps get a smaller accuracy benefit, compared to stages with higher resolution feature maps.
        - [TResNet: High Performance GPU-Dedicated Architecture](https://arxiv.org/pdf/2003.13630)
    - Meanwhile, along with performance gains, SE layers also introduce non-negligible computational costs
    - Therefore, we design a strategy to utilize SE layers in a cross-block manner, i.e., adopting the SE layer in the 1st, 3rd, 5th, ... block in each stage, to maximize the accuracy benefit with a minimal latency increment




## Survey Papers
- [Efficient High-Resolution Deep Learning: A Survey](https://arxiv.org/pdf/2207.13050)
- [A Survey of Algorithmic and Hardware Optimization Techniques for Vision Convolutional Neural Networks on FPGAs](https://cosicdatabase.esat.kuleuven.be/backend/publications/files/journal/3329)