## Convolutional Neural Networks(CNN)
- CNNs have emerged as the predominant approach for computer vision tasks due to their natural inductive locality biases and translation equivalence.
- However, the extensive computation of standard CNNs renders them unsuitable for deployment on resource-constrained mobile devices.
- CNN Models
  - **RegNet** - [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678)

##   Lightweight CNN Design Principles
- **Separable Convolutions**
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)
- **Inverted Residual Bottlenecks**
  - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)
- **Channel Shuffling**
  - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083)
  - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164)
- **Structural Re-parameterization**
  - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697) 
  - [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/pdf/1908.03930)

##   Representative Lightweight CNN Models
- **MobileNets**
- **MobileNet V2**
- **ShuffleNet**
- **ShuffleNet V2**
- **RepVGG**

##   Vision Transformers (ViTs)
- ViT adapts the transformer architecture to achieve state-of-the-art performance on    large-scale image recognition tasks, surpassing that of CNNs.
- **ViT Models**
    - **Vanilla ViT** - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
    - **DeiT** - [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877)
    - **CoAtNet** - [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803)
    - **CMT** - [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263)
    - **Swin Transformer** - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)

##   MetaFormer
- [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/pdf/2111.11418)
- [MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452)


##   ViT Applications

    * **Image Classification**
        * \[39] -   Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 1, 2, 4
        * \[63] -   Wu, H.; Chen, J.; Xu, L.; Dai, X.; Yuan, L.; Liu, Z.; and Guo, Y. Cvt: Introducing convolutions to vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22–31, 2021. 1, 4
    * **Semantic Segmentation**
        * \[6] -   Cao, Y.; Wang, Y.; Xu, J.; Lin, S.; Chen, J.; and Hu, H. Swin transformer for semantic segmentation. In Proceedings of the AAAI conference on artificial intelligence, pages 11536–11543, 2022. 1
        * \[66] -   Zheng, S.; Lu, J.; Zhao, H.; Zhu, Y.; Luo, Z.; Wang, Y.; Xu, F.; Li, Y.; Wang, S.; Bu, S.; et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6881–6890, 2021. 1
    * **Object Detection**
        * \[4] -   Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Gros- se, A.; and Chintala, S. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 1
        * \[34] -   Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollár, P. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 1

##   Challenges of ViTs on Mobile Devices

    * **Large Model Sizes and High Latency**
        * \[11] -   Chen, J.; Jiang, B.; Liu, Z.; Xu, L.; Chen, D.; Yuan, L.; Liu, Z.; and Guo, Y. Chatt: Cross-head attention transformer tracker. IEEE Transactions on Image Processing, 31:3188–3200, 2022. 1
        * \[40] -   Liu, S.; Gao, Y.; Liu, X.; and Song, S. Improving transformer-based object detection by promoting global-local attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3059–3068, 2021. 1
    * **Performance Degradation with Direct Size Reduction**
        * \[5] -   Chen, D.; Fan, B.; Panda, R.; and Vasconcelos, N. Mobile-vit: Light- weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8

##   Lightweight ViT Design

    * **Efficient Design Principles for Mobile Devices**
        * \[5, 35, 46, 49] -   \[5] -   Chen, D.; Fan, B.; Panda, R.; and Vasconcelos, N. Mobile-vit: Light- weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8, \[35] -   Yuan, L.; Fu, H.; Lin, T.; and Cui, S. Efficientformerv2: Scaling vision transformers for mobile inference. arXiv preprint arXiv:2209.10855, 2022. 2, 3, 4, 8, 9, 10, 11, 12, 13, \[46] -   Mehta, S. and Rastegari, R. Mobilevit: Light-weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, \[49] -   Kim, Y.; Kim, S.; and Cho, M. Edgevit: Compressing vision transformer for mobile devices with network distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 631–641, 2022. 2, 3, 6, 7
    * **Hybrid Architectures (CNNs + ViTs)**
        * \[5, 46] -   \[5] -   Chen, D.; Fan, B.; Panda, R.; and Vasconcelos, N. Mobile-vit: Light- weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8, \[46] -   Mehta, S. and Rastegari, R. Mobilevit: Light-weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15
    * **Linear Complexity Self-Attention**
        * \[47] -   Liu, S.; Jiang, L.; Song, S.; and Gao, Y. Not all pixels are equal: Modulation transformer for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 966–975, 2021. 2, 3, 4
    * **Dimension-Consistent Design**
        * \[35, 36] -   \[35] -   Yuan, L.; Fu, H.; Lin, T.; and Cui, S. Efficientformerv2: Scaling vision transformers for mobile inference. arXiv preprint arXiv:2209.10855, 2022. 2, 3, 4, 8, 9, 10, 11, 12, 13, \[36] -   Yuan, L.; Yun, S.; Tay, F. E.; Dao, M.; and Lu, Y. Efficientformer: Vision transformers at mobile net speed. Advances in Neural Information Processing Systems, 35:1234–1246, 2022. 1, 2, 3, 4, 5, 9, 10, 12, 13, 15

##   Lightweight ViT Models
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
- **LeViT**
    - [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136)
- **MobileOne**
    - [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/pdf/2206.04040)
- **MobileViG**
    - [MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications](https://arxiv.org/pdf/2307.00395)
- **MobileViT**
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/pdf/2110.02178)



##   CNNs for Edge Devices

    * **Optimized Convolution Operations**
        * \[54, 73] -   \[54] -   Peemen, T. H.; Setio, A. A. W.; Ben Geiger, B. C.; and van Ginneken, B. Comparison of convolution implementations on cpu, gpu, and fpga for medical image analysis. In 2013 IEEE international conference on image processing, pages 960–964. IEEE, 2013. 2, \[73] -   Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017. 2, 5, 6

    ##   Similarities Between Lightweight ViTs and CNNs

    * **Convolutional Modules for Local Representations**
        * \[46, 47, 49, 61] -   \[46] -   Mehta, S. and Rastegari, R. Mobilevit: Light-weight vision transformer with inverted residuals. arXiv preprint arXiv:2110.02178, 2021. 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, \[47] -   Liu, S.; Jiang, L.; Song, S.; and Gao, Y. Not all pixels are equal: Modulation transformer for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 966–975, 2021. 2, 3, 4, \[49] -   Kim, Y.; Kim, S.; and Cho, M. Edgevit: Compressing vision transformer for mobile devices with network distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 631–641, 2022. 2, 3, 6, 7, \[61] -   Stock, P. and Galliani, L. Kerasnetsparse: Neural network sparsity zoo. arXiv preprint arXiv:1901.05834, 2019. 2



##   Training Recipe of Lightweight ViTs
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


##   Differences Between Lightweight ViTs and CNNs
- **Block Structure, Macro/Micro Designs**
- **Block Structure - MetaFormer (ViTs), Inverted Residual Bottleneck (CNNs)**

##   Block Design

    * **MetaFormer Architecture**
        * \[69, 70] -   \[69] -   Yu, W.; Luo, M.; Zhou, F.; Si, C.; Zhou, Y.; Jiang, X.; and Zou, C. Metanet: The simplicity of metamer theory meets self- supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1260–1270, 2023. 2, 3, 6, 7, \[70] -   Yu, W.; Luo, M.; Zhou, F.; Si, C.; Zhou, Y.; Jiang, X.; and Zou, C. Metaformer is actually what you need for vision. arXiv preprint arXiv:2111.11430, 2021. 3, 4
    * **RepViT Block**

##   Macro Design
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

##   Micro Design
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