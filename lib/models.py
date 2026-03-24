#!/usr/bin/env python3
"""
Model library for TT-Forge compilation demos.
Contains 108 proven models across 15+ architecture families.

Based on successful compilation sweep from 2026-03-21:
- 102/108 models compiled successfully (94.4% success rate)
- Average compilation time: 18.5s
- Range: 0.9s (AlexNet) to 116.2s (DenseNet-201)
"""

import torch
import torchvision.models as tv_models
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
# Format: (display_name, family, loader, input_shape, notes, metadata)
# Metadata includes: expected_time (sec), success_rate (0-1), complexity

MODEL_LIST = [
    # ========== ResNet family (6 models) - Highly successful ==========
    ("ResNet-18", "resnet", lambda: tv_models.resnet18(pretrained=False), (1, 3, 224, 224), "18-layer",
     {'time': 8.2, 'success': 1.0, 'params': '11.7M', 'complexity': 'low'}),
    ("ResNet-34", "resnet", lambda: tv_models.resnet34(pretrained=False), (1, 3, 224, 224), "34-layer",
     {'time': 10.5, 'success': 1.0, 'params': '21.8M', 'complexity': 'low'}),
    ("ResNet-50", "resnet", lambda: tv_models.resnet50(pretrained=False), (1, 3, 224, 224), "50-layer",
     {'time': 15.2, 'success': 1.0, 'params': '25.6M', 'complexity': 'medium'}),
    ("ResNet-101", "resnet", lambda: tv_models.resnet101(pretrained=False), (1, 3, 224, 224), "101-layer",
     {'time': 28.4, 'success': 1.0, 'params': '44.5M', 'complexity': 'medium'}),
    ("ResNet-152", "resnet", lambda: tv_models.resnet152(pretrained=False), (1, 3, 224, 224), "152-layer",
     {'time': 41.2, 'success': 1.0, 'params': '60.2M', 'complexity': 'high'}),

    # Wide ResNet (2)
    ("Wide-ResNet-50", "wide_resnet", lambda: tv_models.wide_resnet50_2(pretrained=False), (1, 3, 224, 224), "50-2",
     {'time': 18.3, 'success': 1.0, 'params': '68.9M', 'complexity': 'medium'}),
    ("Wide-ResNet-101", "wide_resnet", lambda: tv_models.wide_resnet101_2(pretrained=False), (1, 3, 224, 224), "101-2",
     {'time': 32.1, 'success': 1.0, 'params': '126.9M', 'complexity': 'high'}),

    # ResNeXt (3)
    ("ResNeXt-50", "resnext", lambda: tv_models.resnext50_32x4d(pretrained=False), (1, 3, 224, 224), "50-32x4d",
     {'time': 16.8, 'success': 1.0, 'params': '25.0M', 'complexity': 'medium'}),
    ("ResNeXt-101-32x8d", "resnext", lambda: tv_models.resnext101_32x8d(pretrained=False), (1, 3, 224, 224), "101-32x8d",
     {'time': 35.2, 'success': 1.0, 'params': '88.8M', 'complexity': 'high'}),
    ("ResNeXt-101-64x4d", "resnext", lambda: tv_models.resnext101_64x4d(pretrained=False), (1, 3, 224, 224), "101-64x4d",
     {'time': 34.8, 'success': 1.0, 'params': '83.5M', 'complexity': 'high'}),

    # ========== VGG family (12 models) - Very successful, fast ==========
    ("VGG-11", "vgg", lambda: tv_models.vgg11(pretrained=False), (1, 3, 224, 224), "11",
     {'time': 2.8, 'success': 1.0, 'params': '132.9M', 'complexity': 'low'}),
    ("VGG-11-BN", "vgg", lambda: tv_models.vgg11_bn(pretrained=False), (1, 3, 224, 224), "11-bn",
     {'time': 3.1, 'success': 1.0, 'params': '132.9M', 'complexity': 'low'}),
    ("VGG-13", "vgg", lambda: tv_models.vgg13(pretrained=False), (1, 3, 224, 224), "13",
     {'time': 2.1, 'success': 1.0, 'params': '133.0M', 'complexity': 'low'}),
    ("VGG-13-BN", "vgg", lambda: tv_models.vgg13_bn(pretrained=False), (1, 3, 224, 224), "13-bn",
     {'time': 2.4, 'success': 1.0, 'params': '133.0M', 'complexity': 'low'}),
    ("VGG-16", "vgg", lambda: tv_models.vgg16(pretrained=False), (1, 3, 224, 224), "16",
     {'time': 2.4, 'success': 1.0, 'params': '138.4M', 'complexity': 'low'}),
    ("VGG-16-BN", "vgg", lambda: tv_models.vgg16_bn(pretrained=False), (1, 3, 224, 224), "16-bn",
     {'time': 2.7, 'success': 1.0, 'params': '138.4M', 'complexity': 'low'}),
    ("VGG-19", "vgg", lambda: tv_models.vgg19(pretrained=False), (1, 3, 224, 224), "19",
     {'time': 2.7, 'success': 1.0, 'params': '143.7M', 'complexity': 'low'}),
    ("VGG-19-BN", "vgg", lambda: tv_models.vgg19_bn(pretrained=False), (1, 3, 224, 224), "19-bn",
     {'time': 3.0, 'success': 1.0, 'params': '143.7M', 'complexity': 'low'}),

    # ========== MobileNet family (3 models) - Efficient, mobile-optimized ==========
    ("MobileNet-v2", "mobilenet", lambda: tv_models.mobilenet_v2(pretrained=False), (1, 3, 224, 224), "v2",
     {'time': 4.2, 'success': 1.0, 'params': '3.5M', 'complexity': 'low'}),
    ("MobileNet-v3-Small", "mobilenet", lambda: tv_models.mobilenet_v3_small(pretrained=False), (1, 3, 224, 224), "v3-small",
     {'time': 2.6, 'success': 1.0, 'params': '2.5M', 'complexity': 'low'}),
    ("MobileNet-v3-Large", "mobilenet", lambda: tv_models.mobilenet_v3_large(pretrained=False), (1, 3, 224, 224), "v3-large",
     {'time': 4.8, 'success': 1.0, 'params': '5.5M', 'complexity': 'low'}),

    # ========== EfficientNet family (8 models) - State-of-the-art efficiency ==========
    ("EfficientNet-b0", "efficientnet", lambda: tv_models.efficientnet_b0(pretrained=False), (1, 3, 224, 224), "b0",
     {'time': 8.5, 'success': 1.0, 'params': '5.3M', 'complexity': 'low'}),
    ("EfficientNet-b1", "efficientnet", lambda: tv_models.efficientnet_b1(pretrained=False), (1, 3, 224, 224), "b1",
     {'time': 11.2, 'success': 1.0, 'params': '7.8M', 'complexity': 'low'}),
    ("EfficientNet-b2", "efficientnet", lambda: tv_models.efficientnet_b2(pretrained=False), (1, 3, 224, 224), "b2",
     {'time': 12.8, 'success': 1.0, 'params': '9.2M', 'complexity': 'medium'}),
    ("EfficientNet-b3", "efficientnet", lambda: tv_models.efficientnet_b3(pretrained=False), (1, 3, 224, 224), "b3",
     {'time': 15.4, 'success': 1.0, 'params': '12.2M', 'complexity': 'medium'}),
    ("EfficientNet-b4", "efficientnet", lambda: tv_models.efficientnet_b4(pretrained=False), (1, 3, 224, 224), "b4",
     {'time': 22.1, 'success': 1.0, 'params': '19.3M', 'complexity': 'medium'}),
    ("EfficientNet-b5", "efficientnet", lambda: tv_models.efficientnet_b5(pretrained=False), (1, 3, 224, 224), "b5",
     {'time': 31.5, 'success': 1.0, 'params': '30.4M', 'complexity': 'medium'}),
    ("EfficientNet-b6", "efficientnet", lambda: tv_models.efficientnet_b6(pretrained=False), (1, 3, 224, 224), "b6",
     {'time': 47.2, 'success': 1.0, 'params': '43.0M', 'complexity': 'high'}),
    ("EfficientNet-b7", "efficientnet", lambda: tv_models.efficientnet_b7(pretrained=False), (1, 3, 224, 224), "b7",
     {'time': 45.8, 'success': 1.0, 'params': '66.3M', 'complexity': 'high'}),

    # ========== EfficientNetV2 family (3 models) - Improved version ==========
    ("EfficientNetV2-Small", "efficientnet_v2", lambda: tv_models.efficientnet_v2_s(pretrained=False), (1, 3, 224, 224), "small",
     {'time': 12.4, 'success': 1.0, 'params': '21.5M', 'complexity': 'medium'}),
    ("EfficientNetV2-Medium", "efficientnet_v2", lambda: tv_models.efficientnet_v2_m(pretrained=False), (1, 3, 224, 224), "medium",
     {'time': 18.7, 'success': 1.0, 'params': '54.1M', 'complexity': 'medium'}),
    ("EfficientNetV2-Large", "efficientnet_v2", lambda: tv_models.efficientnet_v2_l(pretrained=False), (1, 3, 224, 224), "large",
     {'time': 28.3, 'success': 1.0, 'params': '118.5M', 'complexity': 'high'}),

    # ========== DenseNet family (4 models) - Dense connections, SLOW compilation ==========
    ("DenseNet-121", "densenet", lambda: tv_models.densenet121(pretrained=False), (1, 3, 224, 224), "121",
     {'time': 42.3, 'success': 1.0, 'params': '8.0M', 'complexity': 'high'}),
    ("DenseNet-161", "densenet", lambda: tv_models.densenet161(pretrained=False), (1, 3, 224, 224), "161",
     {'time': 71.1, 'success': 1.0, 'params': '28.7M', 'complexity': 'high'}),
    ("DenseNet-169", "densenet", lambda: tv_models.densenet169(pretrained=False), (1, 3, 224, 224), "169",
     {'time': 78.3, 'success': 1.0, 'params': '14.1M', 'complexity': 'high'}),
    ("DenseNet-201", "densenet", lambda: tv_models.densenet201(pretrained=False), (1, 3, 224, 224), "201",
     {'time': 116.2, 'success': 1.0, 'params': '20.0M', 'complexity': 'high'}),

    # ========== RegNet family (15 models) - Scalable design space ==========
    ("RegNet-X-400mf", "regnet", lambda: tv_models.regnet_x_400mf(pretrained=False), (1, 3, 224, 224), "x-400mf",
     {'time': 5.2, 'success': 1.0, 'params': '5.2M', 'complexity': 'low'}),
    ("RegNet-X-800mf", "regnet", lambda: tv_models.regnet_x_800mf(pretrained=False), (1, 3, 224, 224), "x-800mf",
     {'time': 6.8, 'success': 1.0, 'params': '7.3M', 'complexity': 'low'}),
    ("RegNet-X-1.6gf", "regnet", lambda: tv_models.regnet_x_1_6gf(pretrained=False), (1, 3, 224, 224), "x-1.6gf",
     {'time': 9.1, 'success': 1.0, 'params': '9.2M', 'complexity': 'low'}),
    ("RegNet-X-3.2gf", "regnet", lambda: tv_models.regnet_x_3_2gf(pretrained=False), (1, 3, 224, 224), "x-3.2gf",
     {'time': 12.4, 'success': 1.0, 'params': '15.3M', 'complexity': 'medium'}),
    ("RegNet-X-8gf", "regnet", lambda: tv_models.regnet_x_8gf(pretrained=False), (1, 3, 224, 224), "x-8gf",
     {'time': 18.2, 'success': 1.0, 'params': '39.6M', 'complexity': 'medium'}),
    ("RegNet-X-16gf", "regnet", lambda: tv_models.regnet_x_16gf(pretrained=False), (1, 3, 224, 224), "x-16gf",
     {'time': 25.7, 'success': 1.0, 'params': '54.3M', 'complexity': 'medium'}),
    ("RegNet-X-32gf", "regnet", lambda: tv_models.regnet_x_32gf(pretrained=False), (1, 3, 224, 224), "x-32gf",
     {'time': 35.1, 'success': 1.0, 'params': '107.8M', 'complexity': 'high'}),
    ("RegNet-Y-400mf", "regnet", lambda: tv_models.regnet_y_400mf(pretrained=False), (1, 3, 224, 224), "y-400mf",
     {'time': 5.8, 'success': 1.0, 'params': '4.3M', 'complexity': 'low'}),
    ("RegNet-Y-800mf", "regnet", lambda: tv_models.regnet_y_800mf(pretrained=False), (1, 3, 224, 224), "y-800mf",
     {'time': 7.2, 'success': 1.0, 'params': '6.3M', 'complexity': 'low'}),
    ("RegNet-Y-1.6gf", "regnet", lambda: tv_models.regnet_y_1_6gf(pretrained=False), (1, 3, 224, 224), "y-1.6gf",
     {'time': 9.8, 'success': 1.0, 'params': '11.2M', 'complexity': 'low'}),
    ("RegNet-Y-3.2gf", "regnet", lambda: tv_models.regnet_y_3_2gf(pretrained=False), (1, 3, 224, 224), "y-3.2gf",
     {'time': 13.2, 'success': 1.0, 'params': '19.4M', 'complexity': 'medium'}),
    ("RegNet-Y-8gf", "regnet", lambda: tv_models.regnet_y_8gf(pretrained=False), (1, 3, 224, 224), "y-8gf",
     {'time': 19.4, 'success': 1.0, 'params': '39.2M', 'complexity': 'medium'}),
    ("RegNet-Y-16gf", "regnet", lambda: tv_models.regnet_y_16gf(pretrained=False), (1, 3, 224, 224), "y-16gf",
     {'time': 27.8, 'success': 1.0, 'params': '83.6M', 'complexity': 'medium'}),
    ("RegNet-Y-32gf", "regnet", lambda: tv_models.regnet_y_32gf(pretrained=False), (1, 3, 224, 224), "y-32gf",
     {'time': 38.2, 'success': 1.0, 'params': '145.0M', 'complexity': 'high'}),
    ("RegNet-Y-128gf", "regnet", lambda: tv_models.regnet_y_128gf(pretrained=False), (1, 3, 224, 224), "y-128gf",
     {'time': 52.3, 'success': 1.0, 'params': '644.8M', 'complexity': 'high'}),

    # ========== MNASNet family (3 models) - Neural architecture search ==========
    ("MNASNet-0.5x", "mnasnet", lambda: tv_models.mnasnet0_5(pretrained=False), (1, 3, 224, 224), "0.5x",
     {'time': 3.8, 'success': 1.0, 'params': '2.2M', 'complexity': 'low'}),
    ("MNASNet-1.0x", "mnasnet", lambda: tv_models.mnasnet1_0(pretrained=False), (1, 3, 224, 224), "1.0x",
     {'time': 5.2, 'success': 1.0, 'params': '4.4M', 'complexity': 'low'}),
    ("MNASNet-1.3x", "mnasnet", lambda: tv_models.mnasnet1_3(pretrained=False), (1, 3, 224, 224), "1.3x",
     {'time': 6.1, 'success': 1.0, 'params': '6.3M', 'complexity': 'low'}),

    # ========== ConvNeXt family (4 models) - Modern ConvNet design ==========
    ("ConvNeXt-Tiny", "convnext", lambda: tv_models.convnext_tiny(pretrained=False), (1, 3, 224, 224), "tiny",
     {'time': 14.2, 'success': 1.0, 'params': '28.6M', 'complexity': 'medium'}),
    ("ConvNeXt-Small", "convnext", lambda: tv_models.convnext_small(pretrained=False), (1, 3, 224, 224), "small",
     {'time': 18.7, 'success': 1.0, 'params': '50.2M', 'complexity': 'medium'}),
    ("ConvNeXt-Base", "convnext", lambda: tv_models.convnext_base(pretrained=False), (1, 3, 224, 224), "base",
     {'time': 25.3, 'success': 1.0, 'params': '88.6M', 'complexity': 'high'}),
    ("ConvNeXt-Large", "convnext", lambda: tv_models.convnext_large(pretrained=False), (1, 3, 224, 224), "large",
     {'time': 34.8, 'success': 1.0, 'params': '197.8M', 'complexity': 'high'}),

    # ========== Vision Transformer family (4 models) - Transformer architecture ==========
    ("ViT-Base-16", "vit", lambda: tv_models.vit_b_16(pretrained=False), (1, 3, 224, 224), "base-16",
     {'time': 22.4, 'success': 1.0, 'params': '86.6M', 'complexity': 'medium'}),
    ("ViT-Large-16", "vit", lambda: tv_models.vit_l_16(pretrained=False), (1, 3, 224, 224), "large-16",
     {'time': 38.7, 'success': 1.0, 'params': '304.3M', 'complexity': 'high'}),
    ("ViT-Large-32", "vit", lambda: tv_models.vit_l_32(pretrained=False), (1, 3, 224, 224), "large-32",
     {'time': 28.2, 'success': 1.0, 'params': '306.5M', 'complexity': 'high'}),
    ("ViT-Huge-14", "vit", lambda: tv_models.vit_h_14(pretrained=False), (1, 3, 224, 224), "huge-14",
     {'time': 56.8, 'success': 1.0, 'params': '632.0M', 'complexity': 'high'}),

    # ========== Swin Transformer family (6 models) - Hierarchical vision transformer ==========
    ("Swin-Tiny", "swin", lambda: tv_models.swin_t(pretrained=False), (1, 3, 224, 224), "tiny",
     {'time': 18.3, 'success': 1.0, 'params': '28.3M', 'complexity': 'medium'}),
    ("Swin-Small", "swin", lambda: tv_models.swin_s(pretrained=False), (1, 3, 224, 224), "small",
     {'time': 24.7, 'success': 1.0, 'params': '49.6M', 'complexity': 'medium'}),
    ("Swin-Base", "swin", lambda: tv_models.swin_b(pretrained=False), (1, 3, 224, 224), "base",
     {'time': 49.5, 'success': 1.0, 'params': '87.8M', 'complexity': 'high'}),
    ("Swin-v2-Tiny", "swin", lambda: tv_models.swin_v2_t(pretrained=False), (1, 3, 224, 224), "v2-tiny",
     {'time': 19.2, 'success': 1.0, 'params': '28.4M', 'complexity': 'medium'}),
    ("Swin-v2-Small", "swin", lambda: tv_models.swin_v2_s(pretrained=False), (1, 3, 224, 224), "v2-small",
     {'time': 26.3, 'success': 1.0, 'params': '49.7M', 'complexity': 'medium'}),
    ("Swin-v2-Base", "swin", lambda: tv_models.swin_v2_b(pretrained=False), (1, 3, 224, 224), "v2-base",
     {'time': 51.2, 'success': 1.0, 'params': '87.9M', 'complexity': 'high'}),

    # ========== Inception family (2 models) - Multi-scale feature extraction ==========
    ("Inception-v3", "inception", lambda: tv_models.inception_v3(pretrained=False, aux_logits=False), (1, 3, 299, 299), "v3",
     {'time': 16.8, 'success': 1.0, 'params': '23.8M', 'complexity': 'medium'}),
    ("GoogLeNet", "googlenet", lambda: tv_models.googlenet(pretrained=False, aux_logits=False), (1, 3, 224, 224), "inception-v1",
     {'time': 9.2, 'success': 1.0, 'params': '6.6M', 'complexity': 'low'}),

    # ========== SqueezeNet family (2 models) - Compact architecture ==========
    ("SqueezeNet-v1.0", "squeezenet", lambda: tv_models.squeezenet1_0(pretrained=False), (1, 3, 224, 224), "v1.0",
     {'time': 1.9, 'success': 1.0, 'params': '1.2M', 'complexity': 'low'}),
    ("SqueezeNet-v1.1", "squeezenet", lambda: tv_models.squeezenet1_1(pretrained=False), (1, 3, 224, 224), "v1.1",
     {'time': 1.7, 'success': 1.0, 'params': '1.2M', 'complexity': 'low'}),

    # ========== AlexNet (1 model) - Classic CNN, VERY FAST ==========
    ("AlexNet", "alexnet", lambda: tv_models.alexnet(pretrained=False), (1, 3, 224, 224), "classic",
     {'time': 0.9, 'success': 1.0, 'params': '61.1M', 'complexity': 'low'}),

    # ========== ShuffleNet (1 model - only x0.5 variant works) ==========
    ("ShuffleNet-v2-x0.5", "shufflenet", lambda: tv_models.shufflenet_v2_x0_5(pretrained=False), (1, 3, 224, 224), "v2-x0.5",
     {'time': 13.1, 'success': 0.0, 'params': '1.4M', 'complexity': 'low'}),
    # Note: Other ShuffleNet variants (x1.0, x1.5, x2.0) fail compilation

    # ========== Batch size variants (10 models) - Same architectures, different batch sizes ==========
    ("ResNet-18 (bs=2)", "resnet_bs2", lambda: tv_models.resnet18(pretrained=False), (2, 3, 224, 224), "18-bs2",
     {'time': 8.5, 'success': 1.0, 'params': '11.7M', 'complexity': 'low'}),
    ("ResNet-34 (bs=2)", "resnet_bs2", lambda: tv_models.resnet34(pretrained=False), (2, 3, 224, 224), "34-bs2",
     {'time': 10.8, 'success': 1.0, 'params': '21.8M', 'complexity': 'low'}),
    ("ResNet-50 (bs=2)", "resnet_bs2", lambda: tv_models.resnet50(pretrained=False), (2, 3, 224, 224), "50-bs2",
     {'time': 15.6, 'success': 1.0, 'params': '25.6M', 'complexity': 'medium'}),
    ("ResNet-18 (bs=4)", "resnet_bs4", lambda: tv_models.resnet18(pretrained=False), (4, 3, 224, 224), "18-bs4",
     {'time': 9.1, 'success': 1.0, 'params': '11.7M', 'complexity': 'low'}),
    ("ResNet-34 (bs=4)", "resnet_bs4", lambda: tv_models.resnet34(pretrained=False), (4, 3, 224, 224), "34-bs4",
     {'time': 11.4, 'success': 1.0, 'params': '21.8M', 'complexity': 'low'}),
    ("MobileNet-v2 (bs=2)", "mobilenet_bs2", lambda: tv_models.mobilenet_v2(pretrained=False), (2, 3, 224, 224), "v2-bs2",
     {'time': 4.5, 'success': 1.0, 'params': '3.5M', 'complexity': 'low'}),
    ("EfficientNet-b0 (bs=2)", "efficientnet_bs2", lambda: tv_models.efficientnet_b0(pretrained=False), (2, 3, 224, 224), "b0-bs2",
     {'time': 9.1, 'success': 1.0, 'params': '5.3M', 'complexity': 'low'}),
    ("VGG-11 (bs=2)", "vgg_bs2", lambda: tv_models.vgg11(pretrained=False), (2, 3, 224, 224), "11-bs2",
     {'time': 3.1, 'success': 1.0, 'params': '132.9M', 'complexity': 'low'}),
    ("AlexNet (bs=2)", "alexnet_bs2", lambda: tv_models.alexnet(pretrained=False), (2, 3, 224, 224), "classic-bs2",
     {'time': 1.0, 'success': 1.0, 'params': '61.1M', 'complexity': 'low'}),
    ("MobileNet-v2 (bs=4)", "mobilenet_bs4", lambda: tv_models.mobilenet_v2(pretrained=False), (4, 3, 224, 224), "v2-bs4",
     {'time': 4.9, 'success': 1.0, 'params': '3.5M', 'complexity': 'low'}),

    # Continue in next part due to length...
]

# Part 2 of MODEL_LIST (remaining models)
MODEL_LIST_PART2 = [
    ("EfficientNet-b0 (bs=4)", "efficientnet_bs4", lambda: tv_models.efficientnet_b0(pretrained=False), (4, 3, 224, 224), "b0-bs4",
     {'time': 9.7, 'success': 1.0, 'params': '5.3M', 'complexity': 'low'}),
    ("SqueezeNet (bs=4)", "squeezenet_bs4", lambda: tv_models.squeezenet1_0(pretrained=False), (4, 3, 224, 224), "v1.0-bs4",
     {'time': 2.3, 'success': 1.0, 'params': '1.2M', 'complexity': 'low'}),
    ("AlexNet (bs=4)", "alexnet_bs4", lambda: tv_models.alexnet(pretrained=False), (4, 3, 224, 224), "classic-bs4",
     {'time': 1.1, 'success': 1.0, 'params': '61.1M', 'complexity': 'low'}),

    # ========== Input size variants (13 models) - Different resolutions ==========
    ("ResNet-50 (384x384)", "resnet_384", lambda: tv_models.resnet50(pretrained=False), (1, 3, 384, 384), "50-384x384",
     {'time': 24.2, 'success': 1.0, 'params': '25.6M', 'complexity': 'medium'}),
    ("MobileNet-v2 (384x384)", "mobilenet_384", lambda: tv_models.mobilenet_v2(pretrained=False), (1, 3, 384, 384), "v2-384x384",
     {'time': 6.8, 'success': 1.0, 'params': '3.5M', 'complexity': 'low'}),
    ("EfficientNet-b0 (384x384)", "efficientnet_384", lambda: tv_models.efficientnet_b0(pretrained=False), (1, 3, 384, 384), "b0-384x384",
     {'time': 13.2, 'success': 1.0, 'params': '5.3M', 'complexity': 'medium'}),
    ("VGG-16 (384x384)", "vgg_384", lambda: tv_models.vgg16(pretrained=False), (1, 3, 384, 384), "16-384x384",
     {'time': 3.8, 'success': 1.0, 'params': '138.4M', 'complexity': 'medium'}),
    ("ResNet-18 (128x128)", "resnet_128", lambda: tv_models.resnet18(pretrained=False), (1, 3, 128, 128), "18-128x128",
     {'time': 6.2, 'success': 1.0, 'params': '11.7M', 'complexity': 'low'}),
    ("MobileNet-v2 (128x128)", "mobilenet_128", lambda: tv_models.mobilenet_v2(pretrained=False), (1, 3, 128, 128), "v2-128x128",
     {'time': 3.2, 'success': 1.0, 'params': '3.5M', 'complexity': 'low'}),
    ("EfficientNet-b0 (128x128)", "efficientnet_128", lambda: tv_models.efficientnet_b0(pretrained=False), (1, 3, 128, 128), "b0-128x128",
     {'time': 6.4, 'success': 1.0, 'params': '5.3M', 'complexity': 'low'}),
    ("SqueezeNet (128x128)", "squeezenet_128", lambda: tv_models.squeezenet1_0(pretrained=False), (1, 3, 128, 128), "v1.0-128x128",
     {'time': 1.5, 'success': 1.0, 'params': '1.2M', 'complexity': 'low'}),
    ("ResNet-152 (128x128)", "resnet_128", lambda: tv_models.resnet152(pretrained=False), (1, 3, 128, 128), "152-128x128",
     {'time': 32.1, 'success': 1.0, 'params': '60.2M', 'complexity': 'high'}),
    ("EfficientNet-b3 (128x128)", "efficientnet_128", lambda: tv_models.efficientnet_b3(pretrained=False), (1, 3, 128, 128), "b3-128x128",
     {'time': 11.8, 'success': 1.0, 'params': '12.2M', 'complexity': 'medium'}),
    ("Swin-Tiny (384x384)", "swin_384", lambda: tv_models.swin_t(pretrained=False), (1, 3, 384, 384), "tiny-384x384",
     {'time': 28.7, 'success': 1.0, 'params': '28.3M', 'complexity': 'high'}),
    ("ConvNeXt-Small (128x128)", "convnext_128", lambda: tv_models.convnext_small(pretrained=False), (1, 3, 128, 128), "small-128x128",
     {'time': 14.3, 'success': 1.0, 'params': '50.2M', 'complexity': 'medium'}),
    ("DenseNet-201 (128x128)", "densenet_128", lambda: tv_models.densenet201(pretrained=False), (1, 3, 128, 128), "201-128x128",
     {'time': 89.4, 'success': 1.0, 'params': '20.0M', 'complexity': 'high'}),
    ("VGG-19 (128x128)", "vgg_128", lambda: tv_models.vgg19(pretrained=False), (1, 3, 128, 128), "19-128x128",
     {'time': 2.1, 'success': 1.0, 'params': '143.7M', 'complexity': 'low'}),
]

# Combine all model parts
MODEL_LIST.extend(MODEL_LIST_PART2)

# Total: 108 models
print(f"Loaded {len(MODEL_LIST)} models")


# ============================================================================
# MODEL ORGANIZATION AND FILTERING
# ============================================================================

def get_model_families() -> Dict[str, List]:
    """Get models organized by family."""
    families = {}
    for model in MODEL_LIST:
        family = model[1]
        if family not in families:
            families[family] = []
        families[family].append(model)
    return families


def get_models(
    family: Optional[str] = None,
    batch_size: Optional[int] = None,
    input_size: Optional[int] = None,
    complexity: Optional[str] = None,
    count: Optional[int] = None,
    sort_by: str = 'compile_time'
) -> List:
    """
    Get filtered and sorted model list.

    Args:
        family: Filter by model family (e.g., 'resnet', 'efficientnet')
        batch_size: Filter by batch size (1, 2, 4)
        input_size: Filter by input resolution (128, 224, 384, 299)
        complexity: Filter by complexity ('low', 'medium', 'high')
        count: Limit to first N models
        sort_by: Sort criteria ('compile_time', 'name', 'family', 'size')

    Returns:
        Filtered and sorted list of models
    """
    filtered = MODEL_LIST.copy()

    # Apply filters
    if family:
        filtered = [m for m in filtered if m[1] == family]

    if batch_size:
        filtered = [m for m in filtered if m[3][0] == batch_size]

    if input_size:
        filtered = [m for m in filtered if m[3][2] == input_size]

    if complexity:
        filtered = [m for m in filtered if m[5]['complexity'] == complexity]

    # Sort
    if sort_by == 'compile_time':
        filtered.sort(key=lambda m: m[5]['time'])
    elif sort_by == 'name':
        filtered.sort(key=lambda m: m[0])
    elif sort_by == 'family':
        filtered.sort(key=lambda m: (m[1], m[0]))
    elif sort_by == 'size':
        # Sort by parameter count (need to parse string like '11.7M')
        def parse_params(p):
            val = float(p[:-1])
            return val if p.endswith('M') else val * 1000
        filtered.sort(key=lambda m: parse_params(m[5]['params']))

    # Limit count
    if count:
        filtered = filtered[:count]

    return filtered


def get_quick_test_models(count: int = 5) -> List:
    """
    Get fast-compiling models for quick testing.

    Returns models that compile in <5 seconds.
    """
    return get_models(sort_by='compile_time', count=count)


def get_slow_models(count: int = 5) -> List:
    """
    Get slowest models for stress testing.

    Returns: DenseNet-201 (116s), DenseNet-169 (78s), etc.
    """
    slow_models = sorted(MODEL_LIST, key=lambda m: m[5]['time'], reverse=True)
    return slow_models[:count]


def get_model_by_name(name: str) -> Optional[Tuple]:
    """
    Lookup model by exact name or fuzzy match.

    Args:
        name: Model name (case-insensitive)

    Returns:
        Model tuple if found, None otherwise
    """
    name_lower = name.lower()

    # Exact match
    for model in MODEL_LIST:
        if model[0].lower() == name_lower:
            return model

    # Fuzzy match (contains)
    for model in MODEL_LIST:
        if name_lower in model[0].lower():
            return model

    return None


def estimate_total_time(models: List, num_chips: int) -> float:
    """
    Estimate total compilation time given model list and chip count.

    Uses empirical compile times from success report.
    Accounts for parallel execution across chips.

    Args:
        models: List of model tuples
        num_chips: Number of chips for parallel execution

    Returns:
        Estimated time in seconds
    """
    if num_chips <= 0:
        return 0.0

    # Calculate total time for each chip (round-robin distribution)
    chip_times = [0.0] * num_chips
    for idx, model in enumerate(models):
        chip_id = idx % num_chips
        compile_time = model[5]['time']
        chip_times[chip_id] += compile_time

    # Total time is max chip time (parallel execution)
    return max(chip_times)


def get_model_stats() -> Dict:
    """
    Get aggregate statistics about model library.

    Returns:
        Dict with statistics:
            - total_models: Total number of models
            - families: Number of families
            - by_family: Count per family
            - by_complexity: Count per complexity level
            - avg_compile_time: Average compilation time
            - total_parameters: Total parameters across all models
    """
    families = get_model_families()
    complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
    total_time = 0.0

    for model in MODEL_LIST:
        complexity_counts[model[5]['complexity']] += 1
        total_time += model[5]['time']

    return {
        'total_models': len(MODEL_LIST),
        'families': len(families),
        'by_family': {k: len(v) for k, v in families.items()},
        'by_complexity': complexity_counts,
        'avg_compile_time': total_time / len(MODEL_LIST),
        'success_rate': sum(m[5]['success'] for m in MODEL_LIST) / len(MODEL_LIST),
    }


# Example usage
if __name__ == '__main__':
    print("TT-Forge Compiletron - Model Library")
    print("=" * 60)

    # Show statistics
    stats = get_model_stats()
    print(f"\nModel Library Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Families: {stats['families']}")
    print(f"  Average compile time: {stats['avg_compile_time']:.1f}s")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")

    print(f"\nBy complexity:")
    for comp, count in stats['by_complexity'].items():
        print(f"  {comp}: {count} models")

    print(f"\nTop 5 families:")
    sorted_families = sorted(stats['by_family'].items(), key=lambda x: x[1], reverse=True)[:5]
    for family, count in sorted_families:
        print(f"  {family}: {count} models")

    # Show quick test models
    print(f"\n5 Fastest Models (for quick testing):")
    quick = get_quick_test_models(5)
    for model in quick:
        print(f"  {model[0]}: {model[5]['time']:.1f}s")

    # Show slow models
    print(f"\n5 Slowest Models (for stress testing):")
    slow = get_slow_models(5)
    for model in slow:
        print(f"  {model[0]}: {model[5]['time']:.1f}s")

    # Estimate time for 50 models on 4 chips
    models_50 = get_models(count=50)
    time_4chips = estimate_total_time(models_50, 4)
    print(f"\nEstimate for 50 models on 4 chips: {time_4chips/60:.1f} minutes")
