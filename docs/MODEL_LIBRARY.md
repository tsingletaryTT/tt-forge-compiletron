# Model Library Catalog

Complete catalog of 101 proven models with success rates and compilation times.

## Quick Stats

- **Total Models**: 101
- **Families**: 15+ architecture families
- **Success Rate**: 99.0% overall
- **Average Compile Time**: 19.2 seconds
- **Range**: 0.9s (AlexNet) to 116.2s (DenseNet-201)

## Model Families

### Vision - Convolutional Networks

#### ResNet Family (5 models)
Residual learning framework - highly successful, medium speed.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| ResNet-18 | 11.7M | 8.2s | Low | Fast, good baseline |
| ResNet-34 | 21.8M | 10.5s | Low | Deeper variant |
| ResNet-50 | 25.6M | 15.2s | Medium | Most popular |
| ResNet-101 | 44.5M | 28.4s | Medium | High accuracy |
| ResNet-152 | 60.2M | 41.2s | High | Very deep |

**Use cases**: Image classification, feature extraction, transfer learning

#### VGG Family (8 models)
Simple architecture - VERY FAST compilation, large memory.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| VGG-11 | 132.9M | 2.8s | Low | Fastest VGG |
| VGG-11-BN | 132.9M | 3.1s | Low | With batch norm |
| VGG-13 | 133.0M | 2.1s | Low | Very fast |
| VGG-13-BN | 133.0M | 2.4s | Low | With batch norm |
| VGG-16 | 138.4M | 2.4s | Low | Classic choice |
| VGG-16-BN | 138.4M | 2.7s | Low | With batch norm |
| VGG-19 | 143.7M | 2.7s | Low | Deepest VGG |
| VGG-19-BN | 143.7M | 3.0s | Low | With batch norm |

**Use cases**: Quick testing, baseline comparisons, feature extraction

#### EfficientNet Family (8 models)
State-of-the-art efficiency - compound scaling.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| EfficientNet-b0 | 5.3M | 8.5s | Low | Most efficient |
| EfficientNet-b1 | 7.8M | 11.2s | Low | Balanced |
| EfficientNet-b2 | 9.2M | 12.8s | Medium | Good accuracy |
| EfficientNet-b3 | 12.2M | 15.4s | Medium | Popular choice |
| EfficientNet-b4 | 19.3M | 22.1s | Medium | High quality |
| EfficientNet-b5 | 30.4M | 31.5s | Medium | Very accurate |
| EfficientNet-b6 | 43.0M | 47.2s | High | Slow compile |
| EfficientNet-b7 | 66.3M | 45.8s | High | Largest |

**Use cases**: Mobile deployment, edge devices, production inference

#### DenseNet Family (4 models)
Dense connections - SLOW compilation but high accuracy.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| DenseNet-121 | 8.0M | 42.3s | High | Slowest |
| DenseNet-161 | 28.7M | 71.1s | High | Very slow |
| DenseNet-169 | 14.1M | 78.3s | High | Very slow |
| DenseNet-201 | 20.0M | 116.2s | High | SLOWEST (2 min) |

**Use cases**: Stress testing, maximum accuracy, research

#### RegNet Family (15 models)
Design space exploration - scalable, efficient.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| RegNet-X-400mf | 5.2M | 5.2s | Low |
| RegNet-X-800mf | 7.3M | 6.8s | Low |
| RegNet-X-1.6gf | 9.2M | 9.1s | Low |
| RegNet-X-3.2gf | 15.3M | 12.4s | Medium |
| RegNet-X-8gf | 39.6M | 18.2s | Medium |
| RegNet-X-16gf | 54.3M | 25.7s | Medium |
| RegNet-X-32gf | 107.8M | 35.1s | High |
| RegNet-Y-400mf | 4.3M | 5.8s | Low |
| RegNet-Y-800mf | 6.3M | 7.2s | Low |
| RegNet-Y-1.6gf | 11.2M | 9.8s | Low |
| RegNet-Y-3.2gf | 19.4M | 13.2s | Medium |
| RegNet-Y-8gf | 39.2M | 19.4s | Medium |
| RegNet-Y-16gf | 83.6M | 27.8s | Medium |
| RegNet-Y-32gf | 145.0M | 38.2s | High |
| RegNet-Y-128gf | 644.8M | 52.3s | High |

**Use cases**: Architecture search, scalable deployment

### Vision - Transformers

#### Vision Transformer (4 models)
Pure transformer - attention-based vision.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| ViT-Base-16 | 86.6M | 22.4s | Medium | Standard ViT |
| ViT-Large-16 | 304.3M | 38.7s | High | Large model |
| ViT-Large-32 | 306.5M | 28.2s | High | Larger patches |
| ViT-Huge-14 | 632.0M | 56.8s | High | Massive model |

**Use cases**: Large-scale classification, foundation models

#### Swin Transformer (6 models)
Hierarchical vision transformer - shifted windows.

| Model | Params | Time | Complexity | Notes |
|-------|--------|------|------------|-------|
| Swin-Tiny | 28.3M | 18.3s | Medium | Efficient |
| Swin-Small | 49.6M | 24.7s | Medium | Balanced |
| Swin-Base | 87.8M | 49.5s | High | Slow compile |
| Swin-v2-Tiny | 28.4M | 19.2s | Medium | Improved |
| Swin-v2-Small | 49.7M | 26.3s | Medium | Improved |
| Swin-v2-Base | 87.9M | 51.2s | High | Improved |

**Use cases**: Object detection, segmentation, downstream tasks

### Mobile & Efficient

#### MobileNet Family (3 models)
Mobile-optimized - inverted residuals.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| MobileNet-v2 | 3.5M | 4.2s | Low |
| MobileNet-v3-Small | 2.5M | 2.6s | Low |
| MobileNet-v3-Large | 5.5M | 4.8s | Low |

**Use cases**: Mobile devices, embedded systems, edge AI

#### MNASNet Family (3 models)
Neural architecture search - automated design.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| MNASNet-0.5x | 2.2M | 3.8s | Low |
| MNASNet-1.0x | 4.4M | 5.2s | Low |
| MNASNet-1.3x | 6.3M | 6.1s | Low |

**Use cases**: Resource-constrained environments

### Modern Architectures

#### ConvNeXt Family (4 models)
Modernized ConvNet - competitive with transformers.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| ConvNeXt-Tiny | 28.6M | 14.2s | Medium |
| ConvNeXt-Small | 50.2M | 18.7s | Medium |
| ConvNeXt-Base | 88.6M | 25.3s | High |
| ConvNeXt-Large | 197.8M | 34.8s | High |

**Use cases**: Research, comparison with transformers

### Classic Models

#### AlexNet (1 model)
FASTEST compilation - classic CNN breakthrough.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| AlexNet | 61.1M | 0.9s | Low |

**Use cases**: Quick testing, benchmarking, teaching

#### SqueezeNet (2 models)
Ultra-compact - fire modules.

| Model | Params | Time | Complexity |
|-------|--------|------|------------|
| SqueezeNet-v1.0 | 1.2M | 1.9s | Low |
| SqueezeNet-v1.1 | 1.2M | 1.7s | Low |

**Use cases**: Extreme compression, embedded systems

## Usage Patterns

### Quick Test Set (< 5s)

Perfect for validating environment and testing changes:
```bash
python3 compiletron.py models quick --count 5
```

Models:
1. AlexNet (0.9s)
2. AlexNet bs=2 (1.0s)
3. AlexNet bs=4 (1.1s)
4. SqueezeNet 128x128 (1.5s)
5. SqueezeNet-v1.1 (1.7s)

### Balanced Set (5-20s)

Good mix of speed and diversity:
- ResNet-18, ResNet-34
- MobileNet variants
- EfficientNet-b0 through b3
- VGG family
- RegNet-X/Y small variants

### Stress Test Set (> 50s)

Test compilation limits:
1. DenseNet-201 (116.2s)
2. DenseNet-169 (78.3s)
3. DenseNet-161 (71.1s)
4. ViT-Huge-14 (56.8s)
5. RegNet-Y-128gf (52.3s)

### Production Set

Commonly deployed models:
- ResNet-50
- EfficientNet-b0 to b4
- MobileNet-v3
- Swin-Tiny
- ViT-Base-16

## Compilation Success Rates

### 100% Success (99 models)
All models except ShuffleNet variants compile successfully.

### 0% Success (1 model)
- ShuffleNet-v2-x0.5 (known issue)

Note: Other ShuffleNet variants (x1.0, x1.5, x2.0) also fail but not included in library.

## Filtering and Selection

### By Family
```bash
python3 compiletron.py models list --family resnet
python3 compiletron.py models list --family efficientnet
```

### By Complexity
```bash
python3 compiletron.py models list --complexity low
python3 compiletron.py models list --complexity medium
python3 compiletron.py models list --complexity high
```

### By Batch Size
```bash
python3 compiletron.py models list --batch-size 1
python3 compiletron.py models list --batch-size 2
python3 compiletron.py models list --batch-size 4
```

### By Input Resolution
```bash
python3 compiletron.py models list --input-size 128
python3 compiletron.py models list --input-size 224
python3 compiletron.py models list --input-size 384
```

## Batch Size Variants

Same architectures with different batch sizes for throughput testing:

**Batch Size 2**:
- ResNet-18, ResNet-34, ResNet-50
- MobileNet-v2
- EfficientNet-b0
- VGG-11
- AlexNet

**Batch Size 4**:
- ResNet-18, ResNet-34
- MobileNet-v2
- EfficientNet-b0
- SqueezeNet
- AlexNet

## Resolution Variants

Same models at different input sizes:

**128×128** (smaller, faster):
- ResNet-18, ResNet-152
- MobileNet-v2
- EfficientNet-b0, EfficientNet-b3
- SqueezeNet
- ConvNeXt-Small
- DenseNet-201
- VGG-19

**384×384** (larger, slower):
- ResNet-50
- MobileNet-v2
- EfficientNet-b0
- VGG-16
- Swin-Tiny

## Time Estimates

### Single Chip
- 10 models: ~3 minutes
- 50 models: ~16 minutes
- 101 models: ~32 minutes

### 4 Chips (Parallel)
- 10 models: ~45 seconds
- 50 models: ~4 minutes
- 101 models: ~8 minutes

### 8 Chips (Parallel)
- 10 models: ~23 seconds
- 50 models: ~2 minutes
- 101 models: ~4 minutes

## Adding Custom Models

To add your own models, edit `lib/models.py`:

```python
MODEL_LIST.append((
    "MyModel",                              # Display name
    "custom",                               # Family
    lambda: my_model_loader(),              # Loader function
    (1, 3, 224, 224),                       # Input shape
    "custom-notes",                         # Notes
    {
        'time': 10.0,                       # Expected compile time
        'success': 1.0,                     # Success rate (0-1)
        'params': '25M',                    # Parameter count
        'complexity': 'medium'              # Complexity level
    }
))
```

## See Also

- [Forge Setup](FORGE_SETUP.md) - Installation
- [Multi-Chip](MULTI_CHIP.md) - Parallel execution
- [Main README](../README.md) - Usage guide
