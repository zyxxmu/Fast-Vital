# Fast-Vital
Speed up vital tracker using pruning or quantification method

## Prerequisites

- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies

## Usage

### Tracking

```bash
 python tracking/run_tracker.py -s DragonBaby -f
```

 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python tracking/run_tracker.py -s [seq name]```
   - ```python tracking/run_tracker.py -j [json path]```

### Pretraining

 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"

 - Pretraining on VOT-OTB

   - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/VOT/vot201x"

    ``` bash
     python pretrain/prepro_vot.py
     python pretrain/train_mdnet.py -d vot
    ```

 - Pretraining on ImageNet-VID

   - Download ImageNet-VID dataset into "datasets/ILSVRC"

    ``` bash
     python pretrain/prepro_imagenet.py
     python pretrain/train_mdnet.py -d imagenet
    ```

### Pretrain by BNN

```bash
 python pretrain/BNN_train_mdnet.py -d vot
```

### Pretrain by TWN

```bash
 python pretrain/TWN_train_mdnet.py -d vot
```

### Pretrain by Random Pruning

```bash
 python pretrain/RandomPrune_train_mdnet.py -d vot
```

