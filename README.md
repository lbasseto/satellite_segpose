# satellite_segpose

Use train_bn_pruning.py to train a network to be pruned. darknet_bn_pruning.py will be used before the actual pruning and then the model will become a regular darknet.py object.

All files with the suffix "_bn_pruning" act on the model before pruning, after compression the corresponding files without the suffix are necessary.

Before pruning, use train_bn_pruning.py and test_withzoom_bn_pruning.py. After, use train.py and test_withzoom.py (for finetuning for example).
