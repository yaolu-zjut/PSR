# PSR
## Step-by-step Instructions
**1. Train a signal model and save it to ./pretrained_signal/result/, taking the RML2016.10a dataset and ResNet56 model as an example:**
```python
python train.py --dataset all_radio128 --model ResNet56_signal --num_epochs 50 --batch_size 128
```

**2. Calculate the CKA matrix and save it to ./PSR. The trained model weights need to be reloaded in ./utils1/get_model.py:**
```python
python similarity.py --gpu 4 --arch ResNet56_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate 
```

**3. Use Fisher algorithm for segmentation. First set the required number of blocks K=4, and then run the following Python code to get the divided blocks:**
```python
python Network_Partition.py --arch ResNet56_signal --set all_radio128
```

**4. Use synflow for fast performance evaluation without a dataset. First, you need to set the partition_ResNet56_signal_all_radio128 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4], then define the pruning rate remain_layer = [1, 1, 1, 1], retain one layer for each block, then reload the original model weights in replace_layer_initialization, and finally run the evaluation code to get the layers that need to be retained. Finally, we get 
Remaining layers: (10,)
Remaining layers: (16,)
Remaining layers: (21,)
Remaining layers: (27,):**
```python
python Reassembly.py --gpu 4 --arch ResNet56_signal --set all_radio128 --num_classes 11 --batch_size 128 --pretrained --evaluate  --zero_proxy synflow
```

**5. After obtaining the layers that need to be retained, the original model structure in ./PSR/model_signal/resnet.py and the model weights in ./PSR/utils1/get_model.py need to be modified, and the corresponding layers need to be retained. In resnet.py, the modified model structure is used to adapt the previous layers with the later layers if the channels do not match. The final definition is ResNet56_signal_KD. In get_model.py, load the original model weights, set which layers need to be retained, and load the weights of these layers:**

**6. Finally, fine-tune the pruned model and save the trained pruned model weights:**
```python
python finetune.py --gpu 1 --arch ResNet56_signal_KD --set all_radio128 --batch_size 128 --weight_decay 0.005 --epochs 50 --lr 0.001 --finetune
```

**7. Finally, reload the trained model weights in ./PSR/calculating_flops.py and run the code to calculate the pruned model params, flops, and acc:**
```python
python calculating_flops.py --gpu 3 --arch ResNet56_signal_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate
```

## Acknowledgement

[//]: # (- The evaluation of the LLM: [lm-evaluation-harness]&#40;https://github.com/EleutherAI/lm-evaluation-harness&#41;)

[//]: # (- Code Framework: https://github.com/horseee/LLM-Pruner )
