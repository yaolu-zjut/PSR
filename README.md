# PSR
## Introduction

[//]: # (![image]&#40;https://github.com/yaolu-zjut/Navigation-LLM-layer-pruning/blob/main/framework.JPG&#41;)
[//]: # (Although large language models &#40;LLMs&#41; have achieved remarkable success across various domains, their considerable scale necessitates substantial computational resources, posing significant challenges for deployment in resource-constrained environments. Layer pruning, as a simple yet effective compression method, removes layers of a model directly, reducing computational overhead. However, what are the best practices for layer pruning in LLMs? Are sophisticated layer selection metrics truly effective? Does the LoRA &#40;Low-Rank Approximation&#41; family, widely regarded as a leading method for pruned model fine-tuning, truly meet expectations when applied to post-pruning fine-tuning? To answer these questions, we dedicate thousands of GPU hours to benchmarking layer pruning in LLMs and gaining insights across multiple dimensions. Our results demonstrate that a simple approach, i.e., pruning the final 25\% of layers followed by fine-tuning the \texttt{lm\_head} and the remaining last three layer, yields remarkably strong performance. Following this guide, we prune Llama-3.1-8B-It and obtain a model that outperforms many popular LLMs of similar size, such as ChatGLM2-6B, Vicuna-7B-v1.5, Qwen1.5-7B and Baichuan2-7B. We release the optimal model weights on Huggingface, and the code is available on GitHub.)

[//]: # (### Supported LLMs:)

[//]: # (- [Vicuna-7b-v1.5]&#40;https://huggingface.co/lmsys/vicuna-7b-v1.5&#41;)

[//]: # (- [Qwen1.5-7B]&#40;https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwim-qfT1IaJAxUNr1YBHU-wF8UQFnoECB4QAQ&url=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen1.5-7B&usg=AOvVaw2E2lUSV7wML81PPxhzIfqJ&opi=89978449&#41;)

[//]: # (- [Gemma2-2B-It]&#40;https://huggingface.co/google/gemma-2-2b-it&#41;)

[//]: # (- [Llama-3.1-8B-It]&#40;https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct&#41;)

[//]: # (### Our Pruned Models)

[//]: # (- [Llama-3.1-6.3B-It-Alpaca]&#40;https://huggingface.co/anonymousICLR/Llama-3.1-6.3B-It-Alpaca&#41; )

[//]: # (- [Llama-3.1-6.3B-It-Dolly]&#40;https://huggingface.co/anonymousICLR/Llama-3.1-6.3B-It-Dolly/&#41;)


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

**5. 得到需要保留的层之后，需要修改./PSR/model_signal/resnet.py中原始模型结构和./PSR/utils1/get_model.py的模型权重，需要保留其对应的层。在resnet.py中将修改后的模型结构，如果通道不匹配则用后面的层去适配前面的层，最终定义为ResNet56_signal_KD,在get_model.py中加载原始模型权重，并且设定需要保留哪些层，将这些层的权重进行加载。:**

**6. 最后微调剪枝后的模型，将训练好的剪枝后的模型权重进行保存。:**
```python
python finetune.py --gpu 1 --arch ResNet56_signal_KD --set all_radio128 --batch_size 128 --weight_decay 0.005 --epochs 50 --lr 0.001 --finetune
```

**7. 最后将训练好的模型权重在./PSR/calculating_flops.py中重新加载，运行代码计算剪枝后的模型params、flops以及acc。:**
```python
python calculating_flops.py --gpu 3 --arch ResNet56_signal_KD --set all_radio128 --input_signal_size 128  --pretrained --evaluate
```


### Zero-shot Evaluation

[//]: # (![image]&#40;https://github.com/yaolu-zjut/Navigation-LLM-layer-pruning/blob/main/sota.JPG&#41;)

## Acknowledgement

[//]: # (- The evaluation of the LLM: [lm-evaluation-harness]&#40;https://github.com/EleutherAI/lm-evaluation-harness&#41;)

[//]: # (- Code Framework: https://github.com/horseee/LLM-Pruner )
