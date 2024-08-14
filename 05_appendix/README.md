# Appendix 

주로 다루지는 않았지만 model을 training 할 때 유용하게 쓰일 방법들에 대해서 다룹니다. 

## LoRA, QLoRA (fine-tuning) 
최근 LLM의 발전 방향은 parameter 개수와 model size를 늘려 성능을 향상하는 추세입니다. 이에 따라 training에 필요한 시간과 GPU 자원의 요구량도 이전과는 비교할 수 없이 커지고 있습니다. fine-tuning은 pretraining에 비해 적은 크기의 데이터셋을 사용하기 때문에, training에 비교적 짧은 시간이 소요됩니다. 하지만 학습시키는 parameter의 개수는 pretraining과 같습니다. 최근 트렌드에 따라, 모델의 크기가 커질수록 fine-tuning 하는 데 걸리는 시간도 길어집니다. 이러한 이유로 fine-tuning을 보다 효율적으로 수행하기 위해 만들어진 기법이 바로 LoRA(Low-Rank Adaptation)입니다.
LoRA는 [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685)라는 논문에서 처음 제시된 기법입니다.
<img src="/home/sslunder13/project/SSL-LLM-tutorial/image/lora.png" alt="Alt text" width="458" height="396">
LoRA는 PEFT(Parameter Efficient Fine Tuning) 기법 중 가장 유명하고 자주 쓰이는 방법입니다. 간단히 소개하자면, 모델의 모든 parameter를 학습시키는 대신, pretrain 된 모델의 parameter는 freeze 시키고 ‘r’만큼의 rank를 가지는 low-rank matrix A와 B만 학습시킨 후 둘을 더해 최종 모델을 완성하는 기법입니다. 이 방법을 통해 학습시켜야 할 parameter의 개수는 획기적으로 줄이는 동시에, 동등하거나 심지어 향상된 성능을 보일 수 있습니다. 자세한 이론적인 내용은 논문에서 찾을 수 있으니, lora를 사용한 모델과 그렇지 않은 모델 간의 traning 시간을 비교하는 간단한 실험을 노트북을 활용해 첨부합니다.

다만, 첨부한 노트북에서는 training에 걸리는 시간을 비교할 수는 있으나 메모리 사용량을 비교하려면 kernel 재시작이 필요해 바로 비교하는 것이 불가능합니다. 빠른 이해를 위해 kernel 재시작 후 실험한 결과를 이미지로 첨부합니다.

<img src="/home/sslunder13/project/SSL-LLM-tutorial/image/lora_comparison.png" alt="Alt text" width="528" height="444">

하지만 저희는 LoRA를 적극적으로 활용하지 않았습니다. LoRA 사용을 최적화하기 위해서는 새로운 최적의 하이퍼 파라미터 조합을 실험을 통해 찾아야 하는데, GPT2의 경우 모델의 size가 작아 LoRA로 얻을 수 있는 이득보다 최적의 파라미터를 찾는 데 걸리는 시간이 더 길다고 판단했기 때문입니다.

- [LoRA tutorial](https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48)



## DataParallel
DataParallel은 PyTorch에서 여러 개의 GPU를 병렬적으로 사용해 모델을 train 할 수 있게 해주는 기능입니다. 여러 개의 GPU를 사용하도록 설정해도, DataParallel을 사용하지 않으면 training 시간은 하나의 GPU를 사용할 때와 같습니다. DataParallel은 아래와 같이 사용합니다. 

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)
```

이렇게 모델이 CUDA로 설정해 준 GPU에서 병렬적으로 실행되게 만들어주면 사용되는 메모리와 학습 시간을 줄일 수 있습니다.


## Finding datasets

Training 할 때 가장 중요하고 오랜 시간이 걸리는 것은 적절한 dataset을 찾는 것입니다. dataset의 quality와 size는 모델의 성능에 큰 영향을 미칩니다. 하지만 예산과 시간이 제한된 상황에서 직접 양질의 dataset을 만들기는 매우 어렵습니다. 그렇기에 dataset의 존재 유무, 그리고 public open 되어있는지 유무가 굉장히 중요합니다. 만약 어떤 방향으로 fine-tuning을 진행하고 싶다면, 그에 맞는 dataset이 충분히 존재하는지에 대한 조사가 최우선입니다. 

아래는 dataset을 찾을 때 자주 사용하는 사이트입니다. 

- [Hugging Face](https://huggingface.co/)\
다국어, 여러 task에 최적화되어 있는 데이터셋을 제공하는 사이트입니다. Hugging Face 자체적으로 개발한 dataloader 패키지를 사용하면 local에 데이터셋을 다운로드할 필요 없이 바로 사용할 수 있어 매우 편리합니다. 데이터셋뿐만 아니라 여러 모델도 제공해 유용하게 사용할 수 있습니다. 

- [AI hub](https://www.aihub.or.kr/)\
다량의 한국어 데이터셋을 제공하는 사이트입니다. 데이터셋을 다운로드하려면 회원가입이 필요합니다. 

[Kaggle](https://www.kaggle.com/), [Google Dataset Search](https://datasetsearch.research.google.com/) 등등 다른 유명한 사이트들도 많으나 저희가 가장 자주 사용한 사이트 위주로 정리했습니다. 구글링 또한 좋은 데이터셋 탐색 방법이니, 적절하게 사용하시면 되겠습니다.


## Error Handling
Training을 할 때 아주 많은 Error를 맞닥뜨릴 수 있습니다. 그 중에서 저희가 자주 마주친 Error들을 위주로, 덜 헤맬 수 있게 정리를 해놓았습니다.

### CUDA Out Of Memory
GPU RAM 에 RAM size보다 큰 데이터가 올라갔을 경우 발생합니다. batch size를 줄여 해결할 수 있습니다.

### Index Size Error

### cuda error: device-side assert triggered
다양한 이유가 있을 수 있지만, 대부분의 경우 tokenizer에 token을 추가한 후 모델과 embedding size가 맞지 않아 발생했습니다.
model.resize_token_embeddings(len(tokenizer))
위 코드를 token 추가 후에 삽입함으로써 해결했습니다.

### CUDA call failed lazily at initialization with error
```python
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
→
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
```

Training 시 사용할 GPU를 설정해주기 전에, torch가 0번 GPU를 차지하기 때문에 import torch보다 먼저 cuda의 visibility를 설정해줘야 합니다.
