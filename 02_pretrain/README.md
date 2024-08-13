# Pre-Training 개요

Pre-Training은 Fine-Tuning 이전에 Language Model의 성능을 향상시키기 위한 작업입니다. Fine-Tuning과는 다르게 Corpus, 즉 말뭉치들을 넣어서 성능을 향상시키는 것이 특징입니다. 따로 label이 없기 때문에 Unsupervised Learning 이라고 부릅니다.

## a. Tokenizer

Transformer 모델에 대해서 배우셨다면, tokenizer 의 존재도 이해할 수 있을 것입니다. 우리는 text를 받아서 text를 출력하는 model을 다루고 있습니다. 그 사이에 이 text 자체를 word embedding vector로 바꾸는 과정이 있었음을 기억할 것입니다. 그렇다면 word embedding으로 어떻게 바꾸는 것일까요? 바로 이 text를 어떤 방식으로 tokenize 하여 그에 맞는 token을 찾아 embedding vector를 대응시켜 주는 것입니다.

그렇다면 이 tokenizer는 어떻게 만드는 것일까요? GPT-2의 경우를 예를 들어 설명하자면, tokenizer 또한 corpus에서 train됩니다. 물론 language modeling 하는 중에 train이 되는 것은 아니고, tokenizer을 미리 train 시켜 놓고, 이 tokenizer을 이용해서 language modeling을 진행하는 것입니다.

tokenizer을 train할 때 주목할 것은 vocab.json 과 `merge.txt` 입니다. 처음에 tokenizer는 문장을 음절 단위로 해체합니다. “안녕하세요.” 를 예로 들어보겠습니다.

<center>[‘안’, ‘녕’, ‘하’, ‘세’, ‘요’]</center> <br>

이후에 `merge.txt`가 이 분리된 음절들을 vocab.json에 있는 단어로 뭉칩니다. 뭉치는 기준은 이전에 train 하는 과정에서 학습된 것입니다. 자주 마주쳤던 단어들을 위주로 뭉치게 될 것입니다. 여기서는 ‘안녕’과 ‘하세요’가 뭉칠 가능성이 큽니다. (혹은 `merge.txt`에 ‘안녕하세요’ 도 있다면 ‘안녕하세요’가 한 vocab이 되는 것입니다.)

<center>[‘안녕’, ‘하세요’]</center> <br>

이렇게 되면 이 두 단어들은 vocab.json에 존재하는 단어입니다. 그렇다면 이 vocab들을 찾아서 token id로 바꿔줍니다.

<center>[2312, 40203]</center> <br>

Tokenizer을 얼마나 train 해야 하는지 궁금하실 수 있습니다. 어느 정도로 train 해야 하는가에 대해 정확히 기준이 있지는 않지만, GPT-2의 경우 Vocab size는 약  50000개를 기준으로 했을 때, huggingface에서 제시한 traininng dataset의 size는 약 [16MB 였습니다](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb#scrollTo=ulliiRU8pGN6).



## b. Language Modeling

Language Modeling을 할 때, model의 종류에 따라 modeling을 하는 방법이 다릅니다. GPT의 경우 이전 token들을 보고 다음 token을 예측하는 CLM(Causal Language Modeling)으로 training 합니다. BERT와 같은 Encoder model들은 문장의 일부분을 mask한 후 그 위치의 token을 예측하는 MLM(Masked Language Modeling)으로 training합니다.

Language Modeling의 경우는 어느 정도의 corpus를 넣어야 할까요? OpenAI의 GPT-2 모델의 경우 40GB 가량의 corpus로 modeling 했다고 합니다. SKT사의 KoGPT2 또한 마찬가지로 40GB 가량을, German GPT2 의 경우 16GB 가량을 training 했다고 합니다.

Language Modeling을 통해서 Non-English LLM을 만들 수 있습니다. 해당 프로젝트에서는 한국어와 독일어 모델을 만들었으며, 사용한 데이터셋은 다음과 같습니다. Tokenizer은 각각 `skt/kogpt2-base-v2` 와 `dbmdz/german-gpt2` 의 Tokenizer을 사용했습니다.