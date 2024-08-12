# Fine-Tuning 개요

Fine-Tuning은 pre-train 된 language model을 세부 분야에 적합하게 성능을 낼 수 있도록 training 하는 작업입니다. Label이 된 dataset을 넣어 training 시키며, 이를 Supervised Learning이라고 합니다. 이 과정에서 저희는 pre-trained model을 가져와서 사용합니다. Language model을 직접 pre-train 하는 과정은 시간이 많이 걸리기 때문에, fine-tuning을 통해서 model이 어떻게 학습되는지, evaluate은 어떻게 할 수 있는지 등의 전체적인 과정을 먼저 확인해 보시기 바랍니다.

## a. German GPT2 Recipe

독일어 말뭉치로 pre-train 된 `dbmdz/german-gpt2`를 독일어로 작성된 recipe dataset을 이용해 fine-tuning 합니다.
- [German GPT2](https://huggingface.co/dbmdz/german-gpt2)
- [Original tutorial](https://github.com/philschmid/fine-tune-GPT-2/tree/master)


## b. Korean Chatbot (Single-turn)

SKT에서 GPT2를 기반으로 제작한 한국어 GPT2 모델 `skt/kogpt2-base-v2`를 한국어 챗봇 dataset을 이용해 fine-tuning 합니다.

- [KoGPT2](https://huggingface.co/skt/kogpt2-base-v2)
- [Korean Chatbot dataset](https://github.com/songys/Chatbot_data?tab=readme-ov-file)
