# CausalLanguageModeling

run_clm.py를 사용할 때 쓰는 parameter를 설명합니다. 

**예시**

```bash
$ CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python3 run_clm.py  --train_file ./dataset/training.txt    --model_type gpt2 --tokenizer_name openai-community/gpt2    --per_device_train_batch_size 128    --per_device_eval_batch_size 128    --do_train    --do_eval    --output_dir ./output  --save_steps 500  --save_total_limit 2  --overwrite_output_dir False
```


**CUDA_VISIBLE_DEVICES**            0~7번의 GPU중 사용할 GPU를 선택 \
**--train_file**                    model에게 train시킬 text의 주소 (e.g. ./dataset/training.txt) \
**--model_type**                    (처음부터 modeling 할 때 사용) model의 type (e.g. gpt2, bert-base-uncased...) \
**--model_name_or_path**            (이미 pretrain된 model의 경우) model의 path (e.g. ./output/mymodel/) \
**--tokenizer_name**                tokenizer의 이름 (e.g. dbmdz/german-gpt2, skt/kogpt2-base-v2) \
**--per_device_train_batch_size**   하나의 GPU당 Batch size (train 시) \
**--per_device_eval_batch_size**    하나의 GPU당 Batch size (evaluate 시) \
**--do_train**                      train을 시킴 \
**--do_eval**                       evaluate을 시킴 \
**--output_dir**                    model output의 주소 (e.g. ./output) \
**--save_steps**                    N step마다 checkpoint로 model을 임시 저장 (e.g. 500, 1000) \
**--save_total_limit**              checkpoint model이 저장되는 최대 갯수. 넘어가면 이전에 저장되었던 checkpoint 자동 삭제 (e.g. 2, 3, 4) \
**--overwrite_output_dir**          output directory에 train 결과를 overwrite 할 것인지 선택 \
\
(**중요!** Training을 끊고 재개할 때 overwrite_output_dir를 False로 지정해야 checkpoint부터 시작할 수 있다. True로 재개하면 처음부터 다시 시작. 처음 시작할 때는 True를 사용하고, 이후 끊고 재개할 때는 False로 바꾸어 줄 것.)

