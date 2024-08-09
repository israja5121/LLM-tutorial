# NSMC, IMDB Benchmark

## 설명

`openai-community/gpt2` 와 `skt/kogpt2-base-v2` 를 이용하여 해당 Fine-Tuning을 진행합니다. \
\
`ratings_en_test.txt`, `ratings_en_train.txt` 은 IMDb Dataset 입니다. \
`ratings_ko_test.txt`, `ratings_ko_train.txt` 은 NSMC Dataset 입니다.


## Results
아래는 여러 알려진 IMDb와 NSMC Benchmark의 결과들입니다.

* NSMC Benchmark

  |       Model       | Accuracy (%) |
  | ----------------- | ------------ |
  | KoBERT            | **89.63**    |
  | DistilKoBERT      | 88.41        |
  | Bert-Multilingual | 87.07        |
  | FastText          | 85.50        |


<br/>


* IMDb Benchmark

  |       Model       | Accuracy (%) |
  | ----------------- | ------------ |
  | BERT large        | **95.49**    |
  | DistilBERT 66M    | 93.06        |
  | CNN+LSTM          | 88.9         |
  | VGG16             | 86           |


## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [IMDb Benchmark](https://paperswithcode.com/sota/sentiment-analysis-on-imdb)
- [NSMC dataset](https://github.com/e9t/nsmc)
