# Pre-Training

이번에는 직접 pre-train한 language model의 성능을 측정해 봅니다, GPT-2의 경우, 최근에 나오고 있는 여러 LLM들에 비해 성능이 좋지는 않습니다. 그래서 이에 맞춰 5~6년 전의 Benchmark들을 시행해 볼 예정입니다. Benchmark의 과정은 크게 Fine-Tuning과 Evaluate의 과정으로 나뉩니다. Train dataset으로 pre-trained model을 fine-tuning 한 후 test dataset으로 이를 evaluate 해서 성능을 내는 것입니다.

## a. NSMC, IMDb Benchmark

**NSMC(Naver Sentimental Movie Corpus) Benchmark** 는 네이버 영화 사이트에서 영화 한줄평을 각각 Positive 와 Negative 로 분류해 놓은 dataset입니다. 약 200K의 리뷰가 있고, 이를 170K와 30K의 데이터셋으로 split 하여 학습시킵니다. <br>

IMDb Benchmark 는 IMDb라는 영화 사이트에서 영화 한줄평을 각각 Positive와 Negative로 분류해 놓은 dataset입니다.

## b. GLUE Benchmark

**GLUE (General Language Understanding Evaluation) Benchmark** 는 LLM 모델의 전반적인 능력을 평가하기 위한 Benchmark 입니다. GLUE는 총 아홉 종류의 작업으로 구성되어 있으며, 이를 통해 모델의 성능을 종합적으로 평가합니다. 다음은 GLUE Benchmark의 세부 평가 항목입니다. <br> <br>

**CoLA (Corpus of Linguistic Acceptability)**: 문장이 문법적으로 올바른지 아닌지를 판단합니다. <br>
<br>
**SST-2 (Stanford Sentiment Treebank)**: 문장의 감정을 긍정적 또는 부정적으로 분류합니다. 영화 리뷰 등의 감정 분석에 사용됩니다. <br>
<br>
**MRPC (Microsoft Research Paraphrase Corpus)**: 두 문장이 서로 의미가 같은지 아닌지를 판단합니다. <br>
<br>
**STS-B (Semantic Textual Similarity Benchmark)**: 두 문장의 의미적 유사성을 측정합니다. 문장 쌍 간의 유사도를 0부터 5까지의 스케일로 평가합니다. <br>
<br>
**QQP (Quora Question Pairs)**: 두 개의 질문이 동일한 의미를 가지는지 여부를 판단합니다. <br>
<br>
**MNLI (Multi-Genre Natural Language Inference)**: 두 문장 간의 논리적 관계를 추론합니다. 다양한 장르의 텍스트에서 가정, 반박, 중립을 판별합니다. <br>
<br>
**QNLI (Question Natural Language Inference)**: 질문과 문장 쌍에서 질문이 문장에서 답변을 포함하고 있는지를 판단합니다. <br>
<br>
**RTE (Recognizing Textual Entailment)**: 두 문장 간의 의미적 관계를 평가합니다. 주어진 문장과 가정 간의 관계가 참인지 거짓인지 판단합니다. <br>
<br>
**WNLI (Winograd Schema Challenge)**: 대명사의 의미를 해결하여 문장 간의 관계를 추론합니다. 문맥을 통해 대명사의 참조 대상을 식별합니다. <br>


<br>
<br>

여러 LLM 모델들의 GLUE Benchmark 점수는 다음 사이트에서 참고할 수 있습니다.   [GLUE Leaderboard](https://gluebenchmark.com/leaderboard)