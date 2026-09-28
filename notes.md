# Char tfidf
## Params
ngram_range=(3, 5),
min_df=2,

python -m scripts.run_char_tfidf
Train rows: 406349
Validation rows: 91324
Local corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 530063)
Vocabulary size: 530063
Retrieving Candidates: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 258/258.02734375 [07:03<00:00,  1.64s/it]

Recall@10: 0.051592
Recall@20: 0.078290
Recall@50: 0.133350

Queries with <50 candidates: 5
Mean candidates: 49.997310030395134


# Word tfidf
## Params
ngram_range=(1, 2),
min_df=2,

python -m scripts.run_word_tfidf
Train rows: 406349
Validation rows: 91324
Local corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 385419)
Vocabulary size: 385419

Recall@10: 0.062766
Recall@20: 0.093236
Recall@50: 0.149818

Queries with <50 candidates: 1861
Mean candidates: 49.11588145896656


# BM25
## Params
ngram_range=(1, 1),
k1=1.5,
b=0.75,
min_df=2,

python -m scripts.run_bm25
Train rows: 406349
Validation rows: 91324
Local corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 78281)
Vocabulary size: 78281

Recall@10: 0.062347
Recall@20: 0.091431
Recall@50: 0.148368

Queries with <50 candidates: 1861
Mean candidates: 49.11588145896656


# BM25
## Params
ngram_range=(1, 2),
k1=1.5,
b=0.75,
min_df=2,

python -m scripts.run_bm25
Train rows: 406349
Validation rows: 91324
Local corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 385419)
Vocabulary size: 385419

Recall@10: 0.063441
Recall@20: 0.094005
Recall@50: 0.150592

Queries with <50 candidates: 1861
Mean candidates: 49.11588145896656


# Lexical fusion (rrf)
python -m scripts.run_lexical_fusion
word + bm25
  RRF Recall@50: 0.151888
  Pool recall:    0.166171
  Mean pool size: 63.19

char + word
  RRF Recall@50: 0.154720
  Pool recall:    0.192303
  Mean pool size: 79.49

char + bm25
  RRF Recall@50: 0.155315
  Pool recall:    0.192543
  Mean pool size: 78.56

char + word + bm25
  RRF Recall@50: 0.158917
  Pool recall:    0.203353
  Mean pool size: 89.87


# Standalone Dense Retriever
Recall@10: 0.074621
Recall@20: 0.110082
Recall@50: 0.176046

Queries with <50 candidates: 0
Mean candidates: 50.0


# Dense + Lexical fusions
python -m scripts.run_dense_lexical_fusion
dense
  RRF Recall@50: 0.176046
  Pool recall:    0.176046
  Mean pool size: 50.00

dense + bm25
  RRF Recall@50: 0.180775
  Pool recall:    0.230193
  Mean pool size: 87.31

dense + word
  RRF Recall@50: 0.180803
  Pool recall:    0.229771
  Mean pool size: 87.05

dense + char
  RRF Recall@50: 0.170968
  Pool recall:    0.222253
  Mean pool size: 89.43

dense + char + word + bm25
  RRF Recall@50: 0.171974
  Pool recall:    0.260035
  Mean pool size: 123.07

===== Dense weight test =====

dense_weight=1.0: recall@50=0.171974
dense_weight=1.5: recall@50=0.178058
dense_weight=2.0: recall@50=0.182792
dense_weight=3.0: recall@50=0.182217



# LightGBM
python -m scripts.run_lgbm_ranker
Validation signatures: 65800
Validation normalized queries: 14742

Meta train signatures: 45554
Meta eval signatures: 20246
Meta train queries: 10319
Meta eval queries: 4423
Loading dense: artifacts/predictions/dense_val.pkl
Loading bm25: artifacts/predictions/bm25_val.pkl
Loading word: artifacts/predictions/word_tfidf_val.pkl
Loading char: artifacts/predictions/char_tfidf_val.pkl

Building LightGBM train dataset...
Built dataset: 13059 queries, 1,500,628 candidate rows
Positive rows: 14121
Skipped queries without candidates: 0
Skipped train queries without positive in pool: 32495

X_train: (1500628, 12)
Train groups: 13059

Building LightGBM eval dataset...
Built dataset: 20246 queries, 2,568,016 candidate rows
Positive rows: 6215
Skipped queries without candidates: 0
Skipped train queries without positive in pool: 0

X_eval: (2568016, 12)
Eval groups: 20246

Training LightGBM...
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.016896 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 740
[LightGBM] [Info] Number of data points in the train set: 1500628, number of used features: 12
/home/cohle/machine_learning/avito-service-retrieval/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2830: UserWarning: X does not have valid feature names, but LGBMRanker was fitted with feature names
  warnings.warn(

### RESULTS
Dense Recall@50:        0.170228
Weighted RRF Recall@50: 0.175106
LightGBM Recall@50:     0.173694
Full pool recall:       0.255821
Mean pool size:        126.84

Feature importance:
            feature  importance
          mean_rank        1760
reciprocal_rank_sum        1645
          bm25_rank        1286
         dense_rank        1285
          word_rank        1218
          char_rank        1111
          best_rank         431
        num_sources         239
       char_present          11
       bm25_present           7
      dense_present           4
       word_present           3



# Potential Architecture
                         QUERY
                           │
          ┌────────────────┴───────────────┐
          │                                │
        LOCAL                            GLOBAL
          │                                │
    same location                       all items
          │                                │
   ┌──────┼──────┐                ┌────────┼───────┐
   │      │      │                │        │       │
 char   word   dense             char     word    dense
   │      │      │                │        │       │
   └──────┴──────┴────────────────┴────────┴───────┘
                           │
                     microcat/history
                           │
                         UNION
                         ~150+
                           │
                        LightGBM
                           │
                         TOP 50



# Local word TF-IDF
python -m scripts.run_local_word_tfidf
Corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 409669)
Vocabulary: 409669
Locations: 2637

Recall@10: 0.485176
Recall@20: 0.555936
Recall@50: 0.630114

Queries with <50 candidates: 19916
Mean candidates: 37.305851063829785

Saved: artifacts/predictions/word_local_val.pkl

## Понятно почему учет локации кратно поднимает recall - для услуг локальность еще важнее чем просто для товаров. Врядли человек поедет в другой город для условного массажа.


# Local char TF-IDF
python -m scripts.run_local_char_tfidf
Corpus items: 344825
Validation signatures: 65800
TF-IDF matrix: (344825, 192981)
Vocabulary: 192981
Locations: 2637

Recall@10: 0.501239
Recall@20: 0.567846
Recall@50: 0.631423

Queries with <50 candidates: 16835
Mean candidates: 39.74562310030395

Saved: artifacts/predictions/char_local_val.pkl


# Local lexical fusion
python -m scripts.run_local_lexical_fusion
Validation signatures: 65800

word local
  Recall@10: 0.485176
  Recall@20: 0.555936
  Recall@50: 0.630114
  Mean candidates: 37.31

char local
  Recall@10: 0.501239
  Recall@20: 0.567846
  Recall@50: 0.631423
  Mean candidates: 39.75

RRF char + word
  Recall@10: 0.529054
  Recall@20: 0.609694
  Recall@50: 0.687223
  Mean candidates: 41.67

Full local lexical pool
  Pool recall: 0.712561
  Mean pool size: 64.06
  Median pool size: 75.00
  Max pool size: 100

Candidate overlap
  Mean intersection: 12.99
  Mean union: 64.06


# Local Dense retrieval
python -m scripts.run_local_dense                                                           
Corpus items: 344825
Validation signatures: 65800
Validation locations: 1811
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8040.02it/s]
Encoding items...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2694/2694 [04:13<00:00, 10.64it/s]
Item embeddings: (344825, 768)
Saved embeddings: artifacts/embeddings/e5_base_items.npy
Embedding matrix: (344825, 768)
Locations: 2637
Queries: 65800
Unique queries: 21097
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 165/165 [00:03<00:00, 49.31it/s]
Dense retrieval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1811/1811 [00:01<00:00, 1067.72it/s]

Recall@10: 0.552546
Recall@20: 0.627787
Recall@50: 0.697309

Queries with <50 candidates: 11412
Mean candidates: 42.44062310030395
Median candidates: 50.0
Min candidates: 0

Saved predictions: artifacts/predictions/dense_local_val.pkl
Saved scores: artifacts/predictions/dense_local_scores_val.pkl


# Local lexical + local/global dense fusion
python -m scripts.run_local_dense_fusion
Loading char_local: artifacts/predictions/char_local_val.pkl
Loading word_local: artifacts/predictions/word_local_val.pkl
Loading dense_local: artifacts/predictions/dense_local_val.pkl
Loading dense_global: artifacts/predictions/dense_val.pkl

### STANDALONE
char local           R@50=0.631423 | mean candidates=39.75
word local           R@50=0.630114 | mean candidates=37.31
dense local          R@50=0.697309 | mean candidates=42.44
dense global         R@50=0.176046 | mean candidates=50.00

### RRF
dense local + char local                      RRF@50=0.697762
dense local + word local                      RRF@50=0.713216
dense local + char + word                     RRF@50=0.714407
dense(x1.5) + char + word                     RRF@50=0.712354
dense(x2) + char + word                       RRF@50=0.707256

### POOL COVERAGE
char + word                                   pool_recall=0.712561 | mean_pool=64.06
dense local + char                            pool_recall=0.722644 | mean_pool=68.14
dense local + word                            pool_recall=0.735240 | mean_pool=62.68
dense local + char + word                     pool_recall=0.745797 | mean_pool=84.40
local all + global dense                      pool_recall=0.794956 | mean_pool=133.39



# MICROCAT PRIOR QUALITY
Microcat Recall@1: 0.781337
Microcat Recall@3: 0.920865
Microcat Recall@5: 0.949018
Microcat Recall@10: 0.971355
Microcat Recall@20: 0.979855

Top-1 neighbor similarity:
  mean:   0.980595
  median: 0.988292
  p10:    0.951477
  p90:    0.998283

Mean top-5 neighbor similarity:
  mean:   0.969674
  median: 0.976632


# Следующий эксперимент - добавить microcat prior как фичу для LightGBM: сколько добавляет microcat prior как reranking feature?
# Потом - использовать top-3/top-5 predicted microcats для генерации новых кандидатов. Это может поднять pool recall ceiling выше текущего 0.7935,
# что ranker уже сделать не способен.



# Result after adding microcat top-1 as a 5th candidate generation source
RANKER EVALUATION
Rich LightGBM
  Recall@10: 0.633307
  Recall@20: 0.714048
  Recall@50: 0.779861

================================================================================
FEATURE IMPORTANCE
================================================================================
                     feature  gain_pct  split
                   rrf_score    63.056    477
        microcat_prior_score     5.063    642
     local_dense_query_range     3.820    493
            dense_local_rank     3.347    241
               same_location     3.288    197
  query_location_items_log1p     2.914    542
            microcat_support     2.835    355
           best_present_rank     2.371    275
             word_local_rank     1.833    476
          local_source_count     1.581     81
      local_dense_gap_to_top     1.152    416
local_dense_margin_over_last     1.041    349
       item_params_len_words     0.557    729
         microcat_prior_rank     0.555    160
           mean_present_rank     0.551    421
          query_title_recall     0.549    256
           dense_global_rank     0.489    347
   qparams_itemparams_recall     0.404    330
    neighbor_query_sim_mean5     0.374    469
          char_local_present     0.371     90
   item_location_items_log1p     0.362    446
             title_len_words     0.359    333
  qparams_itemparams_jaccard     0.352    437
                 log_reviews     0.328    575
         dense_microcat_rank     0.322    169
           local_dense_score     0.232    393
        dense_microcat_score     0.200    224
        exact_query_in_title     0.198    152
                   log_price     0.190    447
   dense_microcat_gap_to_top     0.187    258
     neighbor_query_sim_top1     0.166    333
                 item_rating     0.129    285
       query_title_precision     0.128    189
        qparams_title_recall     0.125    202
         query_title_overlap     0.104    127
         query_location_lt50     0.102     10
  qparams_itemparams_overlap     0.096    167
         query_title_jaccard     0.076    142
           global_dense_only     0.053      5
   all_query_tokens_in_title     0.031     14
         microcat_top1_match     0.031      9
        dense_global_present     0.017     14
           item_phone_hidden     0.011     29
                 num_sources     0.010     28
             char_local_rank     0.008     21
         dense_local_present     0.008      2
          word_local_present     0.008      6
      item_message_forbidden     0.006     15
         microcat_top5_match     0.005     10
      dense_microcat_present     0.004      2
         microcat_top3_match     0.000      0

================================================================================
SAVE
================================================================================
Saved model: artifacts/models/lgbm_rich_ranker.txt
Saved predictions: artifacts/predictions/lgbm_rich_val.pkl


# Global dense recall after fix
Recall@10: 0.072943
Recall@20: 0.108067
Recall@50: 0.174138



# LightGBM on all sources raw score features
RANKER EVALUATION
Rich LightGBM
  Recall@10: 0.635334
  Recall@20: 0.713596
  Recall@50: 0.776279

FEATURE IMPORTANCE
                         feature  gain_pct  split
                       rrf_score    44.013    242
               best_present_rank    14.546    228
            microcat_prior_score     4.643    473
                   same_location     4.505    170
    dense_local_margin_over_last     4.159    316
                microcat_support     3.944    244
   dense_local_query_score_range     2.761    290
              local_source_count     2.738     69
     char_local_margin_over_last     2.638    231
    char_local_query_score_range     2.145    301
          dense_local_gap_to_top     1.806    319
      query_location_items_log1p     1.425    289
               dense_local_score     1.121    289
           word_local_gap_to_top     1.012    281
   dense_global_margin_over_last     0.768    283
     word_local_margin_over_last     0.588    313
             microcat_prior_rank     0.514    124
                char_local_score     0.503    206
                 title_len_words     0.453    293
              query_title_recall     0.430    197
           item_params_len_words     0.414    509
               mean_present_rank     0.403    240
           char_local_gap_to_top     0.387    197
       qparams_itemparams_recall     0.377    257
      qparams_itemparams_jaccard     0.351    333
                word_local_score     0.342    249
       item_location_items_log1p     0.297    302
                     log_reviews     0.271    407
 dense_microcat_margin_over_last     0.253    139
        neighbor_query_sim_mean5     0.240    250
    word_local_query_score_range     0.178    162
  dense_global_query_score_range     0.175    254
                       log_price     0.136    282
            qparams_title_recall     0.135    179
             query_title_overlap     0.134     81
           query_title_precision     0.123    124
dense_microcat_query_score_range     0.117    180
              dense_global_score     0.115    133
             query_title_jaccard     0.111    143
         neighbor_query_sim_top1     0.101    170
            dense_microcat_score     0.099    122
      qparams_itemparams_overlap     0.092    108
                     item_rating     0.091    172
            exact_query_in_title     0.085     69
       dense_microcat_gap_to_top     0.081    144
         dense_global_gap_to_top     0.067    119
             microcat_top1_match     0.036     16
       all_query_tokens_in_title     0.027      8
               global_dense_only     0.021      2
               item_phone_hidden     0.010     24
             query_location_lt50     0.008      6
                     num_sources     0.006     11
          item_message_forbidden     0.003      8
             microcat_top5_match     0.001      1
             microcat_top3_match     0.001      1


# Добавление item description к dense энкодеру
Dense description LOCAL
  Recall@10: 0.570208
  Recall@20: 0.644935
  Recall@50: 0.714084
  Mean candidates: 42.44

Dense description GLOBAL
  Recall@10: 0.073136
  Recall@20: 0.107423
  Recall@50: 0.176051
  Mean candidates: 50.00

================================================================================
POOL COVERAGE
================================================================================
Current 5 sources                             recall=0.798702 | mean_pool=148.64
+ local dense description                     recall=0.811239 | mean_pool=159.08
+ global dense description                    recall=0.808653 | mean_pool=173.59
+ local + global dense description            recall=0.820621 | mean_pool=183.98

================================================================================
POOL GAINS
================================================================================
Local desc gain: +0.012537
Global desc gain: +0.009951
Both desc gain: +0.021919



# BGE-M3 vs E5 Ablation
```bash
.venv/bin/python -m scripts.run_bge_m3_ablation --stage local --batch-size 32
.venv/bin/python -m scripts.run_bge_m3_ablation --stage global --batch-size 32
.venv/bin/python -m scripts.run_bge_m3_ablation --stage microcat --batch-size 32
.venv/bin/python -m scripts.run_bge_m3_ablation --stage evaluate
```

 Источник                  E5 Recall@50    BGE-M3     delta
━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
 Dense local                     69,73%    68,05%      −1,69
────────────────────────  ──────────────  ────────  ─────────
 Dense global                    17,41%    14,49%      −2,92
────────────────────────  ──────────────  ────────  ─────────
 Dense microcat                  14,78%    14,13%      −0,64
────────────────────────  ──────────────  ────────  ─────────
 Local RRF                       71,44%    71,14%      −0,30
────────────────────────  ──────────────  ────────  ─────────
 Общий RRF                       71,72%    71,42%      −0,30
────────────────────────  ──────────────  ────────  ─────────
 Полнота candidate pool          79,87%    79,49%      −0,38



# LightGBM Ranker on added dense description features
RANKER EVALUATION
Rich LightGBM
  Recall@10: 0.646129
  Recall@20: 0.724659
  Recall@50: 0.790243

FEATURE IMPORTANCE
                            feature  gain_pct  split
                          rrf_score    37.216    263
              dense_desc_local_rank    18.584    221
                 local_source_count     8.845     78
                  best_present_rank     4.761    198
                   microcat_support     3.439    283
               microcat_prior_score     3.264    491
 dense_desc_local_query_score_range     2.582    250
        char_local_margin_over_last     2.283    247
      dense_local_query_score_range     2.078    181
       char_local_query_score_range     1.738    290
                      same_location     1.656    174
  dense_desc_local_margin_over_last     0.996    277
        dense_desc_local_gap_to_top     0.933    276
             dense_local_gap_to_top     0.851    257
         query_location_items_log1p     0.836    309
              word_local_gap_to_top     0.796    300
        word_local_margin_over_last     0.722    259
                   char_local_score     0.566    246
                 query_title_recall     0.513    215
                microcat_prior_rank     0.453    119
 dense_desc_global_margin_over_last     0.387    216
             dense_desc_local_score     0.375    242
          qparams_itemparams_recall     0.373    265
                    title_len_words     0.371    307
              item_params_len_words     0.346    499
              char_local_gap_to_top     0.336    204
                        num_sources     0.329     26
                  mean_present_rank     0.328    258
          item_location_items_log1p     0.294    345
      dense_global_margin_over_last     0.258    155
                        log_reviews     0.252    430
                   word_local_score     0.231    217
           neighbor_query_sim_mean5     0.225    299
    dense_microcat_margin_over_last     0.196    130
         qparams_itemparams_jaccard     0.195    288
                  dense_local_score     0.161    233
     dense_global_query_score_range     0.155    251
       word_local_query_score_range     0.154    187
                          log_price     0.127    311
               dense_microcat_score     0.122    104
            dense_desc_global_score     0.118    175
                        item_rating     0.115    249
            neighbor_query_sim_top1     0.108    225
                query_title_overlap     0.106     91
dense_desc_global_query_score_range     0.103    211
               exact_query_in_title     0.101     75
            dense_global_gap_to_top     0.099     93
               qparams_title_recall     0.096    161
       dense_local_margin_over_last     0.096    162
              query_title_precision     0.094    138
   dense_microcat_query_score_range     0.092    189
                query_title_jaccard     0.075    136
         qparams_itemparams_overlap     0.075    115
                 dense_global_score     0.074     96
          dense_microcat_gap_to_top     0.072    130
       dense_desc_global_gap_to_top     0.055    122
             dense_desc_global_rank     0.046     73
                microcat_top1_match     0.045     15
             dense_microcat_present     0.039      1
          all_query_tokens_in_title     0.029     14
                query_location_lt50     0.013      5
                microcat_top5_match     0.008     10
                  item_phone_hidden     0.007     18
           dense_desc_local_present     0.003      1
             item_message_forbidden     0.003      8
                microcat_top3_match     0.002      4
                  global_dense_only     0.001      2
          dense_desc_global_present     0.000      0
