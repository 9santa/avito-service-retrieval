# Avito Services Retrieval

Candidate retrieval для поиска услуг Avito.

## Данные

Ожидаемая структура:

```text
data/
├── raw/
│   ├── train.parquet
│   ├── benchmark_queries.parquet
│   └── benchmark_items.parquet
└── interim/
    ├── train_part.parquet
    └── val_part.parquet
```

`train.parquet` содержит пары «поисковый запрос → кликнутое объявление». Benchmark состоит из 2 452 запросов и корпуса из 189 212 объявлений. Исходные данные и веса моделей в репозиторий не включены (они большие).

## Валидация

Основная проблема локальной оценки — leakage между одинаковыми запросами. Поэтому split строится по нормализованному тексту запроса: один `search_query_norm` целиком попадает либо в train, либо в validation.

Единица оценки — `query_signature`, то есть комбинация:

```text
search_query_norm
search_location_id
search_is_delivery_search
search_infm_params_text
search_category
```

Фиксированный split:

- train: 406 349 interaction rows;
- validation: 91 324 rows;
- validation: 65 800 query signatures.

Создание split:

```bash
python -m scripts.export_validation_split
```

## Финальный pipeline

Candidate pool собирается из семи источников:

1. **Local char TF-IDF** — char n-grams 3–5, `query → item_title`, поиск только внутри `search_location_id`.
2. **Local word TF-IDF** — `query + search params → title×2 + item params`, word 1–2 grams, та же location.
3. **Local dense** — `multilingual-e5-base`, `query + params → title + params`, exact cosine search внутри location.
4. **Global dense** — тот же E5 по всему corpus; используется как важный доп. источник для out-of-location positives.
5. **Microcat-guided dense** — nearest historical queries предсказывают microcategory, затем dense search выполняется внутри top-1 microcat.
6. **Local dense + description** — E5, item text `title + params + description`, `max_seq_length=256`, поиск внутри location.
7. **Global dense + description** — то же самое что и №6, только по всему корпусу.

Union кандидатов передаётся в `LightGBMRanker` (`lambdarank`). Selector использует ranks/presence источников, RRF, raw retrieval scores и score margins, location-признаки, query-item lexical overlap, item metadata и microcat prior.

Microcat prior для validation строится **только по `train_part`**, чтобы query не видел собственные клики. Для benchmark prior строится по всему `train.parquet`.

## Ключевые гипотезы и результаты

### 1. Location — основной routing signal

Same-location positive rate оказался около 0.83 по rows и 0.80 macro по signatures. Hard-filter только по location поэтому недостаточен (около 20% в приниципе отпадают), но local retrieval резко сужает пространство поиска.

| Retriever | Recall@50 |
|---|---:|
| Global word TF-IDF | 0.1498 |
| Global BM25 (1–2 grams) | 0.1506 |
| Local word TF-IDF | **0.6301** |
| Local char TF-IDF | **0.6314** |
| Local char + word RRF | **0.6872** |
| Local dense E5 | **0.6973** |

Команды:

```bash
python -m scripts.run_local_word_tfidf
python -m scripts.run_local_char_tfidf
python -m scripts.run_local_lexical_fusion
python -m scripts.run_local_dense
```

Вывод: для услуг location надо использовать уже на candidate generation, но global как дополнительный источник необходимо сохранять.

### 2. Разные retriever-ы дополняют друг друга

Local char + word + dense давали pool recall `0.7458`. Добавление global dense поднимало coverage примерно до `0.795`, хотя standalone global dense был слабым (~0.17 Recall@50). Это хороший пример источника, ценного не самостоятельным quality, а уникальными positives.

```bash
python -m scripts.run_local_dense_fusion
python -m scripts.run_dense
```

### 3. Microcategory хорошо переносит intent

Nearest-query prior по `train_part`:

| Metric | Value |
|---|---:|
| Microcat Recall@1 | 0.7813 |
| Recall@3 | 0.9209 |
| Recall@5 | 0.9490 |
| Recall@10 | 0.9714 |

```bash
python -m scripts.build_microcat_prior
python -m scripts.run_dense_microcat
```

Как feature microcat prior заметно улучшил selector. Как candidate generator лучший pool gain дал именно top-1 microcat; расширение до top-3/5/10 повышало standalone recall, но приносило меньше новых positives поверх существующего pool.

### 4. Raw retrieval scores полезны, но важнее candidate coverage

Для всех retrieval branches сохраняются raw cosine/TF-IDF scores. В ranker добавлены `score`, `gap_to_top`, `margin_over_last`, `query_score_range`. LightGBM активно использует эти признаки, но прирост Recall@50 оказался небольшим: они сильнее улучшают порядок в верхней части списка, чем границу 50/51.

Контрольный `LGBMClassifier` с query-balanced weights дал Recall@50 `0.7760` и не обошёл LambdaRank, поэтому финальным selector остался `LGBMRanker`.

```bash
python -m scripts.run_lgbm_classifier
```

### 5. Description оказался полезным независимым сигналом

| Source | Recall@50 |
|---|---:|
| Dense description local | **0.7141** |
| Dense description global | 0.1761 |

На полном validation candidate pool:

| Pool | Recall |
|---|---:|
| 5 sources | 0.7987 |
| + local description | 0.8112 |
| + global description | 0.8087 |
| + both description branches | **0.8206** |

```bash
python -m scripts.run_dense_description
```

После добавления обеих description веток финальный LambdaRank на фиксированном meta-eval показал:

```text
Recall@10 = 0.6461
Recall@20 = 0.7247
Recall@50 = 0.7902
```

### 6. BGE-M3 не улучшил E5

При той же retrieval-схеме BGE-M3 оказался немного хуже `multilingual-e5-base`: local dense −1.69 п.п. Recall@50, global dense −2.92 п.п.; общий pool также немного ухудшился. Поэтому в финальном решении оставлен E5.

Воспроизведение ablation:

```bash
python -m scripts.run_bge_m3_ablation --stage local --batch-size 32
python -m scripts.run_bge_m3_ablation --stage global --batch-size 32
python -m scripts.run_bge_m3_ablation --stage microcat --batch-size 32
python -m scripts.run_bge_m3_ablation --stage evaluate
```

Более подробные логи промежуточных экспериментов сохранены в `notes.md`.

## Воспроизведение финальной модели

Окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Основные validation artifacts удобно строить в таком порядке:

```bash
python -m scripts.export_validation_split
python -m scripts.run_local_word_tfidf
python -m scripts.run_local_char_tfidf
python -m scripts.run_local_dense
python -m scripts.run_dense
python -m scripts.build_microcat_prior
python -m scripts.run_dense_microcat
python -m scripts.run_dense_description
python -m scripts.run_lgbm_ranker
```

Embeddings и predictions кэшируются в `artifacts/`, поэтому повторные запуски части экспериментов дешевле.

Финальный benchmark и submission:

```bash
python -m submission.build_benchmark_sources
python -m submission.run_final_submission
```

`submission/run_final_submission.py` проверяет соответствие candidate lists и raw scores, обучает финальный selector на leakage-safe validation features и формирует `answer.csv`.

Формат результата:

```text
query_id,answer
<query_id>,<item_id_1> <item_id_2> ... <item_id_50>
```

В каждой строке не более 50 уникальных `item_id`; файл сохраняется без индекса.

## Структура проекта

```text
src/         reusable retrieval, validation, fusion и ranker feature code
scripts/     validation experiments и ablations
submission/  построение benchmark sources и финального answer.csv
notebooks/   EDA/diagnostics
notes.md     подробные логи экспериментов
```

