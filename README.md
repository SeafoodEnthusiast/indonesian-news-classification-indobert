# Indonesian Sports News Classification with IndoBERT

A multiclass text classification project that fine-tunes IndoBERT to sort Indonesian sports news into five categories, and compares a single-stage classifier against a two-stage hierarchical pipeline.

[Video Walkthrough](https://youtu.be/zw3CpZmCiTM)

## Problem

A research institute needs a model that can tell apart sports news by competition: the English, Indonesian, Spanish and Italian football leagues, plus all non-football sports. There is no labelled dataset to start from, so the data has to be built first, then used to train and evaluate the classifier.

There are two natural ways to structure the model. A **one-stage** classifier predicts one of five labels directly. A **two-stage** classifier first decides whether an article is about football at all, then sends football articles to a second model that picks the league. The two-stage design mirrors how a person would sort the articles, but it adds a second model whose mistakes depend on the first. This project builds both and measures which one works better.

## Dataset

The notebook builds a dataset of 100 short Indonesian news texts with three columns (`Teks`, `Media`, `Label`) and writes it to `dataset_berita_100.csv`.

| Label | Texts |
|---|---|
| Liga Inggris | 20 |
| Liga Indonesia | 20 |
| Liga Spanyol | 20 |
| Liga Italia | 20 |
| Non Sepak Bola | 20 |

The `Media` column attributes the texts to four outlets: Detik.com, Kompas.com, Bola.net and Liputan6.com. The classes are perfectly balanced.

## Approach

### Preprocessing

Each text goes through the same pipeline before modelling:

1. **Cleansing.** Lowercased and stripped everything except letters and whitespace.
2. **Tokenization and filtering.** Split on whitespace and removed Indonesian stopwords using the Sastrawi stopword list.
3. **Stemming.** Reduced words to their root forms with the Sastrawi stemmer.

### Model

Both approaches fine-tune `indobenchmark/indobert-base-p1`, a BERT model pretrained on the Indo4B Indonesian corpus. I chose it over a multilingual BERT because it was trained specifically on Indonesian text.

| Setting | Value |
|---|---|
| Epochs | 10 (raised from the default of 3, because the dataset is small) |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Max sequence length | 128 |
| Split | 80/20 train/test, fixed random seed |

### One-stage vs two-stage

| Approach | Structure |
|---|---|
| **1-stage** | One model predicts all five labels directly. |
| **2-stage** | Stage 1 is a binary model (football vs non-football). Stage 2 is a four-class model that is trained only on football articles and picks the league. |

## Results

Metrics on the held-out test set. Precision, recall and F1 are weighted averages across classes.

| Model | Test size | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1-stage (5 classes) | 20 | 0.900 | 0.917 | 0.900 | 0.902 |
| 2-stage, stage 1 (football vs non-football) | 20 | 0.900 | 0.912 | 0.900 | 0.891 |
| 2-stage, stage 2 (league) | 16 | 0.875 | 0.911 | 0.875 | 0.854 |

All three models reached 1.00 accuracy and F1 on the training data, so each of them fits the training set completely.

On this data the one-stage model is the simpler choice and performs at least as well. The two-stage design has a structural weakness: if stage 1 sends a football article to the wrong branch, stage 2 cannot recover it, so errors compound.

These numbers need to be read carefully. The test sets are very small, so a single article moves accuracy by 5 to 6 points. The stage 2 score is also computed only on articles that really are football, which means it assumes stage 1 was perfect. It is not a true end-to-end figure, and it is measured on a different test set from the one-stage model. The honest conclusion is that the two approaches are not clearly separable at this data size, and the one-stage model is preferred mainly because it is simpler and has no error propagation.

## Repository Structure

```
├── UASTextMining_No1_AlessandroMorenoLawadinata_2702267672.ipynb   # Dataset, preprocessing, 1-stage and 2-stage training, comparison
├── requirements.txt                                                # Dependencies
├── LICENSE
└── README.md
```

The notebook creates `dataset_berita_100.csv` and `dataset_berita_100_clean.csv` itself when it runs, so neither file needs to be downloaded.

## Running Locally

```bash
git clone https://github.com/SeafoodEnthusiast/indonesian-news-classification-indobert.git
cd indonesian-news-classification-indobert
pip install -r requirements.txt
jupyter notebook
```

Open `UASTextMining_No1_AlessandroMorenoLawadinata_2702267672.ipynb` and run the cells in order from the top. Training uses a GPU if one is available, and the notebook was run on a CUDA GPU. On a CPU it will work but takes noticeably longer.

## Notes on Some Decisions

I used IndoBERT instead of a multilingual model because the data is entirely in Indonesian, and a model pretrained on Indonesian text understands its vocabulary and word forms better.

I raised the number of epochs to 10 because the training set is only 80 texts. With so little data, three epochs would leave the classification head undertrained.

Looking back at this project, these are the things I would change first:

- **Collect real, longer data.** The dataset is small and the texts are short, sentence-length snippets rather than full articles. They were compiled by hand, not scraped programmatically. A scraper over several outlets would give far more data and make the evaluation meaningful.
- **Fix the random seed.** The dataset is shuffled without a seed when it is built, so the exact train/test split, and therefore the exact scores, can differ slightly between runs.
- **Do a proper hyperparameter search.** I fixed the epochs and learning rate instead of searching over them. A small grid over learning rate and epochs, scored with cross-validation, would justify the choice.
- **Use a validation set.** The split is only train and test, so the test set doubles as the validation set. A third split, or cross-validation, would keep the test set untouched.
- **Reconsider stemming.** IndoBERT has its own subword tokenizer and was pretrained on unstemmed text. Stemming before tokenization moves the words away from what the model has seen, so it may be hurting rather than helping.
- **Evaluate the 2-stage pipeline end to end.** Feeding stage 1 predictions into stage 2 and scoring the full pipeline on one shared test set would make the comparison with the one-stage model fair.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · Sastrawi

## Context

Built as the final exam project for the Text Mining course, Data Science program, Bina Nusantara University, January 2026.

Alessandro Moreno Lawadinata
[LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/SeafoodEnthusiast)| 2-stage, stage 2 (league) | 16 | 0.875 | 0.911 | 0.875 | 0.854 |

All three models reached 1.00 accuracy and F1 on the training data, so each of them fits the training set completely.

On this data the one-stage model is the simpler choice and performs at least as well. The two-stage design has a structural weakness: if stage 1 sends a football article to the wrong branch, stage 2 cannot recover it, so errors compound.

These numbers need to be read carefully. The test sets are very small, so a single article moves accuracy by 5 to 6 points. The stage 2 score is also computed only on articles that really are football, which means it assumes stage 1 was perfect. It is not a true end-to-end figure, and it is measured on a different test set from the one-stage model. The honest conclusion is that the two approaches are not clearly separable at this data size, and the one-stage model is preferred mainly because it is simpler and has no error propagation.

## Repository Structure

```
├── dataset_berita_100.csv       # Labelled news texts
├── 1_text_classification.ipynb  # Preprocessing, 1-stage and 2-stage training, comparison
├── requirements.txt             # Dependencies
├── LICENSE
└── README.md
```

## Running Locally

```bash
git clone https://github.com/SeafoodEnthusiast/indonesian-news-classification-indobert.git
cd indonesian-news-classification-indobert
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run the cells in order. Training uses a GPU if one is available, and the notebook was run on a CUDA GPU. On a CPU it will work but takes noticeably longer.

## Notes on Some Decisions

I used IndoBERT instead of a multilingual model because the data is entirely in Indonesian, and a model pretrained on Indonesian text understands its vocabulary and word forms better.

I raised the number of epochs to 10 because the training set is only 80 texts. With so little data, three epochs would leave the classification head undertrained.

Looking back at this project, these are the things I would change first:

- **Collect real, longer data.** The dataset is small and the texts are short, sentence-length snippets rather than full articles. They were compiled by hand, not scraped programmatically. A scraper over several outlets would give far more data and make the evaluation meaningful.
- **Do a proper hyperparameter search.** I fixed the epochs and learning rate instead of searching over them. A small grid over learning rate and epochs, scored with cross-validation, would justify the choice.
- **Use a validation set.** The split is only train and test, so the test set doubles as the validation set. A third split, or cross-validation, would keep the test set untouched.
- **Reconsider stemming.** IndoBERT has its own subword tokenizer and was pretrained on unstemmed text. Stemming before tokenization moves the words away from what the model has seen, so it may be hurting rather than helping.
- **Evaluate the 2-stage pipeline end to end.** Feeding stage 1 predictions into stage 2 and scoring the full pipeline on one shared test set would make the comparison with the one-stage model fair.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · Sastrawi

## Context

Built as the final exam project for the Text Mining course, Data Science program, Bina Nusantara University, January 2026.

Alessandro Moreno Lawadinata
[LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/SeafoodEnthusiast)
