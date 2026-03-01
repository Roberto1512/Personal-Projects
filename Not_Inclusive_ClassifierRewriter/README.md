# FAIR ISLE

FAIR ISLE is a web application that detects non-inclusive language and suggests inclusive rewrites in **Italian** and **English**. The system combines two Transformer-based components:

- **Bias Classifier**: identifies whether a sentence contains non-inclusive wording.
- **Inclusive Rewriter (Seq2Seq)**: generates an inclusive alternative while preserving meaning and intent.

The project includes monolingual pipelines (IT/EN) and an experimental **multilingual** setup.

## Key Features

- Sentence-level analysis with a confidence score.
- Conditional rewriting: suggestions are produced only when the classifier flags non-inclusive language.
- Support for **Italian** and **English**, plus an experimental **multilingual** mode.
- Feedback collection to support continuous improvement and future retraining.
- Local model loading for fast inference inside the webapp and service.

## Project Structure

- `classificatore/`  
  Notebooks for training and experimenting with the classification models (EN/IT/MULTI).

- `riscrittore/`  
  Notebooks for training and experimenting with seq2seq rewriters (EN/IT/MULTI).

- `preprocessing/`  
  Dataset preparation notebook(s), including cleaning, normalization, and split generation.

- `data/`  
  Raw datasets and train/test splits used for classification and rewriting tasks.

- `metriche/`  
  Evaluation notebooks for classification (Accuracy/F1) and rewriting (ROUGE/BLEU/BERTScore).

- `models/` 
  Saved model artifacts (classifier and rewriter checkpoints, tokenizers, and label maps).

- `service/`  
  API-oriented backend layer.

- `webapp/`  
  Flask web application with templates and static assets (CSS/JS), plus a feedback storage folder.

## Models Overview

### English
- Classifier: DistilBERT-based fine-tuning.
- Rewriter: FLAN-T5 fine-tuning (seq2seq).

### Italian
- Classifier: BERTino-based fine-tuning.
- Rewriter: IT5 fine-tuning (seq2seq).

### Multilingual (experimental)
- Classifier: XLM-RoBERTa fine-tuning.
- Rewriter: mT5 fine-tuning (seq2seq).

## Outputs

For each sentence, FAIR ISLE produces:
- a label (**inclusive** / **not inclusive**)
- a confidence score
- an optional inclusive rewrite suggestion

## Notes

- Model artifacts can be large; repository configuration may exclude them depending on the chosen distribution strategy.
- The feedback component stores user validation and rewrite evaluations for future dataset expansion and retraining.
