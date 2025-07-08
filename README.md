# 📝 Transformer-Based Text Summarization of U.S. Congressional Bills

This project fine-tunes the `csebuetnlp/mT5_m2o_english_crossSum` multilingual transformer model on the `FiscalNote/billsum` dataset to automatically generate concise and meaningful summaries of U.S. Congressional bills.

---

## 🚀 Overview

This notebook demonstrates how to:
- Load and preprocess the `billsum` dataset
- Fine-tune the mT5 model for summarization using Hugging Face Transformers
- Evaluate the model using ROUGE scores
- Save and push the trained model to the Hugging Face Model Hub

---

## 📂 Files

| File                                  |            Description                                        |
|---------------------------------------|----------------------------------------------------------------|
| `text_summarization_finetuning.ipynb` | Complete Colab notebook for model training, evaluation, and deployment |
| `README.md`                           | Project overview and usage instructions |

---

## 📊 Dataset

- **Name:** [FiscalNote/billsum](https://huggingface.co/datasets/billsum)
- **Description:** Summarization dataset containing U.S. Congressional and California state bills with human-written summaries.
- **Task:** Generate summaries for full legislative texts.

---

## 🔧 Tools & Technologies

- Python
- Hugging Face Transformers
- Datasets Library
- PyTorch
- Google Colab
- ROUGE Metrics

---

## 🛠️ Model Details

- **Base model:** `csebuetnlp/mT5_m2o_english_crossSum`
- **Task:** Abstractive summarization
- **Training:** Conducted on Google Colab using Hugging Face's `Trainer` API
- **Evaluation Metric:** ROUGE-1, ROUGE-2, ROUGE-L

---

## 🔗 Model on Hugging Face

📦 You can view and try the trained model here:  
👉 [https://huggingface.co/RamshaAnwar/summarization_model_trained_on_reduced_dataset](https://huggingface.co/RamshaAnwar/summarization_model_trained_on_reduced_dataset))

---

## 🧠 Example Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("RamshaAnwar/summarization_model_trained_on_reduced_dataset")
model = AutoModelForSeq2SeqLM.from_pretrained("RamshaAnwar/summarization_model_trained_on_reduced_dataset")

text = "The bill aims to revise the regulation and funding of public health programs..."
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)

print("Summary:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))

