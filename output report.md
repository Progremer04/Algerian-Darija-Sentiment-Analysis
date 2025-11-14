# Algerian Darija Sentiment Analysis - The Complete Pipeline

**Created By:**
- ALLICHE AMINE MOHAMED
- MIHOBI MOHAMED ABDLHAK

**Original Notebook Link:** [**▶️ View on Google Colab**](https://colab.research.google.com/drive/1x6ijtihxsrkG-ch-C_1MEpcvT4oh8P5b?usp=sharing)

---

## 1. Project Overview

This repository contains a complete, end-to-end pipeline for sentiment analysis of Algerian Darija text. It features a high-performance, hybrid deep learning model built with PyTorch that combines a **Long Short-Term Memory (LSTM)** network with a **Convolutional Neural Network (CNN)**. The entire workflow, from data aggregation and advanced preprocessing to training, evaluation, and interactive testing, is fully automated within a single, comprehensive script.

After extensive experimentation with multiple architectures, the LSTM-CNN model was chosen for its superior performance in capturing both the contextual flow and the key features of the text.

---

## 2. Environment and Hardware

-   **Platform**: Google Colaboratory
-   **Hardware**: NVIDIA GPU, specifically a **Tesla T4** as detected during runtime.
    > ```
    > 🖥️  Device: CUDA GPU: Tesla T4
    > ```

---

## 3. The LSTM-CNN Architecture

The core of this project is a hybrid neural network designed to effectively process textual data for classification.

**Architectural Flow:**

1.  **Embedding Layer**: Maps each word in the input text to a high-dimensional vector.
2.  **Bidirectional LSTM**: Processes the sequence of word vectors in both forward and backward directions to capture the full context of each word.
3.  **1D Convolutional Layer**: Scans the output of the LSTM to extract key local patterns and features (similar to n-grams).
4.  **Max Pooling & Fully Connected Layers**: Down-sample the features to produce a final classification, which is then passed through a linear layer to generate the sentiment score.

This architecture leverages the LSTM’s strength in understanding sequence and context, and the CNN’s ability to extract key local features, resulting in a powerful and robust classification model.

---

## 4. Data Sources

The model is trained on a rich, composite dataset created by merging four different public sources to ensure diversity and robustness.

| # | Dataset | URL / Source |
| :- | :--- | :--- |
| 1 | **ADArabic** | [Hugging Face Link](https://huggingface.co/datasets/Abdou/dz-sentiment-yt-comments/resolve/main/ADArabic-3labels-50016.csv) |
| 2 | **DZSentia** | [GitHub Raw Link](https://raw.githubusercontent.com/adelabdelli/DzSentiA/master/dataset.csv) |
| 3 | **Algerian Dialect Excel** | Kaggle: `mafazachabane/sentiment-algerian-operators` |
| 4 | **FASSILA** | [GitHub Repo](https://github.com/amincoding/FASSILA) |

---

## 5. Performance and Results

The model was trained for 25 epochs and achieved excellent performance on the unseen test set.

### Training History

The training and validation loss and accuracy were tracked over the 25 epochs.

![Training History](https://private-us-east-1.manuscdn.com/sessionFile/k8WETMNoWraKSKHOYTOUWZ/sandbox/GLbS3SkjWA9SPGfJlrnJAB-images_1763129855930_na1fn_L2hvbWUvdWJ1bnR1L2FsZ2VyaWFuX3NlbnRpbWVudF9wcm9qZWN0L2ltYWdlcy90cmFpbmluZ19oaXN0b3J5.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvazhXRVRNTm9XcmFLU0tIT1lUT1VXWi9zYW5kYm94L0dMYlMzU2tqV0E5U1BHZkpscm5KQUItaW1hZ2VzXzE3NjMxMjk4NTU5MzBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnNaMlZ5YVdGdVgzTmxiblJwYldWdWRGOXdjbTlxWldOMEwybHRZV2RsY3k5MGNtRnBibWx1WjE5b2FYTjBiM0o1LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Oh9fwZsHg-KZkUBkD85qgraLAYN6eg1xYCdYikC7-qbK9KoxTE5lmz3L9y33YTWHyw4mYUSTOQlvWF1OfGt-HzXbqwzlRNrBI5ZiWO6iTu55MNIYvAjEyKFle5sqjOW1QMpbzC2QLiPSDMPvRM~Rj8moVwNf6lDC~2pOLl~Qq54OdsG3ku5WpGk0eAdRbpnknIhHaCazKeCQv1C4Je1UupcB8VEjEXbz31q8yatIhe9WHVgAU-J2pm3gU7cY8IJx6SxQJzQmSzYXfESaXfULY~2hBPEhAwT4t6H4xLlSi~MPfAQIbnzs44kI4xGRME7tPP6FCNPADHnjrQnGwxhSGQ__)

### Final Test Results

The model achieved a final accuracy of **77%** on the test set.

> **Classification Report:**
> ```
>               precision    recall  f1-score   support
>
>     Negative       0.77      0.81      0.79      6766
>     Positive       0.77      0.72      0.74      5885
>
>     accuracy                           0.77     12651
>    macro avg       0.77      0.77      0.77     12651
> weighted avg       0.77      0.77      0.77     12651
> ```

### Confusion Matrix

The confusion matrix provides a detailed look at the model’s predictions vs. the actual labels.

![Confusion Matrix](https://private-us-east-1.manuscdn.com/sessionFile/k8WETMNoWraKSKHOYTOUWZ/sandbox/GLbS3SkjWA9SPGfJlrnJAB-images_1763129855931_na1fn_L2hvbWUvdWJ1bnR1L2FsZ2VyaWFuX3NlbnRpbWVudF9wcm9qZWN0L2ltYWdlcy9jb25mdXNpb25fbWF0cml4.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvazhXRVRNTm9XcmFLU0tIT1lUT1VXWi9zYW5kYm94L0dMYlMzU2tqV0E5U1BHZkpscm5KQUItaW1hZ2VzXzE3NjMxMjk4NTU5MzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnNaMlZ5YVdGdVgzTmxiblJwYldWdWRGOXdjbTlxWldOMEwybHRZV2RsY3k5amIyNW1kWE5wYjI1ZmJXRjBjbWw0LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=kZJORRjT-1kdHV0VFzSO~ioYzhODStbB2-MZKW9auvE1briSTerayYwWIyMHYnR948C39g8cNz6aH76XCXIAuSGnfIWEigezqzSqKjX97u9TnEinc0AuulJFxGVd3LAS-juhUYAtl4Th4p3SFMbKJ~A-sy8r--Utu9LUzxpk92tcxWC7dZlvqOCPVYG~Fv49~nUvj86HTY7DEow~SqBNNBkoEWXJTV~uXw5yhsd5J2BeVeGyTtnP9GBLE1~ucTFqNyZGZnbGPrNWZV83Rhq0-S0JZKE9I-6td9wylCMW6K5JwAWak9zZG0EE2rrbRrpX2zaHfNKWxZNTdqz-k1grhQ__)

---

## 6. How to Use

### Installation

All dependencies are installed by the first cell of the Google Colab notebook.

### Running the Pipeline

Simply open the [**Google Colab Notebook**](https://colab.research.google.com/drive/1x6ijtihxsrkG-ch-C_1MEpcvT4oh8P5b?usp=sharing) and run the cells in order. The script is fully automated.

### Interactive Testing

After the pipeline has finished running, you can use the `test_sentiment` function to get predictions on your own sentences. The final cells of the notebook demonstrate this with a batch test.

> **Batch Test Output:**
> ```
>                      Text Sentiment Confidence Pos_Prob Neg_Prob
> المنتج ممتاز والسعر مناسب  POSITIVE     96.02%   0.9602   0.0398
>          الجودة رديئة جدا  NEGATIVE     99.31%   0.0069   0.9931
>       خدمة العملاء ممتازة  POSITIVE     96.11%   0.9611   0.0389
>       التجربة سيئة للغاية  NEGATIVE     98.28%   0.0172   0.9828
>     أنصح بهذا المكان بشدة  POSITIVE     78.69%   0.7869   0.2131
> ```

![Batch Test Visualization](https://private-us-east-1.manuscdn.com/sessionFile/k8WETMNoWraKSKHOYTOUWZ/sandbox/GLbS3SkjWA9SPGfJlrnJAB-images_1763129855932_na1fn_L2hvbWUvdWJ1bnR1L2FsZ2VyaWFuX3NlbnRpbWVudF9wcm9qZWN0L2ltYWdlcy9iYXRjaF90ZXN0X3Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvazhXRVRNTm9XcmFLU0tIT1lUT1VXWi9zYW5kYm94L0dMYlMzU2tqV0E5U1BHZkpscm5KQUItaW1hZ2VzXzE3NjMxMjk4NTU5MzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnNaMlZ5YVdGdVgzTmxiblJwYldWdWRGOXdjbTlxWldOMEwybHRZV2RsY3k5aVlYUmphRjkwWlhOMFgzSmxjM1ZzZEhNLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=nR-fYrYsrUYX2PC8NeiUK2prr916FZdcIJ00Clgd61Q-P2Bn01XYykx8CyCLVAsS7gc~SDzpZqDtLmVEsCz3wg945wObbmcEr5bd4mIzVN9MdsvZ0hWP9LhGOv486W3lL8LRDZSPhtya4gUBhhrv4Kog-UvE-JeDw9wcaizQ~D049v6AQuvzRdKYVgenxCdGdOohhE3ftIhyBdXU7nq8KRK1xA5YBW7512bHwpke5-bZlxChogBQ5JLuucTZamVrxNjrNKVXuv4QbFyB-Mg4uykuF4s9J-RZEtLqXCwZlZy~Gn9~XYXJKVPerl2H0ERBdfMSQSH3pgBfcH3MdFxLlw__)

---

## 7. Dependencies

| Library | Purpose & Role in the Project |
| :--- | :--- |
| `numpy` | Numerical computation. |
| `pandas` | Data manipulation and loading. |
| `scikit-learn` | Data splitting and evaluation metrics. |
| `matplotlib` & `seaborn` | Data visualization. |
| `torch` | The core deep learning framework. |
| `pyarabic` | Arabic text normalization. |
| `arabic-reshaper`, `python-bidi` | Rendering Arabic text in plots. |
| `nltk` & `arabic-stopwords` | Stopword removal. |
| `kagglehub` | Downloading data from Kaggle. |
| `wordcloud` | Generating word cloud visualizations. |
| `gdown` | Downloading the Arabic font from Google Drive. |
