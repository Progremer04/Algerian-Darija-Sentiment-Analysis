# The Ultimate Technical Guide to the Algerian Darija Sentiment Analysis Pipeline

## 1. Project Philosophy and Goals

This project was born from a desire to create a truly comprehensive, end-to-end solution for sentiment analysis on a specific, nuanced dialect: Algerian Darija. The goal was not just to build a model, but to engineer a complete, reproducible, and easy-to-understand pipeline. This document serves as an exhaustive guide to every single component of that pipeline, from the initial setup to the final interactive model. We will dissect every line of code, every architectural choice, and every data source to provide a crystal-clear understanding of how this system works.

---

## 2. Section 1: Installation and Dependencies

The foundation of any Python project is its dependencies. The script ensures a smooth setup by installing all necessary libraries at the very beginning.

**Command:**
```bash
!pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision torchaudio --quiet
!pip install pyarabic arabic-reshaper python-bidi nltk arabic-stopwords kagglehub wordcloud gdown --quiet
```

**Dependency Breakdown:**

| Library | Purpose & Role in the Project |
| :--- | :--- |
| `numpy` | The fundamental package for numerical computation. Used for handling arrays and mathematical operations. |
| `pandas` | The primary tool for data manipulation and analysis. Used to load, clean, and structure all datasets in DataFrames. |
| `scikit-learn` | A key machine learning library. Used here for splitting data (`train_test_split`) and for evaluation metrics (`classification_report`, `confusion_matrix`, `accuracy_score`). |
| `matplotlib` | The primary plotting library. Used to create all visualizations, including training history and confusion matrices. |
| `seaborn` | Built on top of Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics. |
| `torch` | The core deep learning framework. Used to build, train, and run the neural network. |
| `torchvision`, `torchaudio` | Companion libraries to PyTorch for vision and audio, installed here as part of a standard PyTorch setup. |
| `pyarabic` | An essential library for Arabic text processing. Used for normalization tasks like stripping diacritics and unifying character forms. |
| `arabic-reshaper`, `python-bidi` | Used to correctly render Arabic text in visualizations, ensuring proper character shaping and right-to-left display. |
| `nltk` | The Natural Language Toolkit. Used specifically to access its corpus of standard Arabic stopwords. |
| `arabic-stopwords` | Another source for a more comprehensive list of Arabic stopwords. |
| `kagglehub` | The official Kaggle API client. Used to programmatically download datasets directly from Kaggle. |
| `wordcloud` | Used to generate word cloud visualizations, providing a quick visual summary of word frequencies. |
| `gdown` | A utility to download large files from Google Drive. Used here to fetch the Arabic font needed for visualizations. |

---

## 3. Section 2: Environment and Configuration

With dependencies installed, the script configures the runtime environment for consistency and optimal performance.

### 3.1. Arabic Font Setup for Visualizations

To ensure Arabic text is rendered correctly in plots (a common challenge), the script downloads and registers a specific Arabic font.

-   **Font URL**: `https://drive.google.com/uc?id=1XWyHLXSapRLoxveyaVsbaWWpe5OAZ0fz`
-   **Logic**: It checks if the font file (`arabic_font.ttf`) already exists. If not, it uses `gdown` to download it from the Google Drive link. It then uses `matplotlib.font_manager` to add the font to Matplotlib’s list of available fonts. This allows plots like word clouds to display Arabic characters beautifully.

### 3.2. Environment Utility Functions

These functions are helpers that provide information about the environment and ensure reproducibility.

-   **`is_colab()`**: Returns `True` if the script is running in Google Colab. This can be used to enable Colab-specific features if needed.
-   **`get_device()`**: This is critical for performance. It checks if `torch.cuda.is_available()` and returns a `torch.device("cuda")` object if a GPU is found, otherwise it falls back to `torch.device("cpu")`. All tensors and the model itself will be moved to this device to leverage GPU acceleration.
-   **`device_info()`**: A simple function that returns the name of the GPU being used, which is useful for logging and debugging.
-   **`set_seed(seed=42)`**: This function is the cornerstone of reproducibility. By setting the random seed for `random`, `numpy`, and `torch` (for both CPU and GPU), it guarantees that any operation with a random element—like weight initialization, dropout, and data shuffling—will produce the exact same result every time the script is run.

---

## 4. Section 3: Data Sourcing and Integration

The model is trained on a rich, aggregated dataset compiled from four different sources. This diversity is key to building a robust and generalizable model.

**Dataset Links and Descriptions:**

| # | Dataset | URL / Source | Description |
| :- | :--- | :--- | :--- |
| 1 | **ADArabic** | `https://huggingface.co/datasets/Abdou/dz-sentiment-yt-comments/resolve/main/ADArabic-3labels-50016.csv` | A dataset of 50,016 YouTube comments in Algerian Darija, labeled as positive, negative, or neutral. |
| 2 | **DZSentia** | `https://raw.githubusercontent.com/adelabdelli/DzSentiA/master/dataset.csv` | A smaller but valuable dataset specifically for Algerian sentiment analysis. |
| 3 | **Algerian Dialect Excel** | Kaggle: `mafazachabane/sentiment-algerian-operators` | An Excel dataset containing sentiment-labeled text related to Algerian mobile operators. |
| 4 | **FASSILA** | GitHub: `amincoding/FASSILA` | A larger dataset originally for fake news detection. The script uses its text and labels, adapting it for sentiment analysis. It loads the test, train, and validation sets. |

**Data Loading Functions:**

-   **`store_data(url)`**: A clever helper that automatically converts standard GitHub URLs (e.g., `github.com/.../blob/...`) into raw content URLs (`raw.githubusercontent.com/.../...`) so `pandas.read_csv` can read them directly.
-   **`load_and_clean_excel(filepath)`**: This function is custom-built for the third dataset. It reads the `.xlsx` file, finds the column that incorrectly merges the label and text, and splits it into two proper columns.
-   **`load_local_datasets()` & `load_fassila_dataset()`**: These orchestrate the loading of the four datasets using the helper functions.
-   **`merge_all_datasets(...)`**: This function is the final step in data integration. It takes all the loaded DataFrames, standardizes the column names and label formats (mapping strings like "Positive" to integers like `1`), and concatenates them into a single, unified DataFrame.

---

## 5. Section 4: The Preprocessing Pipeline

This is one of the most critical stages. Raw text is noisy and must be meticulously cleaned before being fed to a neural network.

**Step 1: Regular Expression-Based Cleaning (`clean_text`)**

Four pre-compiled regex patterns are used to systematically strip noise:

-   `_URL_RE = re.compile(r'https?://\S+|www\.\S+')`: Finds and removes all URLs.
-   `_EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)`: Finds and removes all Unicode emoji characters.
-   `_PUNCT_RE = re.compile(r'[^\w\s\u0600-\u06FF]')`: A subtractive pattern. It removes anything that is *not* a word character, a whitespace character, or a character within the standard Arabic Unicode range.
-   `_LATIN_DIGIT_RE = re.compile(r'[A-Za-z0-9_]+')`: Removes any leftover Latin characters and digits.

**Step 2: Arabic-Specific Normalization (`normalize_arabic`)**

This function uses the `pyarabic` library to perform essential normalization:

-   `araby.strip_tashkeel(text)`: Removes all diacritics (e.g., fatha, damma, kasra).
-   `araby.strip_tatweel(text)`: Removes the `ـ` character used to elongate words.
-   `t.replace(...)`: Manually unifies different forms of the same letter (e.g., `أ`, `إ`, `آ` all become `ا`; `ة` becomes `ه`).

**Step 3: Comprehensive Stopword Removal (`make_stopwords`)**

A large and effective set of stopwords is created by combining three sources:

1.  **NLTK**: `stopwords.words('arabic')`
2.  **arabic-stopwords library**: `stp.stopwords_list()`
3.  **Custom List**: A manually curated set of common Algerian Darija words that don’t carry sentiment (e.g., `تاع`, `بصح`, `راك`).

**Step 4: The Main `preprocess_dataframe` Function**

This function orchestrates the entire pipeline:

1.  Drops any rows with missing text or labels.
2.  Filters the DataFrame to keep only positive (`1`) and negative (`0`) labels, discarding neutral ones for this binary classification task.
3.  Applies the `clean_text` function to every entry in the `text` column.
4.  Applies a lambda function to split each text into words and remove any word present in the stopword set.
5.  Drops any rows where the text became empty after cleaning and removes duplicate entries.

---

## 6. Section 5: Data Preparation for PyTorch

Once the text is clean, it needs to be converted into a numerical format that PyTorch can understand.

-   **`build_vocab(texts, ...)`**: This function builds a word-to-integer mapping. It counts all word frequencies and keeps only the words that appear at least `min_freq` times, up to a maximum vocabulary size of `max_size`. It reserves `0` for a padding token (`<PAD>`) and `1` for out-of-vocabulary words (`<OOV>`).
-   **`encode_text(text, vocab, max_len)`**: This function takes a sentence, splits it into words, and converts each word to its integer ID using the vocabulary. If a word is not in the vocabulary, it is assigned the `<OOV>` ID (`1`). The resulting sequence is then either truncated or padded with `0`s to ensure it has a fixed length of `max_len`.
-   **`TextDataset(Dataset)`**: This is a standard PyTorch `Dataset` class. It’s a wrapper that connects the `encode_text` function to the data. When the `DataLoader` requests an item at a certain index, this class’s `__getitem__` method fetches the corresponding text and label, encodes the text into a tensor of IDs, and returns the pair.

---

## 7. Section 6: The Hybrid LSTM-CNN Architecture

This is the brain of the operation. The `LSTMCNN` class defines a hybrid neural network designed to capture both sequential context and local, salient features from the text.

**Class Definition:**
```python
class LSTMCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, lstm_hidden=128, conv_filters=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(2*lstm_hidden, conv_filters, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_filters, 1)
        self.drop = nn.Dropout(0.3)
```

**Layer-by-Layer Architectural Breakdown:**

1.  **`self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)`**
    *   **Purpose**: To convert the input sequence of word IDs into dense vector representations.
    *   **`vocab_size`**: The total number of unique words in our vocabulary.
    *   **`embed_dim=200`**: The dimensionality of the word vectors. Each word will be represented by a 200-dimensional vector. This is a learnable parameter.
    *   **`padding_idx=0`**: This tells the layer to ignore the padding token (`<PAD>`, which has ID 0) during training. Its vector will not be updated.

2.  **`self.lstm = nn.LSTM(embed_dim, lstm_hidden, bidirectional=True, batch_first=True)`**
    *   **Purpose**: To process the sequence of word vectors and capture contextual information.
    *   **`input_size=embed_dim`**: The size of the input vectors (200 from the embedding layer).
    *   **`hidden_size=lstm_hidden=128`**: The number of features in the hidden state. Each LSTM cell will have a 128-dimensional hidden state.
    *   **`bidirectional=True`**: This is a key feature. It creates two LSTMs: one that processes the sequence from left-to-right and another from right-to-left. The outputs are concatenated, providing a richer context for each word.
    *   **`batch_first=True`**: This specifies that the input tensors will have the batch dimension first (`[batch_size, sequence_length, feature_dim]`), which is a more intuitive standard.

3.  **`self.conv = nn.Conv1d(2*lstm_hidden, conv_filters, 3, padding=1)`**
    *   **Purpose**: To act as a feature detector, scanning the output of the LSTM for important local patterns (like n-grams).
    *   **`in_channels=2*lstm_hidden`**: The input channel size is `2 * 128 = 256`. The `2` comes from the bidirectional LSTM (one forward hidden state, one backward).
    *   **`out_channels=conv_filters=256`**: The number of filters the CNN will learn. It will produce 256 different feature maps.
    *   **`kernel_size=3`**: The size of the scanning window. The CNN will look at patterns of 3 consecutive words at a time.
    *   **`padding=1`**: This adds padding to the input, ensuring the output sequence has the same length as the input.

4.  **`self.relu = nn.ReLU()`**: An activation function that introduces non-linearity, allowing the model to learn more complex patterns.

5.  **`self.pool = nn.AdaptiveMaxPool1d(1)`**: 
    *   **Purpose**: To down-sample the feature maps from the CNN and produce a single, fixed-size output vector.
    *   **`output_size=1`**: It will take the feature maps (of variable length) and reduce each one to a single value by taking the maximum activation. This is a powerful way to capture the single most important feature detected by each filter, regardless of where it appeared in the sentence.

6.  **`self.drop = nn.Dropout(0.3)`**: 
    *   **Purpose**: A regularization technique to prevent overfitting. During training, it will randomly set 30% of the activations from the previous layer to zero, forcing the network to learn more robust and distributed representations.

7.  **`self.fc = nn.Linear(conv_filters, 1)`**: 
    *   **Purpose**: The final classification layer. It takes the fixed-size vector from the pooling layer and maps it to a single output logit.
    *   **`in_features=conv_filters=256`**: The size of the input vector.
    *   **`out_features=1`**: It outputs a single number, which will represent the sentiment score.

**The `forward(self, x)` Method:** This method defines the data flow through the layers in the exact sequence described above.

---

## 8. Section 7: The Training and Evaluation Engine

These functions define the core training loop and the process for evaluating the model.

-   **`train_epoch(...)`**: This function performs one full pass over the training data.
    1.  `model.train()`: Puts the model in training mode (enables dropout, etc.).
    2.  It iterates through the `DataLoader`.
    3.  `opt.zero_grad()`: Clears old gradients before calculating new ones.
    4.  `out = model(X)`: Performs a forward pass to get the model's predictions.
    5.  `loss = loss_fn(out, y)`: Calculates the loss. The script uses `nn.BCEWithLogitsLoss`, which combines a Sigmoid layer and Binary Cross-Entropy loss in one class. It’s numerically stable and expects raw logits from the model.
    6.  `loss.backward()`: Computes the gradients of the loss with respect to the model's parameters.
    7.  `opt.step()`: Updates the model's weights using the optimizer (the script uses `AdamW`, a robust and popular choice).
-   **`eval_epoch(...)`**: This function evaluates the model on a given dataset (validation or test).
    1.  `model.eval()`: Puts the model in evaluation mode (disables dropout, etc.).
    2.  `with torch.no_grad()`: Disables gradient calculation, which speeds up inference and reduces memory usage.
    3.  It iterates through the data, calculates the loss and accuracy, and returns the predictions and probabilities.

---

## 9. Section 8: The Master Pipeline (`run_full_pipeline`)

This is the main orchestrator function that executes the entire 16-step workflow from start to finish. It takes hyperparameters like `epochs`, `batch_size`, and `lr` (learning rate) as arguments.

**The 16 Steps:**

1.  **Setup**: Sets the random seed and gets the device.
2.  **Data Loading**: Calls `load_local_datasets()` and `load_fassila_dataset()`.
3.  **Data Merging**: Calls `merge_all_datasets()`.
4.  **Initial Analysis**: Prints the shape and class distribution of the raw, combined data.
5.  **Word Cloud (Before)**: Generates and plots a word cloud of the raw text to visualize the most common words before cleaning.
6.  **Preprocessing**: Creates the stopword list and calls `preprocess_dataframe` to clean the entire dataset.
7.  **Word Cloud (After)**: Generates a new word cloud on the cleaned text to visually confirm the effect of preprocessing.
8.  **Vocabulary Creation**: Calls `build_vocab` on the cleaned texts.
9.  **Data Splitting**: Splits the data into training (80%), validation (10%), and test (10%) sets.
10. **Dataset & DataLoader Creation**: Instantiates the `TextDataset` and `DataLoader` for each of the three splits.
11. **Model Initialization**: Creates an instance of the `LSTMCNN` model, the `AdamW` optimizer, and the `BCEWithLogitsLoss` function.
12. **Training Loop**: This is the core training block. It iterates for the specified number of `epochs`, calling `train_epoch` and `eval_epoch` in each loop. It keeps track of the best validation accuracy and saves the model's state dict (`torch.save(model.state_dict(), 
best_model.pt")`) whenever a new best is achieved.
13. **Plotting Results**: After the training loop, it calls `plot_history` to generate and display plots of the training/validation loss and accuracy over epochs.
14. **Final Evaluation**: It loads the weights of the best-performing model (`best_model.pt`) and calls `eval_epoch` on the test set. It then prints a detailed `classification_report` from scikit-learn.
15. **Confusion Matrix**: It computes and plots a confusion matrix using `seaborn.heatmap` to visualize the model's performance on the test set.
16. **Artifact Saving**: It saves the final vocabulary (`vocab.json`) and metadata (`meta.json`) for later use in inference.

---

## 10. Section 9: Interactive Model Testing

After the pipeline has run and the model is trained, the script provides two functions for interactive testing, allowing a user to directly query the model.

-   **`test_sentiment(...)`**: This function is designed for single, ad-hoc predictions.
    1.  It takes a raw text string as input.
    2.  It applies the *exact same* cleaning and preprocessing steps (`clean_text`, stopword removal) that were used during training. This is crucial for correct predictions.
    3.  It encodes the cleaned text using the saved vocabulary and `max_len`.
    4.  It feeds the resulting tensor to the model to get a logit.
    5.  It applies the sigmoid function (`torch.sigmoid`) to the logit to get a probability between 0 and 1.
    6.  It determines the final prediction ("POSITIVE" or "NEGATIVE") and prints a user-friendly summary including the sentiment, confidence score, and the raw probabilities.

-   **`batch_test(...)`**: This function is similar but designed to process a list of texts at once.
    1.  It iterates through a list of input texts.
    2.  For each text, it performs the same preprocessing and prediction steps as `test_sentiment`.
    3.  It stores the results for each text in a list of dictionaries.
    4.  Finally, it converts this list into a pandas DataFrame for a clean, tabular output and generates a bar chart visualizing the confidence scores for the batch.

---

## 11. Conclusion

This pipeline is a testament to the power of a well-structured, end-to-end approach in machine learning. Every component, from the multi-source data aggregation and meticulous preprocessing to the hybrid model architecture and detailed evaluation, is designed to be robust, reproducible, and effective. By documenting every dependency, every function, and every line of code, this guide provides a complete and transparent view into the inner workings of a production-ready sentiment analysis system for a complex, low-resource dialect.
