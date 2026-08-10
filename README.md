# Fake News Detection

🔗 **Live demo:** https://fake-news-detection-ilvo3jbhhzabwyxyw8ufnv.streamlit.app/

A machine learning project for classifying news articles as **Fake** or **Real** using Natural Language Processing (NLP).

## Overview

This project explores the use of machine learning and Natural Language Processing to classify news articles as Fake or Real.

The objective is not only to build a high-performing classifier, but also to investigate **what the model actually learns from the data** and how reliably its performance can be interpreted.

The project follows an end-to-end machine learning workflow, including exploratory data analysis, text preprocessing, TF-IDF feature extraction, model comparison, hyperparameter tuning, evaluation on a held-out test set, and error analysis.

Several classification algorithms are evaluated, with a TF-IDF + Linear Support Vector Machine (SVM) pipeline ultimately selected as the final model.

The final model achieves approximately **98.7% accuracy and 98.7% macro F1-score** on the held-out test set. However, further analysis shows that part of this strong performance may be driven by source-specific and editorial patterns present in the dataset rather than by a general ability to determine whether a news claim is factually true.

For this reason, the project places particular emphasis on **model interpretation, error analysis, and limitations**, rather than treating classification accuracy alone as evidence of reliable fake-news detection.

## Dataset

The project uses the **ISOT Fake News Dataset**, which contains news articles labeled as either Fake or Real.

The original dataset is provided as two separate collections:

- `Fake.csv`: articles labeled as fake news
- `True.csv`: articles labeled as real news

The two datasets are combined into a single dataset and assigned a binary target label before analysis and modelling.

Each article contains textual and contextual information such as its title, article text, subject, and publication date. For the classification task developed in this project, the model is trained on the article text.

### Class Distribution

Before modelling, the class distribution was examined to identify potential imbalance between Fake and Real articles.

The dataset contains slightly more Real articles than Fake articles, but the two classes remain relatively balanced. After the train-test split, approximately **55% of observations belong to the Real class and 45% to the Fake class** in both subsets.

The similar class distributions between the training and test sets ensure that the evaluation set remains representative of the class proportions used during training.

### Dataset Considerations

An important limitation of the dataset is that the two classes do not necessarily originate from the same sources or follow the same editorial conventions.

As a result, differences between Fake and Real articles may include not only differences related to misinformation, but also differences in writing style, vocabulary, formatting, attribution patterns, and source-specific conventions.

This distinction becomes particularly important when interpreting the model's high classification performance and is investigated further in the **Error Analysis** section.

## Methodology

The modelling workflow was designed to progressively move from understanding the dataset to building and evaluating a reproducible text-classification pipeline.

### 1. Exploratory Data Analysis

Exploratory analysis was performed before modelling to understand the structure, quality, and linguistic characteristics of the dataset.

The analysis included:

- class distribution and dataset structure;
- missing and duplicate observations;
- article and title length distributions;
- subject distributions across Fake and Real articles;
- word-frequency analysis;
- unigram and bigram exploration;
- identification of recurring publishing and scraping artifacts.

The EDA revealed noticeable lexical and editorial differences between the two classes. These observations informed the preprocessing strategy and later became particularly relevant when interpreting the behaviour of the final classifier.

### 2. Text Preprocessing

Text preprocessing was intentionally kept relatively conservative in order to remove obvious noise without unnecessarily discarding potentially useful linguistic information.

The preprocessing workflow included:

- removal of URLs;
- removal of the explicit `Reuters` source marker;
- removal of recurring scraping and publishing artifacts identified during EDA, such as `featured image`, `getty images`, `pic twitter`, `twitter com`, and `screen capture`;
- normalization of the cleaned text;
- removal of observations whose cleaned text became empty;
- duplicate handling after text cleaning.

The explicit `Reuters` marker was removed to prevent the classifier from relying on a direct source identifier when distinguishing between classes.

No aggressive stemming or lemmatization was applied. Stop words were also retained for the initial modelling approach. This decision preserved stylistic information that could potentially contribute to classification.

Importantly, all preprocessing decisions were made before final test evaluation.

Reusable cleaning logic was moved to `src/preprocessing.py` so that the same transformation can be applied consistently during both experimentation and inference.


### 3. Train-Test Split

The cleaned dataset was separated into training and test sets using a stratified split.

Stratification preserves approximately the same Fake/Real class distribution in both subsets.

The held-out test set was isolated before model selection and hyperparameter tuning and was not used to choose between models.

Model comparison and optimization were instead performed using cross-validation on the training data.


### 4. TF-IDF Feature Extraction

News articles were converted from raw text into numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

TF-IDF assigns greater importance to terms that are informative within a document while reducing the influence of terms that occur very frequently across the corpus.

A minimum document frequency threshold of:

`min_df = 5`

was used to exclude extremely rare terms.

Rather than fitting the vectorizer independently before each modelling step, TF-IDF was integrated into a Scikit-learn `Pipeline` together with the classifier.


### 5. Model Comparison

Five classification algorithms were evaluated:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes
- Decision Tree
- Random Forest

Each model was combined with the same TF-IDF representation and evaluated using **5-fold cross-validation** on the training set.

The following metrics were monitored:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score

Macro F1-score was used as the primary comparison metric because it gives equal importance to both Fake and Real articles rather than allowing the slightly larger class to have more influence on the final score.

The cross-validation results were:

| Model                   |   Accuracy |  Precision |     Recall |   F1-score |   Macro F1 |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Linear SVM**          | **0.9865** | **0.9842** | **0.9914** | **0.9878** | **0.9864** |
| Logistic Regression     |     0.9792 |     0.9757 |     0.9867 |     0.9811 |     0.9789 |
| Random Forest           |     0.9694 |     0.9659 |     0.9788 |     0.9723 |     0.9690 |
| Multinomial Naive Bayes |     0.9350 |     0.9353 |     0.9471 |     0.9412 |     0.9343 |
| Decision Tree           |     0.9300 |     0.9314 |     0.9418 |     0.9366 |     0.9292 |

Linear SVM achieved the strongest overall cross-validation performance and was therefore selected for further optimization.


### 6. Hyperparameter Tuning

The regularization parameter `C` of the Linear SVM was optimized using `GridSearchCV` with 5-fold cross-validation.

The following values were evaluated:

|     C | Mean CV Macro F1 | Standard Deviation |  Rank |
| ----: | ---------------: | -----------------: | ----: |
|  0.01 |           0.9614 |             0.0017 |     4 |
|   0.1 |           0.9817 |             0.0009 |     3 |
| **1** |       **0.9864** |         **0.0010** | **1** |
|    10 |           0.9862 |             0.0011 |     2 |

Performance improved substantially as `C` increased from `0.01` to `1`, suggesting that very strong regularization constrained the model too heavily.

Increasing `C` further to `10` produced no meaningful improvement.

`C = 1` achieved the highest mean cross-validation macro F1-score and was retained for the final model.

The difference between `C = 1` and `C = 10` is extremely small, so the result should not be interpreted as evidence that `C = 1` is substantially superior. It simply achieved the best observed cross-validation score among the tested values.

## Results

After model selection and hyperparameter tuning were completed using only the training data, the final TF-IDF + Linear SVM pipeline was evaluated once on the held-out test set.

### Final Test Performance

The final model achieved:

| Metric             |      Score |
| ------------------ | ---------: |
| **Accuracy**       | **98.69%** |
| **Macro F1-score** | **98.68%** |

Performance was also highly balanced between the two classes:

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| Fake  |      0.99 |   0.98 |     0.99 |   3,480 |
| Real  |      0.98 |   0.99 |     0.99 |   4,239 |

The final test set contained **7,719 articles**.

The test macro F1-score of approximately **98.68%** is extremely close to the cross-validation estimate of approximately **98.64%**, indicating consistent performance between cross-validation and the held-out test set.


### Confusion Matrix

The final model produced the following classification outcomes:

|                 | Predicted Fake | Predicted Real |
| --------------- | -------------: | -------------: |
| **Actual Fake** |      **3,413** |         **67** |
| **Actual Real** |         **34** |      **4,205** |

The classifier correctly predicted:

- 3,413 Fake articles;
- 4,205 Real articles.

It misclassified:

- **67 Fake articles as Real**;
- **34 Real articles as Fake**.

Overall, **7,618 of 7,719 test articles were classified correctly**, leaving 101 misclassifications.

Although these results indicate very strong in-dataset classification performance, aggregate metrics alone do not explain what the classifier has actually learned or why the remaining errors occur.

For this reason, the predictions were examined in greater detail through error and feature analysis.

## Error Analysis

High classification accuracy does not necessarily imply that a model has learned the intended underlying concept.

For a fake-news classifier in particular, strong performance may result from lexical, stylistic, editorial, or source-specific differences between the classes rather than from an ability to assess factual truth.

To better understand the behaviour of the final Linear SVM, the 101 misclassified test articles were therefore examined in greater detail.


### Where Does the Model Make Mistakes?

The 101 errors consist of:

- **67 Fake → Real** misclassifications;
- **34 Real → Fake** misclassifications.

Manual inspection revealed that some Fake articles classified as Real strongly resemble conventional news reporting. They contain long factual passages, quotations, institutional vocabulary, attribution language, and references to established media organizations.

Conversely, several Real articles classified as Fake have structures that differ from conventional agency-style reporting, including long speeches, letters, extended quotations, or more narrative forms of writing.

This suggests that article structure and writing style influence the classifier's decisions.


### Are Errors Close to the Decision Boundary?

Linear SVM provides a signed decision score indicating on which side of the classification boundary an observation lies.

In this project:

- negative scores correspond to the Fake side;
- positive scores correspond to the Real side;
- values closer to zero lie closer to the decision boundary.

The decision score is **not a probability**.

Analysis of the absolute decision scores of the 101 misclassified articles showed:

| Absolute decision score | Errors | Share of all errors |
| ----------------------- | -----: | ------------------: |
| < 0.10                  |     27 |               26.7% |
| < 0.25                  |     59 |               58.4% |
| < 0.50                  |     85 |               84.2% |
| < 1.00                  |     95 |               94.1% |

Most errors therefore occur relatively close to the model's decision boundary.

Only **6 of the 101 errors** have an absolute decision score greater than or equal to 1, indicating that strongly incorrect decisions are comparatively uncommon.


### What Features Drive the Predictions?

Because Linear SVM is a linear classifier, the coefficients associated with individual TF-IDF features can be inspected.

Large negative coefficients push the decision toward **Fake**, while large positive coefficients push it toward **Real**.

Some of the strongest features associated with Fake included:

| Feature | Coefficient |
| ------- | ----------: |
| `via`   |       -9.49 |
| `read`  |       -4.80 |
| `gop`   |       -3.29 |
| `mr`    |       -3.18 |
| `this`  |       -2.97 |
| `us`    |       -2.96 |
| `sen`   |       -2.86 |
| `com`   |       -2.71 |
| `just`  |       -2.70 |
| `watch` |       -2.49 |

Other strongly negative features included `pic`, `flickr`, `wfb`, `hillary`, and `breitbart`.

Several of these terms appear related to publishing conventions, web content, attribution, or particular media sources rather than directly to factual truth.


Features most strongly associated with Real included:

| Feature      | Coefficient |
| ------------ | ----------: |
| `said`       |       +7.87 |
| `washington` |       +4.39 |
| `on`         |       +4.05 |
| `nov`        |       +3.06 |
| `wednesday`  |       +2.70 |
| `thursday`   |       +2.47 |
| `republican` |       +2.42 |
| `tuesday`    |       +2.32 |
| `told`       |       +2.16 |
| `comment`    |       +1.95 |

Additional positive features included `reporters`, `friday`, `EDT`, `EST`, `london`, `saying`, and `berlin`.

Many of these terms are characteristic of conventional news-agency writing:

- attribution verbs such as `said` and `told`;
- references to `reporters`;
- geographical datelines such as `washington`, `london`, and `berlin`;
- weekdays and time-zone markers such as `EDT` and `EST`.


### What Did the Model Actually Learn?

The feature analysis reveals an important limitation.

During preprocessing, the explicit `Reuters` source marker was removed to prevent the classifier from using a trivial direct source identifier.

However, removing the source name does not remove the **linguistic fingerprint of the source**.

The classifier can still exploit patterns such as:

- attribution conventions;
- dateline vocabulary;
- weekday references;
- formatting conventions;
- source-related terminology;
- characteristic writing styles.

The model therefore appears to learn a combination of content-related information and **indirect source/editorial signals**.

This provides an important explanation for the very high test performance.

The model is highly effective at distinguishing Fake and Real articles **within the distribution represented by the ISOT dataset**, but this should not be interpreted as evidence that it can determine the factual truth of arbitrary news articles with 98.7% accuracy.

In other words:

> **High in-distribution classification performance is not equivalent to general-purpose fact-checking ability.**

This distinction is central to the interpretation of the project.

## Interactive Demo

A lightweight **Streamlit application** is included to demonstrate how the trained model can be used on new text.

The application takes a news article as input and returns:

- the predicted class: **Fake** or **Real**;
- the Linear SVM decision score.

The application uses the same reusable inference pipeline as the rest of the project:

```text
Raw text
    ↓
Text preprocessing
    ↓
TF-IDF transformation
    ↓
Linear SVM
    ↓
Fake / Real prediction
```

The TF-IDF vectorizer and Linear SVM classifier are loaded from the serialized pipeline stored in:

```text
models/linear_svm_tfidf.joblib
```

The decision score represents the signed distance from the model's decision boundary:

- negative values correspond to the Fake side;
- positive values correspond to the Real side;
- values farther from zero lie farther from the decision boundary.

The decision score is **not a probability** and should not be interpreted as the probability that an article is true or false.

> **Important:** The application is a demonstration of the trained classifier, not a fact-checking tool. Predictions reflect patterns learned from the ISOT dataset and do not constitute factual verification.


## Installation

### 1. Clone the repository

```bash
git clone git@github.com:nouraastafofana-droid/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project was developed using Python 3.10 and the main dependencies include:

- NumPy
- pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter
- Streamlit


## Usage

### Run the Streamlit application

From the project root:

```bash
streamlit run app.py
```

The application will start locally and provide a text area where a news article can be submitted for classification.


### Run inference from Python

The prediction function can also be used directly:

```python
from src.predict import predict_news

result = predict_news(
    "The president announced a new economic policy during a press conference."
)

print(result)
```

The function returns:

```python
{
    "prediction": "Fake",
    "decision_score": -0.33
}
```

The value above is only an example of the output format. Actual predictions depend on the submitted text.


### Run inference from the terminal

A simple command-line interface is also available:

```bash
python -m src.predict
```

The program prompts for a news article and displays the model prediction and decision score.


## Reproducibility

The final trained pipeline is serialized using Joblib and contains both:

- the fitted TF-IDF vectorizer;
- the fitted Linear SVM classifier.

This ensures that the exact vocabulary, IDF weights, and classifier parameters learned during training can be reused during inference.

The model was serialized using:

```text
scikit-learn==1.7.2
joblib==1.5.3
```

The corresponding versions are pinned in `requirements.txt` to improve reproducibility.

The reusable text-cleaning logic is stored separately in `src/preprocessing.py` and is applied before inference through `src/predict.py`.


## Limitations

Despite its strong test performance, this project has several important limitations.

### Dataset-specific performance

The reported **98.69% accuracy** and **98.68% macro F1-score** measure performance on a held-out portion of the ISOT dataset.

They should not be interpreted as expected performance on arbitrary news articles from different publishers, time periods, countries, writing styles, or domains.


### Source and editorial signals

Feature analysis indicates that the classifier uses patterns associated with writing and publishing conventions.

Examples include:

- attribution language;
- geographical datelines;
- weekday and time-zone references;
- source-related vocabulary;
- web and publishing artifacts.

Although explicit source identifiers such as `Reuters` were removed during preprocessing, indirect source-specific signals remain present in the text.

This may make the classification task easier than real-world fake-news detection.


### Classification is not fact verification

The model performs **text classification**, not factual verification.

It does not:

- search for supporting evidence;
- compare claims against trusted sources;
- retrieve current information;
- reason about the factual consistency of a claim;
- determine whether an event actually occurred.

A Real prediction therefore means that the text resembles patterns associated with Real articles in the training data. It does **not** establish that the claims contained in the article are true.


### Out-of-distribution inputs

The model was trained on news articles following the distributions represented in the ISOT dataset.

Very short texts, social-media posts, articles from substantially different sources, newer writing styles, or other types of documents may behave differently from the test data used in this project.

Predictions on such inputs should therefore be interpreted cautiously.


### Decision scores are not probabilities

`LinearSVC` produces decision-function scores rather than calibrated probabilities.

A decision score of `2.0`, for example, does not mean that an article has a 200%, 2%, or any other probabilistic likelihood of being Real.

The score only indicates the position of the observation relative to the learned decision boundary.


## Future Improvements

Several extensions could be explored to evaluate and improve the robustness of the system.

### Cross-source evaluation

A particularly important next step would be to evaluate the trained classifier on a completely different fake-news dataset.

This would help determine how much of the current performance generalizes beyond the sources and editorial conventions represented in ISOT.


### Source-aware experiments

Additional experiments could investigate how performance changes after removing or normalizing more source-related signals.

Comparing these results with the current model could help quantify how strongly editorial fingerprints contribute to classification performance.


### Alternative text representations

The current model relies on TF-IDF features.

Future experiments could compare this approach with contextual language representations and transformer-based models.

However, more complex models should not automatically be assumed to solve the dataset-bias problem: a powerful language model may also learn source-specific shortcuts if they remain predictive in the training data.


### Probability calibration

If probabilistic outputs were required for a future application, the SVM could be combined with an appropriate calibration method.

Any resulting probabilities would still represent model uncertainty within the learned classification task and should not be interpreted as probabilities of factual truth.


### External validation

A stronger evaluation framework could include articles from:

- previously unseen publishers;
- different time periods;
- different political contexts;
- different geographic regions;
- multiple independent datasets.

Such evaluation would provide a more realistic measure of the model's ability to generalize beyond its original training distribution.


### Explainability

The current project examines global SVM coefficients and misclassified examples.

Future work could extend this analysis with local explanations showing which features contribute most strongly to individual predictions.


## Key Takeaway

The final TF-IDF + Linear SVM classifier achieves excellent performance on the held-out ISOT test set, reaching approximately **98.7% accuracy and macro F1-score**.

However, the most important result of the project is not the score alone.

Error analysis and feature inspection show that the classifier learns meaningful differences between the two classes while also exploiting **editorial, stylistic, and source-related patterns** present in the dataset.

The project therefore demonstrates both the effectiveness of classical NLP methods for text classification and the importance of critically examining **what a high-performing machine learning model has actually learned**.
