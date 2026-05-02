# Conflation in Toxicity Detection: Addressing Demographic Bias in Automated Moderation

## Overview
The development of automated toxicity detection systems has become a critical component of modern online discourse moderation. While these systems operate efficiently at scale, they frequently struggle with demographic bias and algorithmic fairness. Research in Natural Language Processing (NLP) shows that models trained on human-labeled datasets often inherit and amplify existing social prejudices, leading to systematic errors when classifying speech related to protected identity groups.

This project investigates a phenomenon known as **conflation** in Transformer-based classifiers, where the mere mention of a demographic category (e.g., "Muslim," "Gay," "Black") causes the model to incorrectly flag the content as toxic. This unintended bias unfairly suppresses marginalized voices and undermines trust in online communities. We explore the technical mechanisms behind this bias and evaluate targeted interventions designed to build moderation tools that are technically robust and socially equitable.

## Dataset and Objective
Our research utilizes the **Jigsaw Unintended Bias in Toxicity Classification** dataset—created by Jigsaw, a unit of Google. The primary objective is to move beyond standard classification accuracy—which often masks underlying prejudice—and focus on fairness-aware evaluation metrics and robust mitigation strategies. We aim to decouple the False Positive Rate (FPR) and False Negative Rate (FNR) objectives to reduce demographic bias without sacrificing global performance. We also validate our framework through a zero-shot cross-dataset evaluation on the **HateXplain** benchmark to confirm its generalizability.

## Methodology: The Five-Phase Pipeline
We developed this project through a structured, five-phase experimental pipeline to progressively audit, understand, and mitigate demographic bias.

### Phase 1: Baseline Model
We began with a traditional machine learning approach to establish a control model and observe how bias manifests in standard setups.
- **Preprocessing:** Text converted to lowercase, special characters and URLs removed, followed by lemmatization.
- **Feature Extraction:** TF-IDF vectorization.
- **Model:** Logistic Regression.
- **Observation:** The model achieved a very high accuracy of 94%. However, this global accuracy metric successfully obscured underlying bias, confirming that accuracy alone is insufficient for evaluating fairness.

### Phase 2: Fairness Audit
This phase focused on the direct measurement of bias to detect if the model performs differently across distinct demographic groups.
- **Data Partitioning:** Divided the dataset into specific identity subgroups (e.g., muslim, black, female) and background data.
- **Metrics Computed:** False Positive Rate (FPR), False Negative Rate (FNR), and FPR/FNR gaps.
- **Observation:** The audit revealed that the model exhibited highly varied behavior against different demographic groups, proving that accuracy does not reflect fairness.

### Phase 3: Transformer Model and Sample Weighting
We transitioned to more complex Transformer models to increase contextual comprehension, introducing sample weighting to penalize incorrect predictions heavily (3x weight).
- **Setup:** Balanced the dataset into a 50-50 split and implemented standard tokenization.
- **Observation:** While the Transformer improved Toxicity Recall, it inadvertently increased False Positives (FP) across the board.

### Phase 4: Controlled Experiment (Vanilla vs Adaptive Models)
To understand the exact impact of weighting on fairness, we conducted a controlled comparison.
- **Setup:** We trained three distinct models on identical data, splits, and hyperparameters:
  1. Logistic Regression (baseline)
  2. Vanilla Transformer (no weighting)
  3. Adaptive Weighted Transformer (using the adaptive weights derived from Phase 2 FPR gaps)
- **Observation:** While vanilla Transformers significantly increased the FPR, the adaptive weighting strategy increased the FNR and further worsened the FPR, indicating that simple weighting is insufficient for complex bias.

### Phase 5: Advanced Bias Mitigation (Final Model)
Our final model implements a targeted, three-prong approach to explicitly decouple identity terms from toxicity signals.
1. **Domain-Specific Transformer (HateBERT):** We utilized a HateBERT backbone pre-trained on real-world abusive language datasets. Its deeper understanding of genuine abusive speech patterns helps decrease False Negatives (missed toxic content).
2. **Counterfactual Data Augmentation (CDA):** We actively swapped specific identity-related terms within demographic categories (e.g., swapping "Muslim" ↔ "Christian" or "Brown" ↔ "White"). For example, changing "Hindus are dangerous" to "Christians are dangerous" during training forces the model to separate the correlation between identity mentions and toxicity.
3. **Asymmetric Loss Weighting:** We designed a custom, asymmetric loss function strategy to specifically penalize false alarms when neutral identity mention terms are present.

## Conclusion
By implementing specialized model architectures, counterfactual data augmentation, and custom loss functions, our findings demonstrate that it is possible to significantly reduce demographic bias compared to traditional methods while maintaining the classifier's global performance. 

**Keywords:** Toxicity Detection, Demographic Bias, Fairness Metrics (FPR/FNR Gaps), HateBERT, Bias Mitigation
