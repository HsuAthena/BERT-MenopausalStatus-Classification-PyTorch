# BERT-text-classification
PyTorch BERT sentences classification model. Classified post or pre menopause status

Goal: . The goal was to classify sentences as indicating post-menopause, pre-menopause, or not related to menopause status.

Objective: Identify a patient's menopause status. I categorized the sentences into three groups: premenopause, postmenopause, and not related. Just because "menopause" is mentioned in the notes, it doesn't necessarily indicate the patient's current menopause status. For example, a lab test result might show that she is no longer in the premenopause stage. The patient wasn't premenopause stage. The classification model can filter out sentences that don't pertain to the current diagnosis.

I utilized the 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' model and trained it using a Colab GPU.