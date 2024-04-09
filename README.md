# AEQUITAS-Debiasing

This project focuses on the analysis of a proprietary matching algorithm developed by Adecco. This algorithm matches job positions with potential candidates.
The datasets involved are the output of this algorithm. They primarily consist of textual fields, and one of the tasks will be to figure out how to handle this format appropriately.

The project's main objectives are:
- Bias detection: Investigating the presence of any bias in the data.
- Bias mitigation: If bias is detected, applying techniques to mitigate or remove it.

There are two types of matching in the datasets:
- Direct matching: Given a job position, the algorithm suggests the 'best' N candidates.
- Indirect matching: Given a candidate, the algorithm suggests the 'best' M job positions.

Existing tools for bias detection and mitigation:
- IBM AI Fairness 360: https://aif360.res.ibm.com/
- Fairlearn: https://fairlearn.org/
