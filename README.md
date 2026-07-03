A multi-class predictive engine designed to categorize used automobiles into four distinct acceptability tiers: Unacceptable, Acceptable, Good, or Very Good.

Core Architecture -
*   ML Task: Multi-Class Classification (4 target classes)
*   Model Implemented: Random Forest Classifier (Ensemble Learning)
*   Data Preprocessing: Ordinal Encoding (`OrdinalEncoder`) to strictly maintain the physical mathematical rank of qualities (e.g., *low < med < high*).  
Unlike standard categorical datasets, this architecture uses precise mapping to ensure the model retains the inherent "direction" of feature ranks, significantly boosting the baseline Random Forest precision scores.
