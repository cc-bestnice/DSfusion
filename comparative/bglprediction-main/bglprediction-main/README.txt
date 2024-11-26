Citation:
Please cite this work if you find this repository useful for your research:
Hoda Nemat, Heydar Khadem, Mohammad R. Eissa, Jackie Elliott and Mohammed Benaissa, "Blood Glucose Level Prediction: Advanced Deep-Ensemble Learning
Approach".
Requirements:
The codes require Python ≥ 3.6, TensorFlow ≥ 1.15.0, Keras ≥ 2.2.5, Pandas, NumPy, Sklearn, statsmodels, and scikit-posthocs.
Usage:
Access to the Ohio dataset is required to reproduce the results.
Run the 'xml_csv.py' file to extract blood glucose data from XML files and save them in CSV files. Note that the XML path needs is the path on disk where the to the folder containing the XML files for the Ohio dataset.
Run 'data_preparing'.py to take care of missing data and translate the time series problem to a supervised learning task.
Run 'basic_outputs.py' to implement the base-level of learning followed by 'advanced_outputs' to implement the meta-learning approaches. 'prediction_models' includes all the model's architectures required for base- and meta-learners.