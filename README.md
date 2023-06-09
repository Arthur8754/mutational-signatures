# Mutational signatures as predictive biomarkers for immunotherapy response

This project is a multi-disciplinary project, with the Biology Faculty of the Sherbrooke University and the MEDomics UdeS lab from the Science Faculty of the Sherbrooke University. It consists of studying the predictive aspect of mutational signatures for the response to an immune-checkpoint inhibitor treatment in colorectal cancers.

## Prerequisites
It is recommended to create a Python virtual environment. To install the useful dependencies for this project, enter this command at the root of the project :
```
pip install -r requirements.txt
```

## Execute the code
You can execute the following Jupyter notebooks :
- `classification.ipynb` : this notebook executes a binary classifier to predict a response to immunotherapy.
- `cox-regression.ipynb` : this notebook executes the Cox regression model to find the high risk patients and low risk patients and to estimate the survival probability and the no progression probability. 