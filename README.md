# Random Forest Classifier
Create any kind of Random Forest Classifier with any dataset. 
This example uses [Graduate Admission 2](https://www.kaggle.com/mohansacharya/graduate-admissions) available on Kaggle

## Working of RFCs
The random forest classifier is a model made up of many decision trees. 
It works in the following steps:
1. Selects random samples from the given dataset.
2. Constructs decision trees based on different samples. 
3. Predicts result from each of the created decision tree.
4. Performs a vote for each predicted result.
5. Selects the prediction result with the most votes as final prediction.

## Dependencies
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
```

## Run the classifier
1. Clone the repository 
2. Install dependencies
3. Run the rfc.py 

### License
[Apache License 2.0](https://github.com/ani-poroorkara/RandomForestClassifier/blob/master/LICENSE)

##### I recommend using Google Colab or Jupyter notebooks to run the file cell by cell
##### Connect with me on [LinkedIn](https://www.linkedin.com/in/anirudh-poroorkara-34900017b/)
