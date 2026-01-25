#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("./ud120-projects/"))
from tools.feature_format import featureFormat, targetFeatureSplit

# Load the dataset with proper error handling for the known pickle compatibility issue
# The error "STRING opcode argument must be quoted" occurs when loading
# pickle files created with Python 2.x in Python 3.x

# Try to load the dataset, with proper error handling for the compatibility issue
try:
    import joblib
    data_dict = joblib.load(open("./ud120-projects/final_project/final_project_dataset.pkl", "rb"))
except Exception as e:
    if "STRING opcode argument must be quoted" in str(e):
        print("ERROR: Cannot load the dataset due to pickle incompatibility.")
        print("The file './ud120-projects/final_project/final_project_dataset.pkl' was created with")
        print("an older Python/pickle version and is incompatible with the current Python version.")
        print("")
        print("To fix this issue, you need to recreate the pickle file using the current Python version.")
        print("Possible solutions:")
        print("1. Find the original data source and recreate the dataset")
        print("2. Use an older Python version (e.g., Python 3.7 or earlier) to convert the file")
        print("3. Convert the data to a different format (e.g., JSON, CSV)")
        print("")
        print(f"Full error: {e}")
        raise SystemExit(1)  # Exit with error code
    else:
        # Some other error occurred
        raise

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
### Overfit tree
from sklearn import tree
cls = tree.DecisionTreeClassifier()
cls.fit(features, labels)
pred = cls.predict(features)
print("train score:", cls.score(features, labels))


