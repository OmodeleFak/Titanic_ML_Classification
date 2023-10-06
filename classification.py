# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
import matplotlib.pyplot as plt
# %% Survived w.r.t Pclass

sns.countplot(x="Survived", hue = "Pclass", data=train).set_title("Survived by Pclass")

# %% Survived w.r.t Sex

sns.countplot(x="Survived", hue = "Sex", data=train)

#%% Survived w.r.t Embarked 

sns.countplot(x="Survived", hue = "Embarked", data=train).set_title("Survived by Embarked")


# %% Survived w.r.t Age distribution ?
sns.histplot(data = train, 
    x = "Age", 
    bins = 8, 
    hue = "Survived").set_title("Survived by Age_Dist")

#%% Age distribution

sns.displot(data = train, x = "Age", bins = 8, hue = "Survived")
# %% SibSp / Parch distribution ?

sns.histplot(data = train, 
    x = "SibSp", 
    y = "Parch").set_title("SibSp / Parch distribution")

# %% Age distribution ?

sns.histplot(data = train, 
    x = "Age",
    bins = 8).set_title("Age Distribution" )

# %% Survived w.r.t SibSp / Parch  ?

sns.countplot(x="Survived", hue = "SibSp", data=train).set_title("Survived w.r.t SibSp")
# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"

dummy_clf = DummyClassifier(random_state=2020, strategy = "uniform")

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


clf = DecisionTreeClassifier()

ct = ColumnTransformer(
    [
        ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
        ("age", SimpleImputer(), ["Age"]),
    ],
    remainder="passthrough",
)

selected_columns = ["Pclass", "Sex", "Age"]
train_x = train[selected_columns]
train_y = train["Survived"]

train_x =ct.fit_transform(train_x) #fit is for training

clf.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(clf, train_x, train_y))

truth = pd.read_csv("truth_titanic.csv")
test_x = test[selected_columns]
test_y = truth["Survived"]

test_x = ct.transform(test_x) #testing


print("Test Set Performance")
print(evaluate(clf, test_x, test_y))

