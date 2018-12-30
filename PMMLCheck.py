import pandas

iris_df = pandas.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
#from com.mycompany import Aggregator, PowerFunction

iris_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (["sepal_length", "petal_length"], [ContinuousDomain()]), #, Aggregator(function = "mean")
        (["sepal_width", "petal_width"], [ContinuousDomain()]) #, PowerFunction(power = 2)
    ])),
    ("classifier", LogisticRegression())
])
iris_pipeline.fit(iris_df, iris_df["species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_pipeline, "serialisation\\Iris.pmml", user_classpath = ["/path/to/sklearn2pmml-plugin/target/sklearn2pmml-plugin-1.0-SNAPSHOT.jar"])

#-----------------------------------------------
from sklearn.tree import DecisionTreeClassifier

pipeline = PMMLPipeline([
    ("classifier", DecisionTreeClassifier())
])
pipeline.fit(iris_df[iris_df.columns.difference(["species"])], iris_df["species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "serialisation\\DecisionTreeIris.pmml", with_repr = True)

#------------------------------------------------

from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (["sepal_length", "sepal_width", "petal_length", "petal_width"], [ContinuousDomain(), Imputer()])
    ])),
    ("pca", PCA(n_components = 3)),
    ("selector", SelectKBest(k = 2)),
    ("classifier", LogisticRegression())
])
pipeline.fit(iris_df, iris_df["species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "serialisation\\LogisticRegressionIris.pmml", with_repr = True)



#--------------------------------------------------------

import subprocess
from openscoring import Openscoring
import numpy as np
from os import getcwd



#p = subprocess.Popen('java -jar '+ getcwd() +'\\java\\openscoring-server-executable-1.4.3.jar', shell=True)

'''
p = subprocess.Popen('java -jar C:\\Users\\1\\PycharmProjects\\FlaskModelServing\\resourses\\java\\openscoring-client-executable-1.4.3.jar',
                     shell=True)

os = Openscoring("http://localhost:8080/openscoring")
os = Openscoring("http://localhost:8080/openscoring")

# Deploying a PMML document DecisionTreeIris.pmml as an Iris model:
os.deployFile("Iris", "serialisation\\DecisionTreeIris.pmml")

# Evaluating the Iris model with a data record:
arguments = {
    "sepal_length" : 5.1,
    "sepal_width" : 3.5,
    "setal_length" : 1.4,
    "setal_width" : 0.2
}
result = os.evaluate("Iris", arguments)
print(result)
'''


from openscoring import Openscoring
os = Openscoring("http://localhost:5000/openscoring")

# A dictionary of user-specified parameters
kwargs = {"auth" : ("admin", "adminadmin")}
os.deployFile("Iris", "serialisation\\DecisionTreeIris.pmml", **kwargs)

arguments = {
    "sepal_length" : 5.1,
    "sepal_width" : 3.5,
    "petal_length" : 1.4,
    "petal_width" : 0.2
}

'''
result = os.evaluate("Iris", arguments)
print(result)


from openscoring import EvaluationRequest
evaluationRequest = EvaluationRequest("record-001", arguments)
evaluationResponse = os.evaluate("Iris", evaluationRequest)
print(evaluationResponse.result)


os.evaluateCsvFile("Iris", "Iris.csv", "Iris-results.csv")
os.undeploy("Iris", **kwargs)
'''