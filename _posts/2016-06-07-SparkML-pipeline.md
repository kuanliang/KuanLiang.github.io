
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row, SQLContext
```

prepare training dataset from a list of (seq, class) tuples.


```python
training = sqlContext.createDataFrame([
    ("atc tcg cga agc gcg", 1.0),
    ("aaa aat atg tgc", 0.0),
    ("tac act ctg tga", 1.0),
    ("aat atg tga gac acg", 0.0),
    ("ttg atc tcg cga gac", 1.0),
    ("atc tcg cga gac act", 0.0),
    ("aaa aat atc tca caa", 1.0),
    ("aaa aaa aaa aat atc", 0.0),
    ("ttc tcc tcg cga", 1.0),
    ("aat atc tcg cga gat", 0.0),
    ("atc tcg cga gaa aat", 1.0)], ["seq", "class"])
```


```python
training.show()
```

    +-------------------+-----+
    |                seq|class|
    +-------------------+-----+
    |atc tcg cga agc gcg|  1.0|
    |    aaa aat atg tgc|  0.0|
    |    tac act ctg tga|  1.0|
    |aat atg tga gac acg|  0.0|
    |ttg atc tcg cga gac|  1.0|
    |atc tcg cga gac act|  0.0|
    |aaa aat atc tca caa|  1.0|
    |aaa aaa aaa aat atc|  0.0|
    |    ttc tcc tcg cga|  1.0|
    |aat atc tcg cga gat|  0.0|
    |atc tcg cga gaa aat|  1.0|
    +-------------------+-----+
    


Configure an ML pipeline, which consists of three stages:  
- **_tokenizer_**
- **_hashingTF_**
- **_logisticRegression_**  

Since we will combine these 3 stages as a pipeline, we can use **_.getOutputCol()_** function of the stage.


```python
tokenizer = Tokenizer(inputCol="seq", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(labelCol="class", regParam)
```

Before making them as a pipeline, let's take a look of each process.  
First the tokennizer:


```python
tokenizer.transform(training).show()
```

    +-------------------+-----+--------------------+
    |                seq|class|               words|
    +-------------------+-----+--------------------+
    |atc tcg cga agc gcg|  1.0|[atc, tcg, cga, a...|
    |    aaa aat atg tgc|  0.0|[aaa, aat, atg, tgc]|
    |    tac act ctg tga|  1.0|[tac, act, ctg, tga]|
    |aat atg tga gac acg|  0.0|[aat, atg, tga, g...|
    |ttg atc tcg cga gac|  1.0|[ttg, atc, tcg, c...|
    |atc tcg cga gac act|  0.0|[atc, tcg, cga, g...|
    |aaa aat atc tca caa|  1.0|[aaa, aat, atc, t...|
    |aaa aaa aaa aat atc|  0.0|[aaa, aaa, aaa, a...|
    |    ttc tcc tcg cga|  1.0|[ttc, tcc, tcg, cga]|
    |aat atc tcg cga gat|  0.0|[aat, atc, tcg, c...|
    |atc tcg cga gaa aat|  1.0|[atc, tcg, cga, g...|
    +-------------------+-----+--------------------+
    


We can see that we have added a new column called 'words', which store lists tokenized sequences.  
The tokenlize is basically just python.split().

The second step, HashingTF()  
Here is the result DataFrame


```python
hashingTF.transform(tokenizer.transform(training)).show()
```

    +-------------------+-----+--------------------+--------------------+
    |                seq|class|               words|            features|
    +-------------------+-----+--------------------+--------------------+
    |atc tcg cga agc gcg|  1.0|[atc, tcg, cga, a...|(262144,[96509,96...|
    |    aaa aat atg tgc|  0.0|[aaa, aat, atg, tgc]|(262144,[96321,96...|
    |    tac act ctg tga|  1.0|[tac, act, ctg, tga]|(262144,[96402,98...|
    |aat atg tga gac acg|  0.0|[aat, atg, tga, g...|(262144,[96340,96...|
    |ttg atc tcg cga gac|  1.0|[ttg, atc, tcg, c...|(262144,[96912,98...|
    |atc tcg cga gac act|  0.0|[atc, tcg, cga, g...|(262144,[96402,96...|
    |aaa aat atc tca caa|  1.0|[aaa, aat, atc, t...|(262144,[96321,96...|
    |aaa aaa aaa aat atc|  0.0|[aaa, aaa, aaa, a...|(262144,[96321,96...|
    |    ttc tcc tcg cga|  1.0|[ttc, tcc, tcg, cga]|(262144,[98429,11...|
    |aat atc tcg cga gat|  0.0|[aat, atc, tcg, c...|(262144,[96340,96...|
    |atc tcg cga gaa aat|  1.0|[atc, tcg, cga, g...|(262144,[96340,96...|
    +-------------------+-----+--------------------+--------------------+
    


We can see that the default numFeatures is 262144.  
We did not set this parameter above.  
Instead, we will try different numFeatures during gridsearch (**tunning the model**)

Time to pipe, put these 3 stages in a stage list and pass to a **_Pipeline_** function.


```python
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
```

Build up the parameter grid search process and specify hyperparameters for tunning the model.  
We have 3 values for hashingTF.numFeatures and 3 values for lr.regParam,   
therefore the gridsearch will search over 9(3X3) parameter settings with CrossValidation  and choose the best according to the metrics we provided.


```python
paraGrid = ParamGridBuilder().addGrid(
    hashingTF.numFeatures, [10, 100, 1000]
).addGrid(
    lr.regParam, [0.1, 0.01, 0.001]
).build()
```

Put everything into a CrossValidator, and fit the model.  
The default fold number is set to 3.


```python
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paraGrid,\
                    evaluator=BinaryClassificationEvaluator(labelCol="class", metricName="areaUnderPR"),\
                    numFolds=2)
```


```python
cv_model = cv.fit(training)
```

Extract the **_best model_** (among 9 models)


```python
cv_model.bestModel
```




    PipelineModel_451f856bc7e033de20b7




```python
cv_model.bestModel.stages
```




    [Tokenizer_46ffb9fac5968c6c152b,
     HashingTF_40e1af3ba73764848d43,
     LogisticRegression_451b8c8dbef84ecab7a9]




```python
for stage in cv_model.bestModel.stages:
    print 'stage: {}'.format(stage)
    print stage.params
    print '\n'
```

    stage: Tokenizer_46ffb9fac5968c6c152b
    [Param(parent='Tokenizer_46ffb9fac5968c6c152b', name='inputCol', doc='input column name'), Param(parent='Tokenizer_46ffb9fac5968c6c152b', name='outputCol', doc='output column name')]
    
    
    stage: HashingTF_40e1af3ba73764848d43
    [Param(parent='HashingTF_40e1af3ba73764848d43', name='inputCol', doc='input column name'), Param(parent='HashingTF_40e1af3ba73764848d43', name='numFeatures', doc='number of features'), Param(parent='HashingTF_40e1af3ba73764848d43', name='outputCol', doc='output column name')]
    
    
    stage: LogisticRegression_451b8c8dbef84ecab7a9
    []
    
    


The above only shows the parameters we set up for each stage, however  
1. the best parameter is not shown
2. strangely the parameter of the last stage, logisticRegression is not there??

other ways to access parameter from each stage.  


```python
stage1.getNumFeatures()
```




    10




```python
stage2.intercept
```




    1.5791827733883774




```python
stage2.weights
```




    DenseVector([-2.5361, -0.9541, 0.4124, 4.2108, 4.4707, 4.9451, -0.3045, 5.4348, -0.1977, -1.8361])



There should be a .getRegParam for stage2:logistic regression, just like we can get stage1.getNumFeatures.  
But there is no


```python
stage2.getNumFeatures
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-222-0a5834b76a1c> in <module>()
    ----> 1 stage2.getNumFeatures
    

    AttributeError: 'LogisticRegressionModel' object has no attribute 'getNumFeatures'


Prepare test set.


```python
training = sqlContext.createDataFrame([
    ("atc tcg cga agc gcg", 1.0),
    ("aaa aat atg tgc", 0.0),
    ("tac act ctg tga", 1.0),
    ("aat atg tga gac acg", 0.0),
    ("ttg atc tcg cga gac", 1.0),
    ("atc tcg cga gac act", 0.0),
    ("aaa aat atc tca caa", 1.0),
    ("aaa aaa aaa aat atc", 0.0),
    ("ttc tcc tcg cga", 1.0),
    ("aat atc tcg cga gat", 0.0),
    ("atc tcg cga gaa aat", 1.0)], ["seq", "class"])
```


```python
test = sqlContext.createDataFrame([
    ("atc tcg cgt gtt tta", 1.0),
    ("agg gga gac act", 0.0),
    ("atc tcg cga gaa", 1.0),
    ("atc tcc cca caa", 1.0)], ["seq", "class"])
```


```python
test.show()
```

    +-------------------+-----+
    |                seq|class|
    +-------------------+-----+
    |atc tcg cgt gtt tta|  1.0|
    |    agg gga gac act|  0.0|
    |    atc tcg cga gaa|  1.0|
    |    atc tcc cca caa|  1.0|
    +-------------------+-----+
    



```python
# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cv_model.transform(test)
selected = prediction.select('seq', 'probability', 'prediction')
```


```python
selected.show()
```

    +-------------------+--------------------+----------+
    |                seq|         probability|prediction|
    +-------------------+--------------------+----------+
    |atc tcg cgt gtt tta|[0.08486620645810...|       1.0|
    |    agg gga gac act|[0.01613786358370...|       1.0|
    |    atc tcg cga gaa|[0.08150146351361...|       1.0|
    |    atc tcc cca caa|[0.00211916464008...|       1.0|
    +-------------------+--------------------+----------+
    



```python
prediction.show()
```

    +-------------------+-----+--------------------+--------------------+--------------------+--------------------+----------+
    |                seq|class|               words|            features|       rawPrediction|         probability|prediction|
    +-------------------+-----+--------------------+--------------------+--------------------+--------------------+----------+
    |atc tcg cgt gtt tta|  1.0|[atc, tcg, cgt, g...|(10,[2,5,8,9],[1....|[-2.3779943023203...|[0.08486620645810...|       1.0|
    |    agg gga gac act|  0.0|[agg, gga, gac, act]|(10,[2,3,9],[1.0,...|[-4.1103174956886...|[0.01613786358370...|       1.0|
    |    atc tcg cga gaa|  1.0|[atc, tcg, cga, gaa]|(10,[2,7,8,9],[1....|[-2.4221193339725...|[0.08150146351361...|       1.0|
    |    atc tcc cca caa|  1.0|[atc, tcc, cca, caa]|(10,[2,3,4,5],[1....|[-6.1546118924008...|[0.00211916464008...|       1.0|
    +-------------------+-----+--------------------+--------------------+--------------------+--------------------+----------+
    



