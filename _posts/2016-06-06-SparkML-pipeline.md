---
layout: post
title: SparkML pipeline
subtitle: SparkML pipeline implementation
---

```python
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
hc = HiveContext(sc)
```


```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row, SQLContext
```

prepare training document from a list of (id, text, label) tuples.


```python
training = sqlContext.createDataFrame([
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0),
    (4L, "b spark who", 1.0),
    (5L, "g d a y", 0.0),
    (6L, "spark fly", 1.0),
    (7L, "was mapreduce", 0.0),
    (8L, "e spark program", 1.0),
    (9L, "a e c l", 0.0),
    (10L, "spark compile", 1.0),
    (11L, "hadoop software", 0.0)], ["id", "text", "label"])
```


```python
training.show()
```

    +---+----------------+-----+
    | id|            text|label|
    +---+----------------+-----+
    |  0| a b c d e spark|  1.0|
    |  1|             b d|  0.0|
    |  2|     spark f g h|  1.0|
    |  3|hadoop mapreduce|  0.0|
    |  4|     b spark who|  1.0|
    |  5|         g d a y|  0.0|
    |  6|       spark fly|  1.0|
    |  7|   was mapreduce|  0.0|
    |  8| e spark program|  1.0|
    |  9|         a e c l|  0.0|
    | 10|   spark compile|  1.0|
    | 11| hadoop software|  0.0|
    +---+----------------+-----+
    


Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and logisticRegression


```python
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
```


```python
tokenizer.transform(training).show()
```

    +---+----------------+-----+--------------------+
    | id|            text|label|               words|
    +---+----------------+-----+--------------------+
    |  0| a b c d e spark|  1.0|[a, b, c, d, e, s...|
    |  1|             b d|  0.0|              [b, d]|
    |  2|     spark f g h|  1.0|    [spark, f, g, h]|
    |  3|hadoop mapreduce|  0.0| [hadoop, mapreduce]|
    |  4|     b spark who|  1.0|     [b, spark, who]|
    |  5|         g d a y|  0.0|        [g, d, a, y]|
    |  6|       spark fly|  1.0|        [spark, fly]|
    |  7|   was mapreduce|  0.0|    [was, mapreduce]|
    |  8| e spark program|  1.0| [e, spark, program]|
    |  9|         a e c l|  0.0|        [a, e, c, l]|
    | 10|   spark compile|  1.0|    [spark, compile]|
    | 11| hadoop software|  0.0|  [hadoop, software]|
    +---+----------------+-----+--------------------+
    



```python
df = sqlContext.createDataFrame([(["a", "b", "b"],),
                                 (["d", "e", "f"],),
                                 (["g", "e", "c"],)], ["words"])
```


```python
df.show()
```

    +---------+
    |    words|
    +---------+
    |[a, b, b]|
    |[d, e, f]|
    |[g, e, c]|
    +---------+
    



```python
hashingTF = HashingTF(numFeatures=10, inputCol="words", outputCol="features")
```


```python
hashingTF.transform(df).show()
```

    +---------+--------------------+
    |    words|            features|
    +---------+--------------------+
    |[a, b, b]|(10,[7,8],[1.0,2.0])|
    |[d, e, f]|(10,[0,1,2],[1.0,...|
    |[g, e, c]|(10,[1,3,9],[1.0,...|
    +---------+--------------------+
    


use a ParamGridBuilder to construct a grid of parameters to search over  
with 3 values for hashingTF.numFeatures and 2 values for lr.regParam,  
this grid will have 3X2 = 6 parameter settings for CrossValidator to choose from.


```python
paraGrid = ParamGridBuilder().addGrid(
    hashingTF.numFeatures, [10, 100, 1000]
).addGrid(
    lr.regParam, [0.1, 0.01]
).build()
```


```python
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paraGrid, evaluator=BinaryClassificationEvaluator(),
                    numFolds=2)
```


```python
cv_model = cv.fit(training)
```


```python
# Prepare test documents, which are unlabeled (id, text) tuples.
```


```python
test = sqlContext.createDataFrame([
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "mapreduce spark"),
    (7L, "apache hadoop")], ["id", "text"])
```


```python
# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cv_model.transform(test)
selected = prediction.select('id', 'text', 'probability', 'prediction')
```


```python
prediction.show()
```

    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    | id|           text|             words|            features|       rawPrediction|         probability|prediction|
    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    |  4|    spark i j k|  [spark, i, j, k]|(262144,[105,106,...|[-1.0143531895130...|[0.26612878920913...|       1.0|
    |  5|          l m n|         [l, m, n]|(262144,[108,109,...|[2.45505377427970...|[0.92093023893998...|       0.0|
    |  6|mapreduce spark|[mapreduce, spark]|(262144,[62173,14...|[-0.2292614916964...|[0.44293435984699...|       1.0|
    |  7|  apache hadoop|  [apache, hadoop]|(262144,[128334,1...|[1.80181132002375...|[0.85836928288627...|       0.0|
    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    



```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
count_vect = CountVectorizer()
```


```python
trainingPdf = training.toPandas()
```


```python
del trainingPdf['id']
```


```python
trainingPdf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a b c d e spark</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b d</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spark f g h</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hadoop mapreduce</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b spark who</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>g d a y</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spark fly</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>was mapreduce</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>e spark program</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a e c l</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>spark compile</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>hadoop software</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train_counts = count_vect.fit_transform(trainingPdf)
```


```python
X_train_counts.shape
```




    (2, 2)




```python

```
