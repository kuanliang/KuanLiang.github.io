

```python
import os
from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql import SQLContext
os.environ['PYSPARK_PYTHON'] = '/opt/anaconda2-4.0.0/bin/python'
conf = SparkConf()
conf.set('spark.master', 'yarn-client')
conf.set('spark.app.name', 'pipeline_test_andy')
conf.set('spark.yarn.queue', 'default')
conf.set('spark.executor.instances', '8')
conf.set('spark.executor.memory', '10g')
conf.set('spark.executor.cores', '5')
conf.set('spark.driver.memory', '10g')
conf.set('spark.driver.maxResultSize', '15g')
# conf.set('spark.dynamicAllocation.enabled', 'True')
conf.set('spark.jars', 'hdfs://ha/user/mlb/tmp/tony/sqljdbc4.jar')

sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
hc = HiveContext(sc)
```


```python
import mllib, utility
```


```python
from utility import isfloat
```


```python
sc.addFile('mllib.py')
```


```python
sc.addFile('utility.py')
```


```python
loader = mllib.DataLoader(model='N71', station='FCT')
logDf, fatpDf = loader.load_X(hiveContext=hc, shift=-1, span=1)
rpcDf = loader.load_y(sparkContext=sc, connect='jdbc')
```


```python
matrixCreator = mllib.MatrixCreator(logDf, fatpDf, rpcDf)
X = matrixCreator.create_X()
matrix = matrixCreator.create_matrix(X, rpcDf)
```

    top 1 symptom: MSCMDL0, count: 30
    top 1 fail location: U0600, count: 30
    top 2 symptom: MSC1MD48, count: 19
    top 2 fail location: U5200_RF, count: 25
    top 3 symptom: RUNINFACHECKIN01, count: 15
    top 3 fail location: J4100, count: 20
    top 4 symptom: EC032, count: 14
    top 4 fail location: J4200, count: 16



```python
colAnalyzer = mllib.ColAnalyzer()
colInfo = colAnalyzer.analyze_column(matrixCreator=matrixCreator, sparkContext=sc)
```

    There are 101 columns being removed
    the number is correct



```python
matrix
```




    DataFrame[serial_number: string, items: map<string,string>, SN: string, check_in_code: string, symptom: string, fail_location: string, ever_tfb_check_in: int, day: string, y: int, MSCMDL0: int, U0600: int, MSC1MD48: int, U5200_RF: int, RUNINFACHECKIN01: int, J4100: int, EC032: int, J4200: int, random: double, randInt: int]




```python
matrix[['serial_number', 'randInt']].show()
```

    +----------------+-------+
    |   serial_number|randInt|
    +----------------+-------+
    |F3X614301QEG2KNF|     92|
    |F3X61720PNLGKFCF|     66|
    |F3X61830LLAG2KME|     79|
    |F3X61930HYKG2KNF|     64|
    |F3X61940BZ1G2KME|     61|
    |F3X619419BKG2KNF|     59|
    |F3X61950N71G2KNF|     88|
    |F3X61950QJQG2KNF|     32|
    |F3X61951BNEG2KNF|     41|
    |F3X61960CLXG2KNF|     22|
    |F3X620203DJG2KME|     95|
    |F3X62020ARRG2KNF|     29|
    |F3X62020FWZG2KNF|     35|
    |F3X62020MTPG2KNF|     53|
    |F3X62020Q3SG2KNF|     57|
    |F3X62020TQFG2KNF|     19|
    |F3X6202120PG2KNF|     22|
    |F3X62022M9GG2KNF|     46|
    |F3X62022SKKG2KNF|     37|
    |F3X62022VVCG2KNF|      1|
    +----------------+-------+
    only showing top 20 rows
    



```python
iterator = mllib.Iterator(matrix=matrix, matrixCreator=matrixCreator, sparkContext=sc, colInfo=colInfo)
```


```python
modelsDict = iterator.modeling(samplingRatio=0.1)
```

    no target specified
    RUNINFACHECKIN01
    EC032
    MSC1MD48
    MSCMDL0
    J4200
    J4100
    U0600
    U5200_RF



```python
range(1, int(0.4 * 10)+1)
```




    [1, 2, 3, 4]




```python
for key in modelsDict.keys():
    print 'model: {}'.format(key)
    if key == 'all':
        print 'Precision rate during training: {}'.format(modelsDict[key]['model'].best_score_)
        print 'Best parameters: {}'.format(modelsDict[key]['model'].best_params_)
        print '\n'
        print modelsDict[key]['prediction_report_test']
        print '\n'
    else:
        for subkey in modelsDict[key].keys():
            if key == 'code':
                print 'code:{}'.format(subkey)
            elif key == 'location':
                print 'location:{}'.format(subkey)
            print 'precision rate during training: {}'.format(modelsDict[key][subkey]['model'].best_score_)
            print 'Best parameters: {}'.format(modelsDict[key][subkey]['model'].best_params_)
            print modelsDict[key][subkey]['prediction_report_test']
```

    model: all
    Precision rate during training: 0.848792345221
    Best parameters: {'clf__C': 0.1, 'clf__penalty': 'l1', 'clf__tol': 0.01}
    
    
                 precision    recall  f1-score   support
    
              0       0.96      0.92      0.94       278
              1       0.81      0.88      0.84       100
    
    avg / total       0.92      0.91      0.91       378
    
    
    
    model: code
    code:EC032
    precision rate during training: 0.45
    Best parameters: {'clf__C': 10, 'clf__penalty': 'l1', 'clf__tol': 0.1}
                 precision    recall  f1-score   support
    
              0       1.00      0.99      1.00       376
              1       0.33      0.50      0.40         2
    
    avg / total       0.99      0.99      0.99       378
    
    code:RUNINFACHECKIN01
    precision rate during training: 0.0833333333333
    Best parameters: {'clf__C': 1, 'clf__penalty': 'l1', 'clf__tol': 0.1}
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99       372
              1       0.50      0.67      0.57         6
    
    avg / total       0.99      0.98      0.99       378
    
    code:MSC1MD48
    precision rate during training: 0.266666666667
    Best parameters: {'clf__C': 5, 'clf__penalty': 'l1', 'clf__tol': 0.0001}
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.99       376
              1       0.00      0.00      0.00         2
    
    avg / total       0.99      0.98      0.98       378
    
    code:MSCMDL0
    precision rate during training: 0.343434343434
    Best parameters: {'clf__C': 1, 'clf__penalty': 'l1', 'clf__tol': 0.01}
                 precision    recall  f1-score   support
    
              0       0.99      0.96      0.97       369
              1       0.25      0.56      0.34         9
    
    avg / total       0.97      0.95      0.96       378
    
    model: location
    location:J4200
    precision rate during training: 0.0666666666667
    Best parameters: {'clf__C': 5, 'clf__penalty': 'l1', 'clf__tol': 1}
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.98       373
              1       0.10      0.20      0.13         5
    
    avg / total       0.98      0.97      0.97       378
    
    location:J4100
    precision rate during training: 0.141414141414
    Best parameters: {'clf__C': 1, 'clf__penalty': 'l1', 'clf__tol': 0.0001}
                 precision    recall  f1-score   support
    
              0       0.98      0.98      0.98       370
              1       0.25      0.25      0.25         8
    
    avg / total       0.97      0.97      0.97       378
    
    location:U0600
    precision rate during training: 0.251002965289
    Best parameters: {'clf__C': 5, 'clf__penalty': 'l1', 'clf__tol': 0.0001}
                 precision    recall  f1-score   support
    
              0       0.99      0.96      0.98       371
              1       0.19      0.43      0.26         7
    
    avg / total       0.97      0.96      0.96       378
    
    location:U5200_RF
    precision rate during training: 0.0555555555556
    Best parameters: {'clf__C': 1, 'clf__penalty': 'l1', 'clf__tol': 1}
                 precision    recall  f1-score   support
    
              0       0.98      0.94      0.96       370
              1       0.00      0.00      0.00         8
    
    avg / total       0.96      0.92      0.94       378
    



```python
matrix
```




    DataFrame[serial_number: string, items: map<string,string>, SN: string, check_in_code: string, symptom: string, fail_location: string, ever_tfb_check_in: int, day: string, y: int, MSCMDL0: int, U0600: int, MSC1MD48: int, U5200_RF: int, RUNINFACHECKIN01: int, J4100: int, EC032: int, J4200: int, random: double, randInt: int]




```python
predictor = mllib.Predictor()
```


```python
modelTester = mllib.ModelTester()
```


```python
matrixReturn = matrix[matrix['y'] == 1]
```


```python
matrixPass = matrix[matrix['y'] == 0]
```


```python
matrixPassRand = matrixPass[matrixPass['randInt'] == 1]
```


```python
matrixTest = matrixReturn.unionAll(matrixPassRand)
```


```python
matrixTest.count()
```




    1260




```python

```


```python
matrixReturn.count()
```




    394




```python
matrixReturn[matrixReturn['randInt'] == 1].count()
```




    0




```python
matrix[matrix['randInt'] == 1].show()
```

    +----------------+--------------------+----+-------------+-------+-------------+-----------------+----+---+-------+-----+--------+--------+----------------+-----+-----+-----+--------------------+-------+
    |   serial_number|               items|  SN|check_in_code|symptom|fail_location|ever_tfb_check_in| day|  y|MSCMDL0|U0600|MSC1MD48|U5200_RF|RUNINFACHECKIN01|J4100|EC032|J4200|              random|randInt|
    +----------------+--------------------+----+-------------+-------+-------------+-----------------+----+---+-------+-----+--------+--------+----------------+-----+-----+-----+--------------------+-------+
    |F3X62022VVCG2KNF|Map(adc_BIST_amux...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0| 0.01894632058600454|      1|
    |F3X6206018NG2KLF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.010992700667242561|      1|
    |F3X620603W9G2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.012372229753942032|      1|
    |F3X6212019JG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.013524377352855832|      1|
    |F3Y62060873G2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0| 0.01483432332223067|      1|
    |F3X62060GE2G2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.014405837990954184|      1|
    |F3Y62120J06G2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0| 0.01783593520583382|      1|
    |F3X62050XXDG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.015362995151970593|      1|
    |F3X62051DSLG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.017022747698262286|      1|
    |F3X620606KPG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.018531079780110704|      1|
    |F3X62120A10G2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.010654785857626248|      1|
    |F3Y62040VFDG2KME|Map(adc_BIST_amux...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0| 0.01750057605494948|      1|
    |F3X62060C6MG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.010010396082912876|      1|
    |F3X621204XSG2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.011672765783608652|      1|
    |F3Y62060232G2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.012948969309584224|      1|
    |F3Y620603Y7G2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0| 0.01627360594698124|      1|
    |F3X62120H75G2KNF|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.014535288909760746|      1|
    |F3Y620607UVG2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.013535875892556004|      1|
    |F3Y621203PRG2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.017788978093827068|      1|
    |F3Y621206KRG2KME|Map(l21_arc_alive...|null|         null|   null|         null|             null|null|  0|      0|    0|       0|       0|               0|    0|    0|    0|0.017758507031629067|      1|
    +----------------+--------------------+----+-------------+-------+-------------+-----------------+----+---+-------+-----+--------+--------+----------------+-----+-----+-----+--------------------+-------+
    only showing top 20 rows
    



```python
matrixReturn.show()
```

    +----------------+--------------------+----------------+----------------+--------------------+-------------+-----------------+----------+---+-------+-----+--------+--------+----------------+-----+-----+-----+-------------------+-------+
    |   serial_number|               items|              SN|   check_in_code|             symptom|fail_location|ever_tfb_check_in|       day|  y|MSCMDL0|U0600|MSC1MD48|U5200_RF|RUNINFACHECKIN01|J4100|EC032|J4200|             random|randInt|
    +----------------+--------------------+----------------+----------------+--------------------+-------------+-----------------+----------+---+-------+-----+--------+--------+----------------+-----+-----+-----+-------------------+-------+
    |F3X614301QEG2KNF|Map(adc_BIST_amux...|F3X614301QEG2KNF|     MASCFRAT150|tcTxPower tLTE:ba...|     U5200_RF|                2|2016-04-18|  1|      0|    0|       0|       1|               0|    0|    0|    0| 0.9255456960422731|     92|
    |F3X6202120PG2KNF|Map(adc_BIST_amux...|F3X6202120PG2KNF|     MASC1FSW105|ZombieCheck zombi...|        U0600|                0|2016-05-13|  1|      0|    1|       0|       0|               0|    0|    0|    0| 0.6617380030321693|     66|
    |F3X62022M9GG2KNF|Map(adc_BIST_amux...|F3X62022M9GG2KNF|      MASC1FQB74|  Muon_20mA_MaxValue|       FL4212|                4|2016-06-02|  1|      0|    0|       0|       0|               0|    0|    0|    0|   0.79859818507273|     79|
    |F3Y61820KS1G2KME|Map(adc_BIST_amux...|F3Y61820KS1G2KME|RUNINFACHECKIN01|   Runin FA手動Checkin|     J3001_RF|                0|2016-05-06|  1|      0|    0|       0|       0|               1|    0|    0|    0|  0.643039475111802|     64|
    |F3X618214N4H7W1A|Map(adc_BIST_amux...|F3X618214N4H7W1A|        MASCFQB9|1K Tone to Low_Po...|       FL4702|                0|2016-05-11|  1|      0|    0|       0|       0|               0|    0|    0|    0|  0.993745081921457|     99|
    |F3X61930QC1G2KNF|Map(adc_BIST_amux...|F3X61930QC1G2KNF|      MASCFCOS12|HSG profile dent/...|        J4200|                0|2016-05-11|  1|      0|    0|       0|       0|               0|    0|    0|    1|0.10358719282623352|     10|
    |F3X61930QC1G2KNF|Map(adc_BIST_amux...|F3X61930QC1G2KNF|      MASCFCOS12|HSG profile dent/...|        J4100|                0|2016-05-11|  1|      0|    0|       0|       0|               0|    1|    0|    0| 0.8529119201006735|     85|
    |F3X61950AVYG2KNF|Map(adc_BIST_amux...|F3X61950AVYG2KNF|     MASC1-SHORT|             CRB短路專案|        U0600|                0|2016-05-16|  1|      0|    1|       0|       0|               0|    0|    0|    0| 0.9174909845867214|     91|
    |F3Y61930ZSEG2KME|Map(adc_BIST_amux...|F3Y61930ZSEG2KME|      MASCFQA188|adc_temp2_rear_ca...|        R2220|                0|2016-05-11|  1|      0|    0|       0|       0|               0|    0|    0|    0|0.48758548868080454|     48|
    |F3Y619504J8G2KME|Map(adc_BIST_amux...|F3Y619504J8G2KME|      MASC1FQB71|     Ext Mic Present|        U3500|                0|2016-05-12|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.3130390411597612|     31|
    |F3Y62050QPWG2KME|Map(l21_arc_alive...|F3Y62050QPWG2KME|     MSC1MWBT-67|W/n20MHz/TxCal/24...|     R5205_RF|                0|2016-05-21|  1|      0|    0|       0|       0|               0|    0|    0|    0|0.15605225332120176|     15|
    |F3Y620607TDG2KME|Map(l21_arc_alive...|F3Y620607TDG2KME|    MASC1FPROC16|PROX_CAL_BF_AVE_Q...|        U4240|                0|2016-05-26|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.9641452809643636|     96|
    |F3Y621201QLG2KME|Map(l21_arc_alive...|F3Y621201QLG2KME|        MSC1MCC0|Can't test fail-C...|     U_WTR_RF|                0|2016-05-23|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.7965615864711212|     79|
    |F3X61931Q9TG2KNF|Map(adc_BIST_amux...|F3X61931Q9TG2KNF|     MSC1MWBT-67|W/n20MHz/TxCal/24...|     C4901_RF|                0|2016-05-13|  1|      0|    0|       0|       0|               0|    0|    0|    0|0.45633216055789205|     45|
    |F3X61931Q9TG2KNF|Map(adc_BIST_amux...|F3X61931Q9TG2KNF|     MSC1MWBT-67|W/n20MHz/TxCal/24...|     U5200_RF|                0|2016-05-12|  1|      0|    0|       0|       1|               0|    0|    0|    0| 0.6466279123200975|     64|
    |F3X61931Q9TG2KNF|Map(adc_BIST_amux...|F3X61931Q9TG2KNF|     MSC1MWBT-67|W/n20MHz/TxCal/24...|     L5410_RF|                0|2016-05-14|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.9773601799750383|     97|
    |F3X61932CT6G2KNF|Map(adc_BIST_amux...|F3X61932CT6G2KNF|     MASCFRAT150|tcTxPower tLTE:ba...|     J_UAT_RF|                0|2016-05-12|  1|      0|    0|       0|       0|               0|    0|    0|    0|  0.355887100674659|     35|
    |F3X61930V6RG2KNF|Map(adc_BIST_amux...|F3X61930V6RG2KNF|     MASCFRAT150|tcTxPower tLTE:ba...|     J_UAT_RF|                0|2016-05-12|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.4563302461615578|     45|
    |F3X619501JAG2KNF|Map(adc_BIST_amux...|F3X619501JAG2KNF|      MASCFREU50|tcSRL tLTEbB5c206...|     U_VOX_RF|                0|2016-05-13|  1|      0|    0|       0|       0|               0|    0|    0|    0| 0.9698568431689005|     96|
    |F3X619511AAG2KNF|Map(adc_BIST_amux...|F3X619511AAG2KNF|      MASC1FQB53|    X404 Detect Test|        J4100|                0|2016-05-16|  1|      0|    0|       0|       0|               0|    1|    0|    0| 0.2657551223000506|     26|
    +----------------+--------------------+----------------+----------------+--------------------+-------------+-----------------+----------+---+-------+-----+--------+--------+----------------+-----+-----+-----+-------------------+-------+
    only showing top 20 rows
    



```python
reload(mllib)
```




    <module 'mllib' from 'mllib.pyc'>




```python
predictionReportAll = modelTester.predict_all(matrixTest, modelsDict, colInfo)
```


```python
print predictionReportAll
```

                 precision    recall  f1-score   support
    
              0       0.93      0.94      0.93       866
              1       0.86      0.84      0.85       394
    
    avg / total       0.90      0.90      0.90      1260
    



```python
isfloat('0.2')
```




    True




```python
from pyspark.mllib.linalg import Vectors
training = sqlContext.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])
```


```python
training.show()
```

    +-----+--------------+
    |label|      features|
    +-----+--------------+
    |  1.0| [0.0,1.1,0.1]|
    |  0.0|[2.0,1.0,-1.0]|
    |  0.0| [2.0,1.3,1.0]|
    |  1.0|[0.0,1.2,-0.5]|
    +-----+--------------+
    



```python
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row, SQLContext
```


```python
LabeledDocument = Row("id", "text", "label")
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
    



```python
# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
```


```python
param_grid = ParamGridBuilder().addGrid(
    hashingTF.numFeatures, [10, 100, 1000]
).addGrid(
    lr.regParam, [0.1, 0.01]
).build()
```


```python
param_grid
```




    [{Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 10,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.1},
     {Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 100,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.1},
     {Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 1000,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.1},
     {Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 10,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.01},
     {Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 100,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.01},
     {Param(parent='HashingTF_4c95b638f03a58e3a281', name='numFeatures', doc='number of features'): 1000,
      Param(parent='LogisticRegression_4ba093332c0c295c5969', name='regParam', doc='regularization parameter (>= 0)'): 0.01}]




```python
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=BinaryClassificationEvaluator(), numFolds=2)
```


```python
cv_model = cv.fit(training)
```


```python
test = sqlContext.createDataFrame([
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "mapreduce spark"),
    (7L, "apache hadoop")], ["id", "text"])
```


```python
prediction = cv_model.transform(test)
selected = prediction.select('id', 'text', 'probability', 'prediction')
for row in selected.collect():
    print row
```

    Row(id=4, text=u'spark i j k', probability=DenseVector([0.248, 0.752]), prediction=1.0)
    Row(id=5, text=u'l m n', probability=DenseVector([0.9647, 0.0353]), prediction=0.0)
    Row(id=6, text=u'mapreduce spark', probability=DenseVector([0.4248, 0.5752]), prediction=1.0)
    Row(id=7, text=u'apache hadoop', probability=DenseVector([0.69, 0.31]), prediction=0.0)



```python
prediction.show()
```

    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    | id|           text|             words|            features|       rawPrediction|         probability|prediction|
    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    |  4|    spark i j k|  [spark, i, j, k]|(100,[5,6,7,65],[...|[-1.1090504722106...|[0.24804795226775...|       1.0|
    |  5|          l m n|         [l, m, n]|(100,[8,9,10],[1....|[3.30854866477883...|[0.96472091867403...|       0.0|
    |  6|mapreduce spark|[mapreduce, spark]|(100,[10,65],[1.0...|[-0.3029581451762...|[0.42483449974949...|       1.0|
    |  7|  apache hadoop|  [apache, hadoop]|(100,[69,94],[1.0...|[0.79992959245231...|[0.68995942006900...|       0.0|
    +---+---------------+------------------+--------------------+--------------------+--------------------+----------+
    



```python
matrix[['items']].take(1)
```




    [Row(items={u'mojave_pp16v_mesa_measured': u'15982.12801111111', u'buck6_perf4_pwm_measured': u'1206.3492', u'H8P_Gpu_mode3_Frequency': u'550', u'mbus_unlock': u'0', u'buck2_perf5_pwm_measured': u'698.4126', u'CT_accel_delta': u'0.8678139999999814', u'charger_ibat_adc_2000mA': u'1990.0000', u'buck3_perf0_pfm_measured': u'1799.1452', u'buck8_perf5_pfm_measured': u'800.9768', u'pmgr_isp_c_clk_soc_perf1': u'799998720', u'buck3_perf5_pfm_measured': u'1798.5347', u'CT_perfstate_3_0_delta': u'30.66320799999996', u'buck1_perf4_target': u'891', u'perf6_cpu_freq': u'1848', u'CT_pwm_baseline': u'228.9148559999999', u'buck4_perf6_pwm_measured': u'1099.5115', u'fix_iusb_10x_gain': u'1.079145', u'CT_gyro_delta': u'2.217757999999975', u'buck5_perf0_target': u'850', u'CT_Phosphorus_baseline': u'231.42189', u'buck3_perf6_pwm_measured': u'1802.8083', u'SoC_CCC_THERMAL0_Average_end': u'31.17', u'buck4_perf5_pwm_measured': u'1099.5115', u'buck6_perf1_pwm_measured': u'1206.3492', u'H8P_SoC_mode2_Voltage': u'805', u'CT_l21_spk_VBST': u'8.1145', u'buck0_perf1_pwm_pfm_delta': u'1.831500000000005', u'buck1_perf4_pwm_measured': u'894.9938', u'buck1_perf4_pwm_pfm_delta': u'4.884000000000014', u'buck6_perf5_pfm_measured': u'1203.9072', u'adc_BIST_ildo13_pp3v0_mesa': u'0.0000', u'buck0_perf2_pfm_measured': u'686.2026', u'buck1_perf0_target': u'0', u'H8P_Gpu_mode1_Bin_Index': u'1', u'buck8_perf1_target': u'800', u'buck7_perf6_pwm_pfm_delta': u'1.831499999999778', u'buck4_perf4_pwm_measured': u'1099.5115', u'adc_BIST_ildo14_pp1v2': u'72.6593', u'charger_vbat_float': u'4330.0000', u'pll4_clk': u'23999962', u'buck2_perf4_pfm_measured': u'694.7496', u'buck7_perf1_pfm_measured': u'800.9768', u'CT_BT_on': u'277.8025509999999', u'l21_spk_alive_output1': u'1', u'l21_spk_alive_output0': u'0', u'CT_arc_on': u'261.3137209999999', u'l21_arc_vbstmon_cal': u'8.217399999999999', u'perf5_cpu_freq': u'1800', u'CT_accel_off': u'230.650497', u'adc_BIST_ildo8_nc': u'0.0000', u'SoC_THERMAL0_Average_start': u'29.12', u'fix_bl_ch2_gain': u'29.95882', u'CT_bl_low_5500_delta': u'52.55184900000003', u'buck6_perf6_pwm_pfm_delta': u'4.883999999999787', u'buck6_perf1_pwm_pfm_delta': u'0', u'Hibernate_detect_IBAT': u'5.260021', u'bl_vwled1_2_flash': u'18101.2962051282', u'WV_Phosphorus_odr': u'26.993292', u'charger_ibat_adc_500mA': u'490.0000', u'buck7_perf6_pfm_measured': u'1093.4065', u'buck8_perf5_target': u'800', u'CT_codec_hp_aout': u'235.085999', u'buck0_perf0_pwm_pfm_delta': u'0', u'adc_BIST_ildo11_pp3v0_prox_irled': u'0.0000', u'adc_BIST_ldo9_pp2v85_cam_avdd': u'2847.3748', u'H8P_Cpu_perf2_Frequency': u'912', u'SoC_THERMAL1_Average_start': u'30.18', u'adc_BIST_ldo2_pp1v8_va': u'1796.7032', u'pmgr_isp_clk_soc_perf1': u'457124288', u'H8P_fixed_io_leakage': u'5', u'adc_ibuck3_pp1v8_sdram': u'92.7960', u'perf4_Buck0_target_dvfm_delta': u'0', u'adc_ibuck7_pp_cpu_sram': u'5.1409', u'buck4_perf4_target': u'1100', u'buck5_perf0_pwm_measured': u'851.6483', u'buck7_perf3_pwm_measured': u'836.9963', u'buck2_perf0_pwm_pfm_delta': u'3.052500000000009', u'buck2_perf0_pwm_measured': u'698.4126', u'charger_vbus_adc_500mA': u'4600.0000', u'perf4_cpu_freq': u'1512', u'buck7_perf3_pwm_pfm_delta': u'0.6105000000000018', u'VREF_180mV_CHECK': u'179.491638', u'adc_BIST_ldo14_pp1v2': u'1194.7496', u'adc_BIST_ldo10_pp0v9_nand': u'899.2673', u'SoC_CCC_THERMAL2_Average_end': u'31.28', u'CT_Phosphorus_sampling': u'233.157516', u'buck8_perf0_pwm_measured': u'0.0000', u'chestnut_VNEG': u'-5721.2628', u'buck7_perf4_pwm_pfm_delta': u'1.831499999999891', u'buck8_perf0_pwm_pfm_delta': u'0', u'H8P_gpu_io_leakage': u'27', u'buck0_perf2_pwm_pfm_delta': u'4.884000000000014', u'H8P_Cpu_perf1_Frequency': u'600', u'ccc_soc_testClkOut_cpu_perf0': u'395999360', u'ccc_soc_testClkOut_cpu_perf3': u'1199998144', u'ccc_soc_testClkOut_cpu_perf2': u'911998592', u'ccc_soc_testClkOut_cpu_perf5': u'1800247232', u'ccc_soc_testClkOut_cpu_perf4': u'1511997632', u'PA2_accel_average_z': u'-0.976448', u'CT_spk_boost_8V_on': u'329.2938839999999', u'PA2_accel_average_x': u'0.050485', u'PA2_accel_average_y': u'0.024622', u'SoC_THERMAL2_Average_delta': u'0', u'charger_vbus_adc_100mA': u'4625.0000', u'buck2_perf3_pwm_pfm_delta': u'3.662999999999897', u'pre_uvlo_gpio_state_upper': u'1', u'buck2_perf0_pfm_measured': u'695.3601', u'H8P_Gpu_mode2_Voltage': u'735', u'buck0_perf3_pfm_measured': u'763.7362', u'WV_Phosphorus_ASIC_std': u'9.584080', u'H8P_SoC_mode1_Bin_Index': u'1', u'H8P_Gpu_mode2_Bin_Index': u'2', u'WV_Phosphorus_ASIC_odr': u'26.702098', u'charger_vbus_ilim_500mA': u'495.6250550204094', u'TW_CM_gyro_odr': u'201.586218', u'buck0_perf2_target': u'688', u'Vol_up_ADC_pullup': u'1799.9999', u'charger_vbus_adc_990mA': u'4550.0000', u'CT_Phosphorus_delta': u'1.735625999999996', u'charger_vddmain_float_pmu': u'4340.0488', u'charger_ibus_adc_2000mA': u'1981.2000', u'buck0_perf4_pwm_measured': u'879.1208', u'charger_vbus_OV_EVENT': u'1', u'ildo6_acc_load_on': u'109.9340', u'buck0_perf0_pfm_measured': u'550.0610', u'PA2_CM_accel_average_y': u'0.024418', u'l21_arc_alive_output1': u'1', u'l21_arc_alive_output0': u'0', u'buck4_perf5_pwm_pfm_delta': u'5.494500000000016', u'buck6_perf4_pwm_pfm_delta': u'0', u'accel_average_y_delta': u'0.0002039999999999958', u'H8P_Cpu_perf1_Voltage': u'600', u'adc_tcal': u'3846.1538', u'buck4_perf5_target': u'1100', u'buck2_perf3_target': u'697', u'buck0_perf6_target': u'1025', u'WV_Phosphorus_ASIC_temp_average': u'553647.111111', u'CT_arc_boost_8V_on': u'337.2009889999999', u'buck7_perf4_pfm_measured': u'965.2014', u'adc_BIST_ldo7_pp3v0_prox_als': u'2992.6739', u'Vol_dn_ADC_pullup': u'1801.2183', u'buck5_perf5_pfm_measured': u'847.9853', u'buck7_perf0_pwm_pfm_delta': u'5.494599999999991', u'mbus_lock': u'1', u'buck3_perf3_pwm_measured': u'1802.8083', u'VREF_1V8_CHECK': u'1799.621582', u'Gpu_mode2_Buck1_target_dvfm_delta': u'3', u'adc_ibuck4_pp1v1_sdram': u'231.1162', u'pmgr_af_clk_soc_perf1': u'533374176', u'pre_uvlo_trip_hysteresis': u'294.8718000000003', u'SoC_CCC_THERMAL1_Average_start': u'27.98', u'l21_spk_vbat_mean': u'3364.804575', u'H8P_Cpu_perf5_Frequency': u'1800', u'H8P_Cpu_perf4_Bin_Index': u'4', u'buck6_perf6_target': u'1200', u'bl_iwled2_low_5500': u'5.216764211674559', u'bl_vwled1_2_off': u'3706.389325641025', u'charger_vbat_ilim_100mA': u'-101.6268300574294', u'CT_codec_hp_aout_delta': u'6.074723000000005', u'bl_iwled1_low_5500': u'5.171684554450539', u'CT_gpu_perf0': u'230.0719449999999', u'CT_gpu_perf1': u'279.0560909999999', u'CT_gpu_perf2': u'280.309601', u'CT_gpu_perf3': u'281.2738339999999', u'CT_gpu_perf4': u'285.3236389999999', u'buck7_perf2_pwm_pfm_delta': u'4.273499999999899', u'buck4_perf3_pfm_measured': u'1094.0170', u'buck6_perf0_pfm_measured': u'1205.1282', u'CT_stockholm_on': u'236.2431029999999', u'buck7_perf5_pfm_measured': u'1026.8620', u'CT_BT_off': u'276.549042', u'PMU_TDEV3_radio_pa_end': u'26.56', u'chestnut_VIN': u'3829.1262', u'adc_ibuck5_pp_fixed': u'84.8249', u'charger_vbus_ilim_2000mA': u'1993.790083816354', u'PA2_accel_std_y': u'0.000706', u'PMU_TJINT_delta': u'2.920000000000001', u'PMU_TDEV3_radio_pa_delta': u'0.5199999999999996', u'adc_BIST_irtc': u'0.9611', u'adc_BIST_amuxby_pmu_amux_by': u'48.0269', u'tigris_adc_ibat': u'160.0000', u'buck4_perf6_pwm_pfm_delta': u'5.494500000000016', u'TW_die_rev': u'7', u'buck5_perf3_pwm_measured': u'851.0378', u'H8P_Cpu_perf4_Frequency': u'1512', u'Gpu_mode0_Buck1_target_dvfm_delta': u'0', u'adc_in7_offset': u'-8.4493', u'iusb_tristar_sink': u'27.09571002969961', u'buck3_perf0_target': u'1800', u'adc_BIST_amuxa3_button_vol_up_l': u'1799.9999', u'buck6_perf0_target': u'1200', u'buck3_perf2_pwm_pfm_delta': u'4.884100000000216', u'adc_BIST_ldo1_pp3v3_usb': u'3293.0402', u'buck7_perf3_pfm_measured': u'836.3858', u'buck2_perf2_pwm_measured': u'699.0231', u'charger_vbat_adc_500mA': u'3710.0000', u'charger_ibat_adc_1000mA': u'990.0000', u'adc_BIST_amuxb1_bb_to_pmu_amux_smps3': u'48.0269', u'charger_ibat_suspend': u'230.9701832079592', u'diags_leakage_vbat': u'3800', u'CT_acc_load_delta': u'108.4788209999999', u'H8P_Cpu_perf6_Frequency': u'1848', u'buck3_perf4_pfm_measured': u'1798.5347', u'H8P_cpu_io_leakage': u'31', u'buck5_perf2_pfm_measured': u'848.5958', u'H8P_Gpu_mode3_Voltage': u'780', u'Phosphorus_rev': u'45', u'adc_BIST_ldo13_pp3v0_mesa': u'3007.3260', u'buck6_perf3_pfm_measured': u'1206.3492', u'Gpu_mode1_Buck1_target_dvfm_delta': u'0', u'buck8_perf3_pwm_pfm_delta': u'3.66300000000001', u'buck4_perf1_pwm_measured': u'1099.5115', u'charger_ibus_adc_500mA': u'499.1999', u'buck3_perf6_pfm_measured': u'1799.1452', u'buck3_perf0_pwm_measured': u'1802.8083', u'tigris_adc_vbat': u'3800.0000', u'buck5_perf0_pfm_measured': u'846.1538', u'buck4_perf1_target': u'1100', u'adc_BIST_buck6_pp1v2_camera': u'1206.3492', u'SoC_CCC_THERMAL2_Average_delta': u'3.239999999999998', u'buck0_perf1_pfm_measured': u'600.1221', u'adc_BIST_ildo9_pp2v85_cam_avdd': u'0.0000', u'buck2_perf1_target': u'806', u'buck2_perf2_target': u'697', u'buck5_perf5_pwm_pfm_delta': u'3.052499999999895', u'adc_BIST_buck3_sw3_imu_owl': u'1799.7557', u'buck0_perf6_pwm_pfm_delta': u'6.10499999999979', u'buck5_perf6_pwm_pfm_delta': u'3.662999999999897', u'l21_spk_syscfg_vpbr': u'13', u'buck1_perf4_pfm_measured': u'890.1098', u'accel_average_x_delta': u'-6.200000000000648-05', u'diags_leakage_iusb': u'1.128949307090123', u'adc_BIST_amuxa5_lcm_to_chestnut_pwr_en': u'48.0269', u'buck0_perf1_pwm_measured': u'601.9536', u'buck8_perf0_target': u'0', u'buck5_perf6_pwm_measured': u'851.0378', u'SOC_CHIP_ID': u'8000', u'pmgr_mca0_m_clk': u'23999962', u'H8P_Gpu_mode4_Voltage': u'890', u'buck3_perf2_pwm_measured': u'1803.4188', u'CT_speaker_on': u'259.6745', u'adc_BIST_amuxay_pmu_amux_ay': u'48.0269', u'buck3_perf5_pwm_measured': u'1802.8083', u'H8P_Gpu_mode4_Frequency': u'720', u'perf5_Buck0_target_dvfm_delta': u'0', u'CT_strobe_delta': u'24.97418199999992', u'buck8_perf4_pwm_pfm_delta': u'1.221000000000117', u'H8P_Cpu_perf6_Voltage': u'1025', u'PMU_TDEV2_rear_camera_end': u'27.42', u'buck0_perf3_target': u'766', u'H8P_io_leakage': u'82', u'charger_vbus_OV_adc': u'6375.0000', u'CT_soc_perf1_0_delta': u'14.270935', u'pmgr_aop_mcuclk': u'47999924', u'buck7_perf4_pwm_measured': u'967.0329', u'CT_accel_sampling': u'231.518311', u'pmgr_isp_c_clk': u'799998720', u'CT_perfstate_2_0_delta': u'23.62423699999999', u'PMU_TDEV1_forehead_start': u'24.80', u'buck4_perf0_pfm_measured': u'1094.0170', u'accel_std_x_delta': u'-4.700000000000005-05', u'CT_codec_mic_bias': u'231.1326139999999', u'adc_ibuck0_pp_cpu': u'96.6610', u'adc_BIST_amuxa4_button_vol_down_l': u'1796.3449', u'buck3_perf1_pwm_pfm_delta': u'3.663099999999985', u'chestnut_Ineg': u'5.9387', u'adc_BIST_amuxa0_ap_to_pmu_amux_out': u'48.0269', u'perf1_cpu_freq': u'600', u'PA2_accel_std_x': u'0.000755', u'ldo6_acc_load_delta': u'12.20999999999958', u'l21_spk_vbst_cal': u'13', u'H8P_board_id': u'0', u'pmgr_mcu_reg_clk': u'99999840', u'buck7_perf5_pwm_pfm_delta': u'3.052500000000236', u'SoC_THERMAL0_Average_delta': u'3.440000000000001', u'adc_BIST_buck2_pp_var_soc': u'697.1916', u'buck0_perf4_pwm_pfm_delta': u'7.325999999999908', u'Vdroop_recover': u'1', u'buck3_perf1_pfm_measured': u'1799.1452', u'buck2_perf3_pfm_measured': u'694.7496', u'TW_Reg_0x75': u'144', u'TW_Reg_0x74': u'127', u'TW_Reg_0x76': u'0', u'TW_Reg_0x71': u'0', u'TW_Reg_0x70': u'0', u'TW_Reg_0x73': u'0', u'TW_Reg_0x72': u'0', u'VA_compass_average_z': u'-63.078988', u'VA_compass_average_y': u'-81.546888', u'VA_compass_average_x': u'67.041492', u'CT_wifi_off': u'229.589828', u'pmgr_disp0_clk': u'359999424', u'buck5_perf4_target': u'850', u'pll2_clk': u'1199998144', u'CT_codec_mic_bias_delta': u'2.12133799999998', u'adc_BIST_amuxa2_button_ringer_a': u'1796.3449', u'CT_buck6_pwm_delta': u'5.39976500000003', u'buck7_perf2_target': u'800', u'CT_codec_off': u'229.011276', u'pmgr_tmps_clk': u'1199948', u'charger_vddmain_float': u'4340.0000', u'H8P_Cpu_perf1_Bin_Index': u'1', u'H8P_Cpu_perf5_Bin_Index': u'5', u'buck1_perf1_pwm_measured': u'678.8766', u'adc_vddout': u'3813.7973', u'CT_buck6_pwm': u'235.182434', u'buck8_perf4_target': u'956', u'CT_gpu_perf2_0_delta': u'50.23765600000004', u'CT_speaker_delta': u'29.98825099999999', u'bl_vwled1_2_med_12600': u'16212.33825641025', u'perf0_Buck0_target_dvfm_delta': u'2', u'CT_BT_delta': u'6.171295000000043', u'buck3_perf3_pwm_pfm_delta': u'4.273599999999987', u'buck6_perf2_pwm_measured': u'1206.3492', u'buck6_perf2_pwm_pfm_delta': u'1.221000000000003', u'CT_strobe_baseline': u'243.7643429999999', u'buck6_perf6_pwm_measured': u'1206.3492', u'perf3_Buck0_target_dvfm_delta': u'1', u'PMU_TDEV4_ap_start': u'28.20', u'pmgr_isp_sensor1_ref_clk': u'11999980', u'buck5_perf2_target': u'850', u'buck5_perf2_pwm_pfm_delta': u'2.441999999999893', u'brickid_gain': u'1.0021', u'buck2_perf1_pwm_pfm_delta': u'5.494500000000016', u'CT_bl_off_delta': u'0', u'buck6_perf1_pfm_measured': u'1206.3492', u'SoC_CCC_THERMAL1_Average_end': u'31.09', u'buck4_perf2_pwm_pfm_delta': u'4.884000000000014', u'l21_spk_vbstmon_cal': u'8.114499999999999', u'CT_bl_high_22000_delta': u'216.1862479999999', u'adc_temp3_radio_pa': u'9575.2866', u'buck8_perf5_pwm_pfm_delta': u'4.273500000000012', u'CT_nand_cgpg_off_delta': u'-1.157088999999984', u'chestnut_VBST': u'6043.5407', u'adc_BIST_buck7_pp_cpu_sram': u'807.0818', u'buck5_perf4_pfm_measured': u'847.9853', u'buck2_perf5_pwm_pfm_delta': u'2.441999999999893', u'chestnut_Ground': u'0.0000', u'buck2_perf2_pwm_pfm_delta': u'4.273500000000012', u'adc_BIST_buck4_sw1_pp1v1': u'1094.6275', u'buck6_perf5_pwm_measured': u'1206.3492', u'H8P_binning_rev': u'2', u'charger_vbat_adc_100mA': u'3980.0000', u'charger_vbus_adc_1000mA': u'4525.0000', u'adc_temp1_forehead': u'10012.2292', u'TW_CM_gyro_temp': u'29.376636', u'TW_wafer_id': u'16', u'charger_vbus_adc_2000mA': u'4825.0000', u'CT_gpu_perf1_0_delta': u'48.98414599999998', u'charger_vbus_ilim_490mA': u'484.3351912856938', u'adc_BIST_ildo12_pp1v8_always': u'0.0146', u'CT_codec_baseline': u'229.2041319999999', u'CT_speaker_off': u'229.686249', u'Vdroop_active': u'0', u'ldo6_bypass': u'3827.8388', u'buck5_perf4_pwm_pfm_delta': u'3.66300000000001', u'buck7_perf0_pwm_measured': u'805.2503', u'VA_compass_std_y': u'0.429905', u'l21_spk_vbat_max': u'3365.689799999999', u'VA_compass_std_z': u'0.280603', u'adc_BIST_amuxb0_bb_to_pmu_amux_smps1': u'48.0269', u'adc_BIST_amuxb2_nc': u'48.0269', u'TW_CM_gyro_average_z': u'-0.291880', u'buck1_perf6_pfm_measured': u'674.6031', u'CT_acc_load_on': u'338.16507', u'adc_BIST_ldo5_pp3v0_nand': u'2992.6739', u'buck3_perf4_pwm_pfm_delta': u'4.273599999999987', u'buck6_perf3_target': u'1200', u'buck1_perf0_pfm_measured': u'0.0000', u'buck0_perf4_target': u'875', u'RingerA_ADC_no_short': u'759.5375', u'adc_temp_buck8': u'30.6675', u'adc_temp_buck3': u'28.7092', u'adc_temp_buck2': u'31.0148', u'adc_temp_buck1': u'30.8412', u'adc_temp_buck0': u'31.0148', u'adc_temp_buck7': u'28.5355', u'adc_temp_buck6': u'31.5355', u'CT_strobe_warm_delta': u'24.58850100000006', u'adc_temp_buck4': u'31.8828', u'PMU_TJINT_end': u'31.83', u'buck2_perf6_pwm_pfm_delta': u'3.662999999999897', u'CT_wifi_delta': u'42.04142799999993', u'fix_ibat_10x_gain': u'0.791232', u'buck5_perf4_pwm_measured': u'851.6483', u'adc_BIST_ldo15_pp1v8_mesa': u'1795.4822', u'H8P_SoC_mode2_Bin_Index': u'2', u'buck8_perf2_pwm_measured': u'827.2283', u'diags_leakage_ibat': u'169.3781343525034', u'Vol_dn_ADC_no_short': u'48.0269', u'WV_Phosphorus_std': u'0.002873', u'charger_vbat_ilim_500mA': u'-495.816018563456', u'buck1_perf1_pfm_measured': u'675.2136', u'buck4_perf6_target': u'1100', u'buck5_perf1_pfm_measured': u'848.5958', u'buck8_perf1_pwm_pfm_delta': u'4.273500000000012', u'tigris_adc_ibus': u'0.0000', u'buck4_perf0_pwm_pfm_delta': u'5.494500000000016', u'buck5_perf5_target': u'850', u'buck4_perf6_pfm_measured': u'1094.0170', u'buck3_perf5_target': u'1800', u'CT_stockholm_delta': u'6.653274999999951', u'perf1_Buck0_target_dvfm_delta': u'0', u'buck3_perf1_pwm_measured': u'1802.8083', u'buck7_perf5_target': u'1025', u'chestnut_VLDO1': u'5701.3281', u'chestnut_VLDO2': u'5732.8914', u'chestnut_VLDO3': u'5111.5928', u'pmgr_af_clk': u'359999424', u'pmgr_isp_clk': u'457124288', u'buck6_perf1_target': u'1200', u'buck1_perf3_pwm_pfm_delta': u'3.052500000000236', u'ldo6_acc_load_off': u'3299.1452', u'SoC_CCC_THERMAL0_Average_start': u'27.85', u'adc_BIST_buck1_pp_gpu': u'674.6031', u'charger_ibus_adc_490mA': u'483.5999', u'buck3_perf0_pwm_pfm_delta': u'3.663099999999985', u'charger_ibus_adc_990mA': u'982.7999', u'buck8_perf2_pwm_pfm_delta': u'1.831500000000119', u'buck4_perf2_target': u'1100', u'pmgr_sep_clk': u'479999232', u'l21_arc_syscfg_vpbr': u'12', u'buck0_perf6_pwm_measured': u'1028.6935', u'CT_compass_off': u'231.904007', u'CT_wifi_BT_delta': u'48.21272299999998', u'buck7_perf6_pwm_measured': u'1095.2380', u'buck8_perf6_pwm_measured': u'805.2503', u'CT_arc_boost_8V_delta': u'75.887268', u'buck8_perf6_pfm_measured': u'799.1452', u'buck4_perf4_pfm_measured': u'1094.6275', u'buck2_perf4_target': u'697', u'charger_vbus_ilim_100mA': u'99.35078233230954', u'buck5_perf3_pfm_measured': u'847.3748', u'Vol_up_ADC_no_short': u'48.0269', u'buck4_perf4_pwm_pfm_delta': u'4.884000000000014', u'Hibernate_IBAT': u'5.260023', u'buck3_perf6_target': u'1800', u'adc_ibuck1_pp_gpu': u'25.4371', u'buck6_perf2_target': u'1200', u'H8P_fuse_rev': u'11', u'buck7_perf1_target': u'800', u'SoC_mode1_Buck2_target_dvfm_delta': u'2', u'PA2_CM_accel_odr': u'508.104789', u'adc_BIST_buck3_sw2_touch': u'1799.1452', u'buck1_perf0_pwm_measured': u'0.0000', u'PMU_TDEV4_ap_delta': u'1.850000000000001', u'pmgr_sio_p_clk': u'23999962', u'nco_ref0_clk': u'799998720', u'adc_BIST_buck8_pp_gpu_sram': u'798.5347', u'charger_ibat_adc_100mA': u'100.0000', u'CT_codec_on': u'229.396973', u'charger_EOC_vbat': u'4200.0000', u'H8P_SoC_mode1_Voltage': u'695', u'buck5_perf2_pwm_measured': u'851.0378', u'adc_BIST_ildo2_pp1v8_va': u'0.0000', u'buck3_perf4_pwm_measured': u'1802.8083', u'charger_ibat_OV': u'232.5100602604545', u'TW_CM_gyro_average_y': u'-1.521085', u'TW_CM_gyro_average_x': u'0.389778', u'CT_buck6_pfm': u'229.686249', u'adc_ibuck8_pp_gpu_sram': u'12.8522', u'SoC_THERMAL2_Average_end': u'0.00', u'accel_average_z_delta': u'-7.600000000007599-05', u'buck7_perf4_target': u'963', u'buck1_perf5_target': u'675', u'pmgr_usb480_0_clk': u'479999232', u'adc_temp_ldo10': u'32.0563', u'buck6_perf0_pwm_measured': u'1206.3492', u'buck1_perf2_target': u'738', u'CT_perfstate_5': u'300.6555789999999', u'CT_perfstate_4': u'275.1990969999999', u'CT_perfstate_3': u'260.6387329999999', u'CT_perfstate_2': u'253.5997619999999', u'CT_perfstate_1': u'233.350372', u'charger_ibus_adc_100mA': u'101.3999', u'VA_compass_odr': u'102.460367', u'buck7_perf5_pwm_measured': u'1029.9145', u'buck8_perf1_pwm_measured': u'805.2503', u'PA2_accel_odr': u'507.970595', u'pll3_clk': u'23999962', u'WV_Phosphorus_ASIC_average': u'532102.814814', u'l21_spk_vbat_stdev': u'0.4190061384742206', u'charger_EOC_vddmain': u'4400.0000', u'adc_BIST_ldo4_pp0v8_owl': u'799.1452', u'CT_strobe_on': u'268.7385249999999', u'HoldKey_no_short': u'0', u'buck1_perf1_target': u'675', u'buck5_perf0_pwm_pfm_delta': u'5.494500000000016', u'buck2_perf5_target': u'697', u'CT_perfstate_0': u'229.9755249999999', u'ldo6_acc_load_on': u'3286.9352', u'adc_ibuck2_pp_soc': u'143.9025', u'CT_acc_load_off': u'229.686249', u'tigris_adc_vbus': u'4975.0000', u'buck8_perf3_target': u'838', u'buck1_perf5_pwm_pfm_delta': u'3.052499999999895', u'WV_Phosphorus_average': u'99.442021', u'CT_buck6_off': u'229.7826689999999', u'adc_BIST_ildo15_pp1v8_mesa': u'0.0000', u'WV_Phosphorus_temp_average': u'32.588140', u'buck7_perf2_pfm_measured': u'801.5873', u'buck5_perf6_target': u'850', u'PMU_TDEV2_rear_camera_start': u'25.46', u'adc_BIST_buck4_pp1v1_sdram': u'1095.2380', u'buck3_perf1_target': u'1800', u'SoC_mode2_Buck2_target_dvfm_delta': u'1', u'buck0_perf2_pwm_measured': u'691.0866', u'tigris_adc_vdd': u'3800.0000', u'adc_temp_ldo5': u'31.5355', u'fix_vendor_code': u'2', u'adc_tjint': u'28.9193', u'CT_gpu_perf3_0_delta': u'51.20188899999996', u'buck6_perf4_pfm_measured': u'1206.3492', u'adc_temp4_ap': u'8826.2420', u'buck2_perf4_pwm_pfm_delta': u'4.273500000000012', u'buck4_perf5_pfm_measured': u'1094.0170', u'buck1_perf2_pfm_measured': u'737.4847', u'buck4_perf1_pfm_measured': u'1094.0170', u'SoC_CCC_THERMAL2_Average_start': u'28.04', u'buck2_perf5_pfm_measured': u'695.9706', u'buck4_perf2_pfm_measured': u'1094.6275', u'charger_EOC_vddmain_pmu': u'4398.6568', u'pll0_clk': u'1599997504', u'TW_Reg_0x6F': u'57', u'tigris_adc_die_temp': u'25.4920', u'fix_iusb_1000x_gain': u'107.9145', u'CT_wifi_on': u'271.6312559999999', u'BATT_VCC_SHORT_CHECK': u'0.000000', u'accel_std_y_delta': u'5.80000000000001-05', u'adc_BIST_amuxb3_nc': u'48.0269', u'SoC_THERMAL0_Average_end': u'32.56', u'ccc_soc_testClkOut_cpu_perf1': u'599999040', u'adc_BIST_ildo4_pp0v8_owl': u'12.1794', u'buck2_perf0_target': u'697', u'CT_pfm_baseline': u'170.191406', u'CT_nand_cgpg_delta': u'98.45069900000001', u'buck8_perf0_pfm_measured': u'0.0000', u'CT_compass_sampling': u'232.578979', u'H8P_Gpu_mode1_Voltage': u'675', u'CT_l21_arc_VBST': u'8.2202', u'PA2_CM_accel_average_z': u'-0.976372', u'buck3_perf4_target': u'1800', u'PA2_CM_accel_average_x': u'0.050547', u'adc_BIST_ldo6_pp3v3_acc': u'3296.7032', u'SoC_CCC_THERMAL0_Average_delta': u'3.32', u'brickid_offset': u'-48.1318', u'l21_spk_vbat_min': u'3363.8583', u'buck6_perf5_pwm_pfm_delta': u'2.441999999999779', u'buck8_perf5_pwm_measured': u'805.2503', u'buck2_perf2_pfm_measured': u'694.7496', u'adc_temp2_rear_camera': u'9712.6114', u'l21_arc_vbst_cal': u'9', u'adc_BIST_ildo5_pp3v0_nand': u'0.0000', u'buck4_perf3_pwm_pfm_delta': u'5.494500000000016', u'buck4_perf3_target': u'1100', u'charger_vbus_ilim_1000mA': u'1004.797965055669', u'pre_uvlo_gpio_state_lower': u'0', u'buck7_perf2_pwm_measured': u'805.8608', u'CT_stockholm_off': u'229.589828', u'CT_bl_low_5500': u'281.659546', u'buck2_perf6_target': u'697', u'buck3_perf5_pwm_pfm_delta': u'4.273599999999987', u'nco_alg0_clk': u'23999962', u'pmgr_vid0_clk': u'99999840', u'PA2_CM_accel_std_x': u'0.000802', u'PA2_CM_accel_std_y': u'0.000648', u'PA2_CM_accel_std_z': u'0.002258', u'CT_spk_boost_8V_delta': u'69.61938399999996', u'buck1_perf5_pwm_measured': u'678.2661', u'buck1_perf2_pwm_measured': u'742.3687', u'buck0_perf5_pfm_measured': u'1020.7570', u'pll1_clk': u'1439997760', u'buck3_perf2_pfm_measured': u'1798.5347', u'pmgr_usb_clk': u'119999808', u'H8P_Cpu_perf0_Frequency': u'396', u'CT_arc_delta': u'31.72389299999997', u'H8P_Gpu_mode1_Frequency': u'340', u'perf2_cpu_freq': u'912', u'buck6_perf6_pfm_measured': u'1201.4652', u'pmgr_isp_sensor0_ref_clk': u'11999980', u'pmgr_n_clk': u'23999963', u'PMU_TDEV3_radio_pa_start': u'26.04', u'buck0_perf6_pfm_measured': u'1022.5885', u'adc_in7_gain': u'1.0177', u'adc_BIST_buck3_pp1v8_sdram': u'1798.5347', u'buck8_perf6_target': u'800', u'CT_buck6_pfm_delta': u'-0.09641999999996642', u'adc_BIST_amuxb5_bb_to_pmu_amux_ldo11_sim1': u'48.0269', u'USB_SHORT_CHECK': u'-0.070699', u'VA_compass_temp': u'25.803100', u'CT_arc_off': u'229.589828', u'bl_iwled2_flash': u'81.96140235162799', u'buck7_perf0_target': u'800', u'Gpu_mode4_Buck1_target_dvfm_delta': u'1', u'adc_BIST_ldo8_nc': u'0.0000', u'CT_nand_cgpg_on': u'130.5605769999999', u'buck7_perf1_pwm_pfm_delta': u'4.273500000000012', u'buck5_perf3_pwm_pfm_delta': u'3.662999999999897', u'buck5_perf1_pwm_pfm_delta': u'3.052500000000009', u'buck0_perf3_pwm_measured': u'770.4517', u'PMU_TJINT_start': u'28.91', u'buck2_perf3_pwm_measured': u'698.4126', u'pmgr_spi0_n_clk': u'61531152', u'CT_wifi_BT_off': u'228.5291599999999', u'adc_brick_id_usb_dp': u'1787', u'adc_BIST_buck0_pp_cpu': u'548.2295', u'buck6_perf3_pwm_measured': u'1206.3492', u'H8P_Cpu_perf0_Voltage': u'545', u'buck8_perf1_pfm_measured': u'800.9768', u'charger_vbat_adc_1000mA': u'3710.0000', u'perf2_Buck0_target_dvfm_delta': u'3', u'adc_brick_id_usb_dm': u'1787', u'buck5_perf5_pwm_measured': u'851.0378', u'buck5_perf1_target': u'850', u'fix_ibat_1000x_gain': u'79.12320', u'chestnut_ILDO3': u'0.0000', u'chestnut_ILDO2': u'0.0000', u'chestnut_ILDO1': u'0.0000', u'H8P_Cpu_perf5_Voltage': u'1025', u'buck8_perf2_target': u'822', u'diags_leakage_vbus': u'4975', u'PMU_TDEV1_forehead_end': u'25.93', u'VA_compass_std_x': u'0.255020', u'H8P_Cpu_perf6_Bin_Index': u'6', u'buck1_perf5_pfm_measured': u'675.2136', u'charger_iusb_OV': u'4.493366507744556', u'CT_mojave_on': u'236.435944', u'bl_iwled2_high_22000': u'21.82482821419534', u'buck0_perf1_target': u'600', u'pmgr_usb480_1_clk': u'479999232', u'CT_bl_baseline': u'229.107697', u'adc_BIST_vrtc': u'2506.7155', u'adc_ibuck6_pp1v2_camera': u'0.0000', u'buck4_perf3_pwm_measured': u'1099.5115', u'adc_BIST_amuxa6_tristar_to_pmu_usb_brick_id': u'48.0269', u'buck0_perf0_pwm_measured': u'550.0610', u'WV_Phosphorus_ASIC_temp_std': u'377.960740', u'perf0_cpu_freq': u'396', u'charger_vbus_adc_490mA': u'4600.0000', u'buck0_perf0_target': u'547', u'SoC_THERMAL2_Average_start': u'0.00', u'buck1_perf6_pwm_measured': u'678.8766', u'PMU_TDEV2_rear_camera_delta': u'1.96', u'Gpu_mode3_Buck1_target_dvfm_delta': u'1', u'bl_vwled1_2_low_5500': u'15921.15475128205', u'pre_uvlo_vcc_main_meas': u'3268.620199999999', u'buck6_perf5_target': u'1200', u'buck8_perf3_pwm_measured': u'843.1013', u'accel_std_z_delta': u'-9.500000000000046-05', u'charger_iusb_suspend': u'0.462884969119071', u'CT_perfstate_1_0_delta': u'3.374847000000045', u'charger_vbat_adc_2000mA': u'3720.0000', u'CT_nand_cgpg_off': u'229.011276', u'pll_pcie_test_mux_clk': u'99999840', u'pre_uvlo_trip_upper': u'3346.1538', u'adc_BIST_buck3_sw1_pp1v8': u'1799.1452', u'H8P_Cpu_perf0_Bin_Index': u'0', u'adc_BIST_amuxa7_chestnut_to_pmu_adcmux': u'48.0269', u'charger_vbus_ilim_990mA': u'982.2182375862373', u'CT_bl_high_22000': u'445.2939449999999', u'adc_BIST_ildo7_pp3v0_prox_als': u'0.0000', u'buck2_perf1_pfm_measured': u'802.1978', u'clock_32k': u'32768', u'buck3_perf3_pfm_measured': u'1798.5347', u'CT_perfstate_4_0_delta': u'45.22357200000001', u'pll5_clk': u'339999456', u'buck1_perf2_pwm_pfm_delta': u'4.8839999999999', u'CT_strobe_off_delta': u'0.09642099999999232', u'buck8_perf6_pwm_pfm_delta': u'6.105100000000106', u'ildo6_acc_load_off': u'0.0000', u'buck5_perf3_target': u'850', u'H8P_Gpu_mode2_Frequency': u'474', u'CT_perfstate_5_0_delta': u'70.68005400000001', u'H8P_Cpu_perf4_Voltage': u'875', u'H8P_vdd_var_io_leakage': u'15', u'adc_BIST_ldo11_pp3v0_prox_irled': u'3000.0000', u'buck4_perf0_target': u'1100', u'pre_uvlo_fixture_vbat_offset': u'31.37980000000015', u'CT_bl_off': u'229.107697', u'CT_soc_perf1': u'244.2464599999999', u'CT_soc_perf0': u'229.9755249999999', u'buck0_perf5_pwm_pfm_delta': u'7.325999999999908', u'charger_ibus_adc_1000mA': u'990.6000', u'buck8_perf4_pwm_measured': u'961.5384', u'H8P_Cpu_perf3_Bin_Index': u'3', u'buck1_perf6_pwm_pfm_delta': u'4.273499999999899', u'buck8_perf4_pfm_measured': u'960.3174', u'buck1_perf1_pwm_pfm_delta': u'3.662999999999897', u'adc_BIST_ildo1_pp3v3_usb': u'2.2192', u'buck6_perf0_pwm_pfm_delta': u'1.221000000000003', u'buck6_perf2_pfm_measured': u'1205.1282', u'CT_mojave_delta': u'6.942551000000008', u'buck1_perf0_pwm_pfm_delta': u'0', u'diags_rev': u'20151023', u'buck6_perf3_pwm_pfm_delta': u'0', u'pmgr_aop_fab_clk': u'95999852', u'H8P_Cpu_perf3_Frequency': u'1200', u'CT_bl_med_12600_delta': u'121.1107169999999', u'bl_vwled1_2_high_22000': u'16525.91973846153', u'H8P_Cpu_perf2_Voltage': u'685', u'buck5_perf6_pfm_measured': u'847.3748', u'adc_BIST_ldo3_pp3v0_tristar': u'3001.2210', u'H8P_Cpu_perf2_Bin_Index': u'2', u'buck4_perf0_pwm_measured': u'1099.5115', u'buck0_perf5_pwm_measured': u'1028.0830', u'bl_iwled2_med_12600': u'12.38981708892406', u'chestnut_VREF': u'1240.9359', u'CT_strobe_cool_delta': u'24.87777700000003', u'TW_CM_gyro_std_z': u'0.043459', u'TW_CM_gyro_std_y': u'0.048350', u'TW_CM_gyro_std_x': u'0.042050', u'bl_iwled1_high_22000': u'21.79058200980251', u'buck0_perf5_target': u'1025', u'adc_BIST_amuxb7_bb_to_pmu_amux_smps4': u'48.0269', u'CT_strobe_off': u'243.8607639999999', u'chestnut_Die_temp': u'27.0787', u'buck3_perf2_target': u'1800', u'buck7_perf1_pwm_measured': u'805.2503', u'buck1_perf3_pwm_measured': u'785.1037', u'perf3_cpu_freq': u'1200', u'buck2_perf6_pwm_measured': u'698.4126', u'buck2_perf1_pwm_measured': u'807.6923', u'charger_vbat_ilim_2000mA': u'-1983.264200639003', u'buck7_perf6_target': u'1091', u'CT_codec_delta': u'0.3856969999999933', u'fix_bl_ch1_gain': u'29.86582', u'bl_iwled1_med_12600': u'12.30574951566707', u'IBAT_RESET': u'0.798404', u'buck3_perf3_target': u'1800', u'CT_gpu_perf4_0_delta': u'55.25169399999995', u'buck7_perf3_target': u'831', u'PMU_TDEV1_forehead_delta': u'1.129999999999999', u'WV_Phosphorus_temp_std': u'0.127075', u'l21_spk_vbat_pkpk': u'1.831499999999323', u'buck4_perf2_pwm_measured': u'1099.5115', u'H8P_Cpu_perf3_Voltage': u'765', u'CT_gyro_off': u'231.325455', u'CT_gyro_sampling': u'233.5432129999999', u'PA2_accel_std_z': u'0.002163', u'buck1_perf3_pfm_measured': u'782.0512', u'buck2_perf4_pwm_measured': u'699.0231', u'buck1_perf6_target': u'675', u'SoC_THERMAL1_Average_end': u'33.35', u'CT_mojave_off': u'229.4933929999999', u'adc_BIST_amuxa1_sphere_ref_to_amux': u'48.0269', u'CT_bl_med_12600': u'350.2184139999999', u'buck7_perf0_pfm_measured': u'799.7557', u'bl_iwled1_flash': u'82.01265526946859', u'adc_BIST_ildo3_pp3v0_tristar': u'0.3438', u'diags_leakage_pwr': u'649.2534333422863', u'buck0_perf4_pfm_measured': u'871.7948', u'SoC_CCC_THERMAL1_Average_delta': u'3.110000000000002', u'H8P_Gpu_mode3_Bin_Index': u'3', u'TW_die_id': u'5945', u'perf6_Buck0_target_dvfm_delta': u'0', u'charger_vbat_ilim_1000mA': u'-993.1717877942246', u'buck6_perf4_target': u'1200', u'buck4_perf1_pwm_pfm_delta': u'5.494500000000016', u'H8P_Gpu_mode4_Bin_Index': u'4', u'CT_strobe_cool': u'268.64212', u'buck8_perf2_pfm_measured': u'825.3968', u'pre_uvlo_trip_lower': u'3051.281999999999', u'buck0_perf3_pwm_pfm_delta': u'6.71550000000002', u'buck2_perf6_pfm_measured': u'694.7496', u'buck3_perf6_pwm_pfm_delta': u'3.663099999999985', u'CT_strobe_warm': u'268.352844', u'adc_BIST_ldo12_pp1v8_always': u'1797.9242', u'SoC_THERMAL1_Average_delta': u'3.169999999999994', u'adc_BIST_buck5_pp_fixed': u'848.5958', u'ildo6_acc_load_delta': u'109.934', u'buck5_perf1_pwm_measured': u'851.6483', u'adc_BIST_ildo10_pp0v9_nand': u'219.1086', u'CT_compass_delta': u'0.6749719999999968', u'RingerA_ADC_pullup': u'1785.3799', u'pmgr_disp0_clk_soc_perf1': u'480124256', u'buck8_perf3_pfm_measured': u'839.4383', u'adc_temp_buck5': u'31.0148', u'adc_BIST_ildo6_pp3v3_acc': u'0.0000', u'CT_nand_cgpg_baseline': u'230.1683649999999', u'PMU_TDEV4_ap_end': u'30.05', u'buck1_perf3_target': u'781'})]




```python

```
