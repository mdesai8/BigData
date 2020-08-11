from pyspark.sql import *
from pyspark import SparkConf

from pyspark.sql import DataFrame
from pyspark.sql.functions import rand
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
     # white space expression tokenizer
    WordTokenizer = Tokenizer(inputCol="descript", outputCol="words")
    
    # bag of words count
    countVectors = CountVectorizer(inputCol="words", outputCol="features")
    
    # label indexer
    label_stringIdx = StringIndexer(inputCol = "category", outputCol =
    "label")

    # build the pipeline
    nb_pipeline = Pipeline(stages=[WordTokenizer, countVectors,
    label_stringIdx])
    
    return nb_pipeline

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    nb_model_0 = nb_0.fit(training_df)
    nb_model_pred_0 = nb_model_0.transform(training_df)
    nb_pred_0 = nb_model_pred_0.select("nb_pred_0")
    
    nb_model_1 = nb_1.fit(training_df)
    nb_model_pred_1 = nb_model_1.transform(training_df)
    nb_pred_1 = nb_model_pred_1.select("nb_pred_1")
    
    nb_model_2 = nb_2.fit(training_df)
    nb_model_pred_2 = nb_model_2.transform(training_df)
    nb_pred_2 = nb_model_pred_2.select("nb_pred_2")
    
    svm_model_0 = svm_0.fit(training_df)
    svm_model_pred_0 = svm_model_0.transform(training_df)
    svm_pred_0 = svm_model_pred_0.select("svm_pred_0")
    
    svm_model_1 = svm_1.fit(training_df)
    svm_model_pred_1 = svm_model_1.transform(training_df)
    svm_pred_1 = svm_model_pred_1.select("svm_pred_1")
    
    svm_model_2 = svm_2.fit(training_df)
    svm_model_pred_2 = svm_model_2.transform(training_df)
    svm_pred_2 = svm_model_pred_2.select("svm_pred_2")

    join1 = training_df.join(nb_model_pred_0, training_df.id == nb_model_pred_0.id).select(training_df["*"],nb_model_pred_0["nb_pred_0"])
    join2 = join1.join(nb_model_pred_1, join1.id == nb_model_pred_1.id).select(join1["*"],nb_model_pred_1["nb_pred_1"])
    join3 = join2.join(nb_model_pred_2, join2.id == nb_model_pred_2.id).select(join2["*"],nb_model_pred_2["nb_pred_2"])
    join5 = join3.join(svm_model_pred_0, join3.id == svm_model_pred_0.id).select(join3["*"],svm_model_pred_0["svm_pred_0"])
    join6 = join5.join(svm_model_pred_1, join5.id == svm_model_pred_1.id).select(join5["*"],svm_model_pred_1["svm_pred_1"])
    join7 = join6.join(svm_model_pred_2, join6.id == svm_model_pred_2.id).select(join6["*"],svm_model_pred_2["svm_pred_2"])
    join8 = join7.withColumn('joint_pred_0',2*join7['nb_pred_0'] + join7['svm_pred_0'])
    join9 = join8.withColumn('joint_pred_1',2*join8['nb_pred_1'] + join8['svm_pred_1'])
    finaldf = join9.withColumn('joint_pred_2',2*join9['nb_pred_2'] + join9['svm_pred_2'])

    return finaldf

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    meta_base = base_features_pipeline_model.transform(test_df)
    meta_base = meta_base.drop('category','descript','words')
    
    meta_gen_base = gen_base_pred_pipeline_model.transform(meta_base)
    
    join7 = meta_gen_base
    join8 = join7.withColumn('joint_pred_0',2*join7['nb_pred_0'] + join7['svm_pred_0'])
    join9 = join8.withColumn('joint_pred_1',2*join8['nb_pred_1'] + join8['svm_pred_1'])
    finaldf = join9.withColumn('joint_pred_2',2*join9['nb_pred_2'] + join9['svm_pred_2'])
    
    meta_gen_meta = gen_meta_feature_pipeline_model.transform(finaldf)
    
    meta_pred = meta_classifier.transform(meta_gen_meta)
    
    final_pred = meta_pred.select('id','label','final_prediction')

    return final_pred
