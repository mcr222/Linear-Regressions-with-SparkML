package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.DoubleType



object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF()
    rawDF.show(3)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val reg_resp= regexTokenizer.transform(rawDF)
    reg_resp.show(5)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
        .setInputCol("tokens")
        .setOutputCol("tokens_vector")
    val arr2Vect_resp = arr2Vect.transform(reg_resp)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
        .setInputCol("tokens_vector")
        .setOutputCol("year")
    lSlicer.setIndices(Array(0))
   
    val slicer_resp = lSlicer.transform(arr2Vect_resp)
    //slicer_resp.show(3)
    
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((x: Vector) => x(0).toDouble)
        .setInputCol("year")
        .setOutputCol("label")
    val v2d_resp = v2d.transform(slicer_resp)

   //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    val min_year = v2d_resp.sort("label", "tokens_vector").first().get(4)
    println(min_year)
    val lShifter = new DoubleUDF((x:Double) => x-min_year.toString().toDouble)
        .setInputCol("label")
        .setOutputCol("label_shifted")
    //val lShifter_resp = lShifter.transform(v2d_resp)
   //lShifter_resp.show(4)
   
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
        .setInputCol("tokens_vector")
        .setOutputCol("features")
    fSlicer.setIndices(Array(1,2,3))
    
    //val fSlicer_resp = fSlicer.transform(lShifter_resp)
    //fSlicer_resp.show(3)
    
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Fit is a little "weird" when there is no Estimator, since the transformation returned is simply
    //the concatenation of the Transformers in the pipeline. Fit essentially runs and discards the results
    //since it sees no Estimator and the Transformed returned needs no fitting.
    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions ??? Do predictions? about what?? 
    //val test = rawDF.randomSplit(Array(0.8,0.2))
    //This just runs by concatenating all Transforms in pipeline
    val final_result = pipelineModel.transform(rawDF)
    final_result.show(10)
    //Step11: drop all columns from the dataframe other than label and features
    final_result.drop("value", "tokens", "tokens_vector", "year").show(3)
    //or use print below instead
    //final_result.drop("value", "tokens", "tokens_vector", "year").take(3).foreach(x => println(x))

    
  }
}