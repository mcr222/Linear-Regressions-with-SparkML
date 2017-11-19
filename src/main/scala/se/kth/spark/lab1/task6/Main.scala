package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SQLContext, DataFrame }
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import se.kth.spark.lab1._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.evaluation.RegressionEvaluator

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sc.textFile(filePath).toDF()
    
     //Pipeline from previous task ----------------------------------------------------------------------------------------------
     val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokens")
      .setPattern(",")
     val reg_resp= regexTokenizer.transform(obsDF)

      
     val arr2Vect = new Array2Vector()
        .setInputCol("tokens")
        .setOutputCol("tokens_vector")
     
     val lSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("year")
    lSlicer.setIndices(Array(0))

    
     val v2d = new Vector2DoubleUDF((x: Vector) => x(0).toDouble).setInputCol("year").setOutputCol("label")

     
     val min_year =reg_resp.map(x => x.getList(1).get(0).toString().toDouble).reduce((a,b)=> Math.min(a, b))
     val lShifter = new DoubleUDF((x:Double) => x-min_year).setInputCol("label").setOutputCol("label_shifted")
     
     
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3))
    
     
     // Our linear regression related transformation -------------------------------------------------------------------------
     val myLR = new MyLinearRegressionImpl()
        .setLabelCol("label_shifted")
        .setFeaturesCol("features")
    
     val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    
    //Split data into training and test
    // 80% of our data will be for training and 20% will be for testing
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()
    
    //this will show 6 rows of data, but not transformed
    //since this will be done after going through the pipeline
    Predef println("Sample of partition for training data  ---------------------------------------------------------------------")
    train.show(6)
    val lrStage = 6
    
    val pipelineModel: PipelineModel = pipeline.fit(train)
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]
  
    println("Final rmse: " + myLRModel.trainingError(99) + "\n")
        
    //do prediction - print first k
    Predef println("Transformed data and some predictions ---------------------------------------------------------------------")
    val result = pipelineModel.transform(test)
    result.show(4)
    
    val eval = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label_shifted")
      .setPredictionCol("prediction")
    val rmse = eval.evaluate(result)
    println(s"Root-mean-square error on validation data = $rmse")
   
  }
}