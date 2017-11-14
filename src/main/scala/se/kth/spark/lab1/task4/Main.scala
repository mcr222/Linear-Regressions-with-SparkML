package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import se.kth.spark.lab1._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sc.textFile(filePath).toDF()

   //Pipeline from previous task ----------------------------------------------------------
     val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokens")
      .setPattern(",")
      
     val arr2Vect = new Array2Vector()
    .setInputCol("tokens")
     .setOutputCol("tokens_vector")
     
     val lSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("year")
    lSlicer.setIndices(Array(0))

     val v2d = new Vector2DoubleUDF((x: Vector) => x(0).toDouble).setInputCol("year").setOutputCol("label")

     val min_year = 1921 //Change to the real one!!
     val lShifter = new DoubleUDF((x:Double) => x-min_year).setInputCol("label").setOutputCol("label_shifted")
     
     
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3))
    
     
     // Linear regression related transformations ------------------------
    val myLR = new LinearRegression().setElasticNetParam(0.1).setRegParam(0.9).setMaxIter(50).setLabelCol("label_shifted").setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
   

    //Hyperparameter related operations
    //Build the parameter grid
    val paramGrid = new ParamGridBuilder()
    .addGrid( myLR.maxIter, Array(20,30,40,60,70,80))
    .addGrid(myLR.regParam, Array(0.4, 0.6, 0.8, 0.92, 0.95, 1.0))
    .build()
    
    
    val evaluator = new RegressionEvaluator
    
    //Cross validation
    val cvModel: CrossValidator = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(8)
        
    
    //val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]

    
    
    //Split data into training and test
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()
    
    
   
    //print rmse of our model
    //do prediction - print first k
  }
}