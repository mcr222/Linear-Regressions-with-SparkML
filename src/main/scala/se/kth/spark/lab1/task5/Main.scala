package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.sql.{ SQLContext, DataFrame }
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import se.kth.spark.lab1._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder



object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sc.textFile(filePath).toDF()

    //Pipeline creation ----------------------------------------------------------------------------------------------
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
    
     //Note that polyfeatures will include all features (3) and all combinations of 2nd degree
     //(6 additional features)
     val polynomialExpansionT = new PolynomialExpansion()
        .setInputCol("features")
        .setOutputCol("polyfeatures")
        .setDegree(2) 
    
     
     val myLR = new LinearRegression().setElasticNetParam(0.1).setLabelCol("label_shifted").setFeaturesCol("polyfeatures")
   
     //Set the stages
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, polynomialExpansionT, myLR))
   
    
    //Hyperparameter related operations
    //Build the parameter grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.maxIter, Array(20, 30, 40, 50, 60, 70, 80))
      .addGrid(myLR.regParam, Array(0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4))
      .build()
    
    
    val evaluator = new RegressionEvaluator
    
    //Split data into training and test
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache() //In case we test with big amounts of data we can remove .cache()
    val test = splits(1).cache() 
    
    //Cross validation
    val cvModel: CrossValidator = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3) //For simplicity we use 3 folds, but any number can be used

    val c = cvModel.fit(train)
    val lrModel = c.bestModel.asInstanceOf[PipelineModel].stages(7).asInstanceOf[LinearRegressionModel]
   
    //print rmse of our model
   val trainingSummary = lrModel.summary
   Predef println(
       "Root mean squared error:" + trainingSummary.rootMeanSquaredError +
       "\n Mean squared error " + trainingSummary.meanSquaredError + 
       "\n Mean absolute error " + trainingSummary.meanAbsoluteError + "\n")
   
    //do prediction - print first k
    val result = c.transform(test)
    Predef println("Transformed data and some predictions ---------------------------------------------------------------------")
    result.drop("value", "tokens", "tokens_vector", "year").show(10)
    
    val eval = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label_shifted")
      .setPredictionCol("prediction")
    val rmse = eval.evaluate(result)
    println(s"Root-mean-square error on validation data = $rmse")
  }
}