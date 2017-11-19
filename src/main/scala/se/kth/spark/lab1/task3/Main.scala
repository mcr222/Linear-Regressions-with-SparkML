package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vector
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

    //Pipeline from previous task ----------------------------------------------------------
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
    
     //In this case, we use another strategy to find the minimum. The strategy stated in task 2 forces to call the transform
     //function to every transformation, which is not optimal as the pipeline is already calling each of them in an optimal way (DAG construction)
     val min_year =reg_resp.map(x => x.getList(1).get(0).toString().toDouble).reduce((a,b)=> Math.min(a, b))
     
     //To be more efficient we can just hardcode the minimum year
     //val min_year = 1922
     val lShifter = new DoubleUDF((x:Double) => x-min_year.toString().toDouble).setInputCol("label").setOutputCol("label_shifted")
     
     
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3))
    
     /*
      CHOSEN! Results with 0.9-50:
          Root mean squared error:17.273541228273967
     			Mean squared error 298.3752265648805
     			Mean absolute error 14.028234944393072
     	Results with 0.9-10:
     			Root mean squared error:17.386555834410043
         	Mean squared error 302.2923237830579
         	Mean absolute error 14.12238425479688
     	Results with 0.1-50:
     			Root mean squared error:17.391970372286156
 					Mean squared error 302.4806334304795
 					Mean absolute error 14.079566748582169
     	Results with 0.1-10:
     			Root mean squared error:17.485201911624113
 					Mean squared error 305.7322858902636
 					Mean absolute error 14.194679673140822
     */    

     // Linear regression related transformations -----------------------------------------------------------------
    val myLR = new LinearRegression().setElasticNetParam(0.1).setRegParam(0.9).setMaxIter(50)
      .setLabelCol("label_shifted")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    
    //Split data into training and test. 
    // 80% of our data will be for training and 20% will be for testing
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache() //In case we test with big amounts of data we can remove .cache()
    val test = splits(1).cache()
    
    
    val pipelineModel: PipelineModel = pipeline.fit(train)
    //with this we are getting stage 6 of the pipeline (our linear regression), and casting (asInstanceOf) it to a LinearRegressionModel
    val lrModel = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
   val trainingSummary = lrModel.summary
   Predef println(
       "Root mean squared error:" + trainingSummary.rootMeanSquaredError +
       "\n Mean squared error " + trainingSummary.meanSquaredError + 
       "\n Mean absolute error " + trainingSummary.meanAbsoluteError + "\n")
    
    //do prediction - print first k
    val result = pipelineModel.transform(test)
    
     Predef println("Transformed data and some predictions: ---------------------------------------------------------------------")
    result.drop("value", "tokens", "tokens_vector", "year").show(10)
    
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label_shifted")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(result)
    println(s"Root-mean-square error on validation data = $rmse")
  
  }
}