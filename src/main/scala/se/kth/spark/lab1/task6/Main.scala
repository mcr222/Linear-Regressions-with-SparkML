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

object Main {
  def main(args: Array[String]) {
//    val v = VectorHelper.dot(Vectors.dense(2,0,1), Vectors.dense(1,2,3))
//    val d = -0.5
//    val v = VectorHelper.dot(Vectors.dense(2,0,1), d)
//    println(v)
    
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

     val min_year = 1922 
     val lShifter = new DoubleUDF((x:Double) => x-min_year).setInputCol("label").setOutputCol("label_shifted")
     
     
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3))
    
     
     // Linear regression related transformations ------------------------
    //val myLR = new LinearRegression().setElasticNetParam(0.1).setRegParam(0.9).setMaxIter(50).setLabelCol("label_shifted").setFeaturesCol("features")
   
     val myLR = new MyLinearRegressionImpl()
        .setLabelCol("label_shifted")
        .setFeaturesCol("features")
    
     val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    
    //Split data into training and test
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()
    
    //this will show 6 rows of data, but not transformed
    //since this will be done after going through the pipeline
    train.show(6)
    val lrStage = 6
    
    val pipelineModel: PipelineModel = pipeline.fit(train)
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]
  
    println("Final rmse: " + myLRModel.trainingError(99))
//    println("All rmse:")
//    myLRModel.trainingError.foreach(println)

    //TODO: print rmse of validation set!!
    
//   Predef println(
//       "Root mean squared error:" + trainingSummary.rootMeanSquaredError +
//       "\n Mean squared error " + trainingSummary.meanSquaredError + 
//       "\n Mean absolute error " + trainingSummary.meanAbsoluteError)
    
    //do prediction - print first k
   // val result = pipelineModel.transform(test)
   // result.show(10)
    
  }
}