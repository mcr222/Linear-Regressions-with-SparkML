package se.kth.spark.lab1.bonustask

import se.kth.spark.lab1._

import ch.systemsx.cisd.hdf5._
import ch.systemsx.cisd.hdf5.HDF5Factory

import scala.util._
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.archivers.tar._
import org.apache.commons.io.FileUtils
import java.io.File
import java.io.FileInputStream


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

import org.apache.spark._
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.evaluation.RegressionEvaluator

/** IMPORTANT
 *  Also includes bonus task in new package bonustask. Bonus task is ready 
 *  to be run locally (need to change folder path to point where all tar.gz 
 *  file with .h5 files are). The execution is reduced so it can be run fast 
 *  (limits songs read from files and only 4 features are taken). See all TODOs 
 *  in bonustask.Main to see all what to change to run fully.
 * 
 */

object Main {
  
  /*
   * This function extracts all features from a .h5 file (opened with
   * a HDF5 reader, and returns an array with a string of features,
   * the first one being the year of the song
   */
  def getFeatures(reader: IHDF5Reader): Array[String] = {
     val metadata = HDF.getCompoundDS(reader,"/metadata/songs")
     val mb = HDF.getCompoundDS(reader,"/musicbrainz/songs")
     val analysis = HDF.getCompoundDS(reader,"/analysis/songs")
     
     //println(metadata.get[String]("title"))
     var features: Array[String] = Array(mb.get[Int]("year").toDouble.toString())
     features = features :+ analysis.get[Double]("danceability").toString()
     features = features :+ analysis.get[Double]("duration").toString()
     features = features :+ analysis.get[Double]("end_of_fade_in").toString()
     features = features :+ analysis.get[Double]("energy").toString()
     features = features :+ analysis.get[Double]("key_confidence").toString()
     features = features :+ analysis.get[Double]("loudness").toString()
     features = features :+ analysis.get[Double]("mode_confidence").toString()
     features = features :+ analysis.get[Double]("start_of_fade_out").toString()
     features = features :+ analysis.get[Double]("tempo").toString()
     features = features :+ analysis.get[Double]("time_signature_confidence").toString()
     features = features :+ metadata.get[Double]("artist_familiarity").toString()
     features = features :+ metadata.get[Double]("artist_hotttnesss").toString()
     features = features :+ metadata.get[Double]("artist_latitude").toString()
     features = features :+ metadata.get[Double]("artist_longitude").toString()
     features = features :+ metadata.get[Double]("song_hotttnesss").toString()
     for (i <- 0 to (features.size-1)) {
       if(features(i) == "NaN") {
          features(i) = "0" 
       }
      }
     features
  }
  
  def main(args: Array[String]) {
    //TODO: remove .setMaster() to run on cluster (hops.site)
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    
    //TODO: change your folder path
    //val path = "/demo_spark_marccr01/labs::million_song/"
    val path = "/home/mcr222/Downloads/"
    //in order for this to work the file millionsongsubset should be added
    //to resources or some other path. It will also work for the full
    //dataset, it is needed to add the path to all 
    //var path = "src/main/resources/"
    val f:File = new File(path);
    if(f.exists()) { 
        print("Reading files from path: ")
        println(path)
    } else {
      println("Reading path not found")
      return
    }
    var tarFiles: Array[String] = Array()
    val tar_path = path + "millionsongsubset.tar.gz"
    
    //TODO: add all your tar.gz files in main folder path to tarFiles array
    //should add here as many tar.gz files as wanted containing the
    //hdf5 files for the songs
    tarFiles = tarFiles :+ tar_path
    tarFiles = tarFiles :+ tar_path
    //tarFiles = tarFiles :+ (path+"A.tar.gz")
    //tarFiles = tarFiles :+ (path+"B.tar.gz")
    //tarFiles = tarFiles :+ (path+"C.tar.gz")
    
    //This reads all tar.gz files in tarFiles list, and for each .h5
    //file within, it extracts each song's list of features
    //Thus, it gets a list of features for all songs in the files.
    var allHDF5 = sc.parallelize(tarFiles).flatMap(path => {
        val tar = new TarArchiveInputStream(new GzipCompressorInputStream(new FileInputStream(path)))
        var entry: TarArchiveEntry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
        var res: List[Array[Byte]] = List()
        var i = 0
        //TODO: remove condition i<101 to read completely all files
        while (entry != null && i<101) {
            var outputFile:File = new File(entry.getName());
            if (!entry.isDirectory() && entry.getName.contains(".h5")) {
                var byteFile = Array.ofDim[Byte](entry.getSize.toInt)
                tar.read(byteFile);
                res = byteFile :: res
                if(i%100==0) {
                  println("Read " + i + " files")
                }
                i = i+1
                    
            }
            entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
        }
        res
        
      } ).map(bytes => {
        // The toString method for class Object returns a string consisting of the name
        //of the class of which the object is an instance, the at-sign character `@', and 
        //the unsigned hexadecimal representation of the hash code of the object.
        //It should be a UID
         val name = bytes.toString()
         FileUtils.writeByteArrayToFile(new File(name), bytes)
         val reader = HDF5Factory.openForReading(name)
         val features = getFeatures(reader)
         reader.close()
         features
      })
      
      println("Extracted songs from tar.gz, showing 5 examples")
      allHDF5.take(5).foreach(x => { x.foreach(y => print(y+" "))
                           println()})
      
      //There are songs that have no year, these we won't use to predict
      var obsDF = allHDF5.filter(x => x(0)!="0.0").toDF()
      obsDF.printSchema()
      println(obsDF.count())
      
      obsDF.show(10)
    
      //We use the pipeline, without needing the tokenizer since our results
      //are in an array already
     val arr2Vect = new Array2Vector()
        .setInputCol("value")
         .setOutputCol("tokens_vector")
     
     val lSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("year")
    lSlicer.setIndices(Array(0))

     val v2d = new Vector2DoubleUDF((x: Vector) => x(0).toDouble).setInputCol("year").setOutputCol("label")

     val min_year = 1922
     val lShifter = new DoubleUDF((x:Double) => x-min_year).setInputCol("label").setOutputCol("label_shifted")
     
     //For testing purposes this is kept in the pipeline, but it can be removed in order
     //to take into account all features of each song
     //TODO: remove or add as many columns as wanted in features, there are 14 features
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3,4,5))

     // Linear regression related transformations ------------------------
     //We use the optimal parameters found in task 4
    val myLR = new LinearRegression().setElasticNetParam(0.1).setRegParam(1.2).setMaxIter(20)
      .setLabelCol("label_shifted")
      .setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    
    //Split data into training and test
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()
        
    val pipelineModel: PipelineModel = pipeline.fit(train)
    //with this we are getting stage 6 of the pipeline (our linear regression),
    //and casting (asInstanceOf) it to a LinearRegressionModel
    val lrModel = pipelineModel.stages(5).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
   val trainingSummary = lrModel.summary
   Predef println(
       "Root mean squared error:" + trainingSummary.rootMeanSquaredError +
       "\n Mean squared error " + trainingSummary.meanSquaredError + 
       "\n Mean absolute error " + trainingSummary.meanAbsoluteError)
    
    //do prediction - print first k
    val result = pipelineModel.transform(test)
    result.drop("value", "tokens", "tokens_vector", "year").show(10)
    
    //evaluating on validation data
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label_shifted")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(result)
    println(s"Root-mean-square error on validation data = $rmse")
  
  }
}



