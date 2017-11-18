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


object Main {
  
  def getFeatures(reader: IHDF5Reader): Array[String] = {
     val metadata = HDF.getCompoundDS(reader,"/metadata/songs")
     val mb = HDF.getCompoundDS(reader,"/musicbrainz/songs")
     val analysis = HDF.getCompoundDS(reader,"/analysis/songs")
     
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
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    
    val path = "/home/mcr222/Documents/EIT/KTH/Scalable Machine Learning/Linear Regressions with SparkML/vt17-lab1/src/main/resources/"
 
    var tarFiles: Array[String] = Array()
    val tar_path = path + "song-test.tar.gz"
    tarFiles = tarFiles :+ tar_path
    var allHDF5 = sc.parallelize(tarFiles).flatMap(path => {
        val tar = new TarArchiveInputStream(new GzipCompressorInputStream(new FileInputStream(path)))
        var entry: TarArchiveEntry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
        var res: List[Array[Byte]] = List()
        while (entry != null) {
            var outputFile:File = new File(entry.getName());
            if (!entry.isDirectory() && entry.getName.contains(".h5")) {
                var byteFile = Array.ofDim[Byte](entry.getSize.toInt)
                tar.read(byteFile);
                res = byteFile :: res
            }
            entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
        }
        res
        
      } ).map(bytes => {
         FileUtils.writeByteArrayToFile(new File(bytes.toString()), bytes)
         val reader = HDF5Factory.openForReading(bytes.toString())
         getFeatures(reader)
      })
      
      allHDF5.foreach(x => { x.foreach(y => print(y+" "))
                            println()})
                            
      var obsDF = allHDF5.filter(x => x(0)!="0.0").toDF()
      obsDF.printSchema()
      println(obsDF.count())
      
      obsDF.show(3)
    
      
     val arr2Vect = new Array2Vector()
        .setInputCol("value")
         .setOutputCol("tokens_vector")
     
     val lSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("year")
    lSlicer.setIndices(Array(0))

     val v2d = new Vector2DoubleUDF((x: Vector) => x(0).toDouble).setInputCol("year").setOutputCol("label")

     val min_year = 1922
     val lShifter = new DoubleUDF((x:Double) => x-min_year).setInputCol("label").setOutputCol("label_shifted")
     
     
     val fSlicer = new VectorSlicer().setInputCol("tokens_vector").setOutputCol("features")
     fSlicer.setIndices(Array(1,2,3))

     // Linear regression related transformations ------------------------
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
  
  }
}



