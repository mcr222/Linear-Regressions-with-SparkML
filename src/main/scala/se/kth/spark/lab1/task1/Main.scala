package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

case class Song(year: Integer, f1: Double, f2: Double, f3: Double)


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
  

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    //Delimiter: ","
    //Number of features:13
    //Data type: double
    rdd.take(5).foreach(println)
    
    
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x => x.split(","))
    //recordsRdd.take(5).foreach((x => x.foreach(println)))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(s=> Song(s(0).toDouble.toInt, s(1).toDouble, s(2).toDouble, s(3).toDouble))

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()
    
    
    // Questions to solve: ---------------------------------------------------------------------------
    songsDf.registerTempTable("song")
    
    //1. How many songs are there in the DataFrame 
    //RDD function
    val x = songsRdd.count()
    Predef println("Number of songs in the dataset: " + x)
    //SQL
    sqlContext.sql("select count(*) from song").show()
    
    
   //2. How many songs were released between 1998 and 2000
    //RDD function
   val x2=songsRdd.filter(x => x.year>=1998 && x.year<=2000 ).count()
   Predef println("Number of songs between 1998 and 2000: " + x2)
    //SQL
   sqlContext.sql("select count(*) from song where year>=1998 and year<=2000").show()
   
   
   //3. Max, min and mean value of year column
   //RDD function
   val x3=songsRdd.map(x => x.year)
   val x3_max = x3.max()
   val x3_min = x3.min()
   val x3_avg = x3.reduce(_+_)/x3.count()
   Predef println("MaxYear: " + x3_max + " MinYear: " + x3_min + " AvgYear: " + x3_avg)
   //SQL
   sqlContext.sql("select max(year), min(year), avg(year) from song").show()
   
   
   //4. Number of songs per year between 2000 and 2010
   //RDD Function
   val x4 = songsRdd.filter(x => x.year>=2000 && x.year<=2010 ).map(x => (x.year, 1)).reduceByKey(_ + _)
   
   //can just use x4.foreach(println)
   x4.take(11).foreach(println)
   //alternative:
   //    println(songsRdd.filter(song => song.year>=2000 && song.year<=2010).map(_.year).countByValue())
   
   //SQL
   sqlContext.sql("select year, count(*) as numberSongs from song where year between 2000 and 2010 group by year").show()

  }
}