package se.kth.spark.lab1.test

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SQLContext, DataFrame }
import org.apache.spark.ml.PipelineModel

import org.apache.commons.io.IOUtils
import java.net.URL
import java.nio.charset.Charset

case class Bank(age: Integer, job: String, marital: String, education: String, balance: Integer)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1")//.setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    
    var data: Array[String] = Array()
    data = data :+ "\"age\";\"job\";\"marital\";\"education\";\"default\";\"balance\";\"housing\";\"loan\";\"contact\";\"day\";\"month\";\"duration\";\"campaign\";\"pdays\";\"previous\";\"poutcome\";\"y\""
    data = data :+ "30;\"unemployed\";\"married\";\"primary\";\"no\";1787;\"no\";\"no\";\"cellular\";19;\"oct\";79;1;-1;0;\"unknown\";\"no\""
    data = data :+ "33;\"services\";\"married\";\"secondary\";\"no\";4789;\"yes\";\"yes\";\"cellular\";11;\"may\";220;1;339;4;\"failure\";\"no\""
    data = data :+ "35;\"management\";\"single\";\"tertiary\";\"no\";1350;\"yes\";\"no\";\"cellular\";16;\"apr\";185;1;330;1;\"failure\";\"no\""
    data = data :+ "30;\"management\";\"married\";\"tertiary\";\"no\";1476;\"yes\";\"yes\";\"unknown\";3;\"jun\";199;4;-1;0;\"unknown\";\"no\""
    data = data :+ "59;\"blue-collar\";\"married\";\"secondary\";\"no\";0;\"yes\";\"no\";\"unknown\";5;\"may\";226;1;-1;0;\"unknown\";\"no\""
    data = data :+ "35;\"management\";\"single\";\"tertiary\";\"no\";747;\"no\";\"no\";\"cellular\";23;\"feb\";141;2;176;3;\"failure\";\"no\""
    data = data :+ "36;\"self-employed\";\"married\";\"tertiary\";\"no\";307;\"yes\";\"no\";\"cellular\";14;\"may\";341;1;330;2;\"other\";\"no\""
    data = data :+ "39;\"technician\";\"married\";\"secondary\";\"no\";147;\"yes\";\"no\";\"cellular\";6;\"may\";151;2;-1;0;\"unknown\";\"no\""
    data = data :+ "41;\"entrepreneur\";\"married\";\"tertiary\";\"no\";221;\"yes\";\"no\";\"unknown\";14;\"may\";57;2;-1;0;\"unknown\";\"no\""
    
    val bankText = sc.parallelize(data)

        bankText.take(5).foreach(println)
        
    val bank = bankText.map(s => s.split(";")).filter(s => s(0) != "\"age\"").map(
      s => Bank(s(0).toInt,
        s(1).replaceAll("\"", ""),
        s(2).replaceAll("\"", ""),
        s(3).replaceAll("\"", ""),
        s(5).replaceAll("\"", "").toInt)).toDF()
    bank.registerTempTable("bank")
    bank.show(5)
    sqlContext.sql("select age, count(1) from bank where age < 70 group by age order by age").show()
    
      
  }
}
