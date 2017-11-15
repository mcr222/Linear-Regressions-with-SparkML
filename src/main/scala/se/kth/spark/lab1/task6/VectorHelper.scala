package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import scala.collection.breakOut

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
   val v1_array = v1.toArray
   val v2_array = v2.toArray

   (v1_array zip v2_array).map{ Function.tupled(_ * _)}.sum
  }

  def dot(v: Vector, s: Double): Vector = {
    ???
  }

  def sum(v1: Vector, v2: Vector): Vector = {
   
    val t = Vectors.dense((v1.toArray, v2.toArray).zipped.map(_ + _))
    t
  }

  def fill(size: Int, fillVal: Double): Vector = {
    ???
  }
  
}