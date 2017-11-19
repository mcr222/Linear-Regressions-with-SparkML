package se.kth.spark.lab1.bonustask

import java.io.File
import ch.systemsx.cisd.hdf5._
import scala.util._

object HDF {

  case class H5CDS(ds: HDF5CompoundDataMap){
    def vector[T](colName: String): Vector[T] = ds.get(colName).asInstanceOf[Array[T]].toVector
    def get[T](colName: String): T = ds.get(colName).asInstanceOf[T]
  }

  def getCompoundDS(h5: IHDF5Reader, path: String): H5CDS = {
    val data = h5.compound().read(path, classOf[HDF5CompoundDataMap])
    H5CDS(data)
  }

}