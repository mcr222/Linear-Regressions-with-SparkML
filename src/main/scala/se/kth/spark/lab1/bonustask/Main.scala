package se.kth.spark.lab1.bonustask

import hdf.hdf5lib.H5
import hdf.hdf5lib.HDF5Constants
import ch.systemsx.cisd.hdf5
import ch.systemsx.cisd.hdf5._
import ch.systemsx.cisd.hdf5.HDF5Factory
import ch.systemsx.cisd.hdf5.io.HDF5DataSetRandomAccessFile
import ch.systemsx.cisd.base.io.IRandomAccessFile
import java.io.File 
import scala.util._
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.archivers.tar._
import java.io.FileInputStream
import java.io.OutputStream
import java.io.FileOutputStream
import org.apache.commons.io.IOUtils

object HDF5 {

  case class H5CDS(ds: HDF5CompoundDataMap){
    def vector[T](colName: String): Vector[T] = ds.get(colName).asInstanceOf[Array[T]].toVector
    def get[T](colName: String): T = ds.get(colName).asInstanceOf[T]
  }

  def getCompoundDS(h5: IHDF5Reader, path: String): H5CDS = {
    val data = h5.compound().read(path, classOf[HDF5CompoundDataMap])
    H5CDS(data)
  }

  def open(filename: String): Try[IHDF5Reader] = Try{HDF5FactoryProvider.get().openForReading(new File(filename))}

  def close(h5: IHDF5Reader): Try[Unit] = Try{h5.close()}

  // read a vector from array of vectors
  def readArray[T](data: Vector[T], offset: Vector[Int], i: Int): Vector[T] = {
    val j0: Int = offset(i)
    val j1: Int = offset.lift(i+1).getOrElse(data.length)
    data.slice(j0,j1)
  }

}

object Main {
  
  def main(args: Array[String]) {
    println("ahaaha")
    val path = "/home/mcr222/Documents/EIT/KTH/Scalable Machine Learning/Linear Regressions with SparkML/vt17-lab1/src/main/resources/"
    val test_path = path + "test.h5"
    val reader = HDF5Factory.openForReading(test_path)
    println(reader.getStringAttribute("metadata/songs","FIELD_1_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_2_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_3_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_4_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_5_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_6_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_7_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_8_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_9_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_10_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_11_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_12_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_13_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_14_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_15_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_16_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_17_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_18_NAME"))
    println(reader.getStringAttribute("metadata/songs","FIELD_19_NAME"))
//    println(reader.getStringAttribute("metadata/songs","FIELD_19_FILL"))
    
    
    
//    var mydata = reader.getAllGroupMembers("analysis")
    var mydata = reader.getAllAttributeNames("musicbrainz/songs")
     println(reader.getStringAttribute("musicbrainz/songs","FIELD_0_NAME"))
      println(reader.getStringAttribute("musicbrainz/songs","FIELD_1_NAME"))
    println(mydata)
    
    println("-------------")
    println (reader.readDoubleArray("/analysis/segments_loudness_start").toVector)
    
    val metadata = HDF5.getCompoundDS(reader,"/metadata/songs")
    val mb = HDF5.getCompoundDS(reader,"/musicbrainz/songs")
    println (mb.get[Int]("year"))
    println(metadata.get[String]("title"))
    reader.close()
    
    var untaredFiles: Array[String] = Array()
    val tar_path = path + "song-test.tar.gz" 
    val tar = new TarArchiveInputStream(new GzipCompressorInputStream(new FileInputStream(tar_path)))
    var entry: TarArchiveEntry = null; 
    entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
    while (entry != null) {
        var outputFile:File = new File(path, entry.getName());
        if (entry.isDirectory()) {
            println(String.format("Attempting to write output directory %s.", outputFile.getAbsolutePath()));
            if (!outputFile.exists()) {
                println(String.format("Attempting to create output directory %s.", outputFile.getAbsolutePath()));
                if (!outputFile.mkdirs()) {
                    throw new IllegalStateException(String.format("Couldn't create directory %s.", outputFile.getAbsolutePath()));
                }
            }
        } else {
            println(String.format("Creating output file %s.", outputFile.getAbsolutePath()));
            var outputFileStream: OutputStream = new FileOutputStream(outputFile); 
            IOUtils.copy(tar, outputFileStream);
            outputFileStream.close();
            if(outputFile.getAbsolutePath.contains(".h5")) {
              untaredFiles = untaredFiles :+ outputFile.getAbsolutePath
            }
        }
        entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
    }
    println("----------")
    untaredFiles.foreach(println)
    //TODO: can do here parallelize and then transform each .h5 into a song array
    //TODO: parallelize datasets A.tar.gz till Z.tar.gz
 //   var bytereader = HDF5DataSetRandomAccessFile.read()
    
//    var file_id = H5.H5Fcreate(test_path, HDF5Constants.H5F_ACC_TRUNC, HDF5Constants.H5P_DEFAULT,
//                    HDF5Constants.H5P_DEFAULT);
//    
//    H5.H5Dread(dataset_id, HDF5Constants.H5T_NATIVE_INT, HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
//                        HDF5Constants.H5P_DEFAULT, dset_data);
  }
}