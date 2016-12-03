/**
  * Created by akhil0 on 10/26/16.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.HashSet

object eBirdMining {

  def isAllDigits(x: String) = x forall Character.isDigit

  def main(args: Array[String]) {

    // Spark configuration
    val conf = new SparkConf()
    conf.setAppName("Training")
    conf.setMaster("local")
    val sc = new SparkContext(conf)

    // Load the bz2files into RDD
    val inputRDD = sc.textFile(args(0))

    // Go through each line and map with lambda as Parser provided


    var columnsSet = new mutable.HashSet[Int]()

    val ColumnsRDD = sc.textFile(args(1))

    ColumnsRDD.collect().foreach(value => columnsSet.add(value.split('#')(0).toInt))

    print(columnsSet.size)


    inputRDD.map(line => {
      val fields = line.split(",")
      var sb = ""
      var index = 0
      fields.foreach(col => {
        if (columnsSet.contains(index)) {
          if (isAllDigits(col) && col.toInt > 0 || !col.equals("")) {
            sb += index + ":" + col + "#"
          }
        }
        index = index + 1
      })
      sb = sb.substring(0, sb.length-1)
      sb
    }).foreach(x => println(x))

    sc.stop()
  }


}