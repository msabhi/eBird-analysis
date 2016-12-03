/**
  * Created by akhil0 on 10/26/16.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.HashSet

object eBirdMining {

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

    ColumnsRDD.foreach(k => println(k))


    ColumnsRDD.map(line => line.split(", ")).map(fields => fields(1)).persist()

    ColumnsRDD.collect().foreach(value => columnsSet.add(value.toInt))

    //print(columnsSet.size)


    inputRDD.map(f => f.split(","))
      .map(fields => {
        var sb = "";
        var index = 0;
        fields.foreach(f => {
          if(columnsSet.contains(index))
            sb += "(" + index + ":" + f + ");"
          index = index + 1
        })

        println(sb)

      })


    sc.stop()
  }


}