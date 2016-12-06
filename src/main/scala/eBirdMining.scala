/**
  * Created by akhil0 on 10/26/16.
  */

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.{SparkConf, SparkContext}


import scala.collection.mutable
import scala.collection.mutable.HashSet

object eBirdMining {

  def isAllDigits(x: String) = x forall Character.isDigit


  def doClassification(_1: Int, _2: Array[Unit]) = _1 match {

    case 0 => 0
    case 1 => 1
    case 2 => 2
    case 3 => 3

  }

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




    //    val parsedData = inputRDD.map(line => {
    //      val fields = line.split(",")
    //      var sb = ""
    //      var index = 0
    //      var features = Array.ofDim[Double](columnsSet.size)
    //      println("SIZE =>" + columnsSet.size + " " + features.length)
    //      var arrayIndex = 0
    //      fields.foreach(col => {
    //        if (columnsSet.contains(index)) {
    //          if (isAllDigits(col)) {
    //            features(arrayIndex) = col.toDouble
    //          }
    //          else{
    //            println("INDEX =>" + arrayIndex)
    //            features(arrayIndex) = 1
    //          }
    //          arrayIndex = arrayIndex + 1
    //        }
    //        index = index + 1
    //      })
    //      LabeledPoint(features(0), Vectors.dense(features.tail))
    //    })

    val parsedData =  inputRDD.map(line => {
      val fields = line.split(",")
      var sb = ""
      var mainPoint = "";
      var index = 0
      fields.foreach(col => {
        if (columnsSet.contains(index)) {
          if (isAllDigits(col) && col.toInt > 0 || !col.equals("")) {
            if(index == 26 && !col.equals("Agelaius_phoeniceus")){
              if(isAllDigits(col) && col.toInt > 0)
                mainPoint = "1"
              else
                mainPoint = "0"
            }
            else
              sb += col + "#"
          }
        }
        index = index + 1
      })
      sb = mainPoint + "#" + sb.substring(0, sb.length-1)
      //(scala.util.Random.nextInt(4), sb)
    }).persist()


    val splits = parsedData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //println(trainingData.count())
    trainingData.map(line => (scala.util.Random.nextInt(4), Array(line)))
      .reduceByKey((a,b) => a++b)
      .map(a => doClassification(a._1, a._2))



    //parsedData.keys.foreach(x => println(x))

    sc.stop()
  }


}