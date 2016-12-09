/**
  * Created by akhil0 on 10/26/16.
  */

import java.lang.System

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.impurity.{Gini, Variance}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}

import scala.collection.mutable
import scala.collection.mutable.HashSet

object Test2 {

  // Checks if the String is actually a number
  def isAllDigits(x: String) = x forall Character.isDigit


  // parse the data to convert into Features Array
  def parseData(line: String, columnsSet: mutable.HashSet[Int]) = {
    val fields = line.split(",")
    if(!fields(0).equals("SAMPLING_EVENT_ID") && !fields(26).equals("?") && !fields(26).equals("X")) {
      var index = 0
      val features = Array.ofDim[Double](columnsSet.size)
      var arrayIndex = 1
      features(0) = if (fields(26).toInt > 0)  1 else 0
      var keep = true
      fields.foreach(col => {
        if (columnsSet.contains(index) && keep && index != 26) {
          if((col.trim.equals("?") || col.trim.equals("X")) && (index != 25)){
            keep = false
          }
          else{
            features(arrayIndex) = col.toDouble
          }
          arrayIndex += 1
        }
        index = index + 1
      })

      if(keep.equals(true)) {LabeledPoint(features(0), Vectors.dense(features.tail))} else {null}
    }else{null}}



  def predictLabel(point: LabeledPoint, decisionTreeModel: DecisionTreeModel, logisticRegressionModel: GradientBoostedTreesModel,
                   randomForestModel: RandomForestModel, gradientBoostModel: LogisticRegressionModel) = {
    val decisionTreePrediction = decisionTreeModel.predict(point.features)
    val logisticRegressionPrediction = logisticRegressionModel.predict(point.features)
    val randomForestPrediction = randomForestModel.predict(point.features)
    val gradientBoostPrediction = gradientBoostModel.predict(point.features)

    // Add weights in future for each model to do weighted mean.
    var avgPrediction = (decisionTreePrediction + logisticRegressionPrediction +
      randomForestPrediction + gradientBoostPrediction)/4
    if(avgPrediction < 0.50)
      avgPrediction = 0
    else
      avgPrediction = 1
    (point.label, avgPrediction)
  }

  def main(args: Array[String]) {


    // Spark configuration
    val conf = new SparkConf()
    conf.setAppName("Training")
    conf.setMaster("local")
    val sc = new SparkContext(conf)

    // val month = 5, 01 -12
    //val bcr = 1-37
    // val OMERNIK_L3_ECOREGION = 962, 1-120

    // Load the bz2files into RDD
    val inputRDD = sc.textFile(args(0))

    // Go through each line and map with lambda as Parser provided


    val columnsSet = new mutable.HashSet[Int]()

    val ColumnsRDD = sc.textFile(args(1))

    ColumnsRDD.collect().foreach(value => columnsSet.add(value.split('#')(0).toInt))

    //print(columnsSet.size)


    val parsedData =  inputRDD.map(line => parseData(line, columnsSet)).filter(x=> x!=null).foreach(line => println(line))



      sc.stop()
  }


}