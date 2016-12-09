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
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}

import scala.collection.mutable


object eBirdMining {

  // Checks if the String is actually a number
  def isAllDigits(x: String) = x forall Character.isDigit


  // parse the data to convert into Features Array
  def parseTrainingData(line: String, columnsSet: mutable.HashSet[Int]) = {
    val fields: Array[String] = line.split(",")
    if(!fields(0).equals("SAMPLING_EVENT_ID") && !fields(26).equals("?") && !fields(26).equals("X")) {
      var index: Int = 0
      val features: Array[Double] = Array.ofDim[Double](columnsSet.size)
      var arrayIndex: Int = 1
      features(0) = if (fields(26).toInt > 0)  1 else 0
      var keep = true
      fields.foreach(col => {
        if (columnsSet.contains(index) && keep && index != 26) {
          if(col.trim.equals("?") || col.trim.equals("X")){
            keep = false
          }
          else{
            features(arrayIndex) = col.toDouble
          }
          arrayIndex += 1
        }
        index = index + 1
      })

      LabeledPoint(features(0), Vectors.dense(features.tail))
    }else{null}}

  // parse the data to convert into Features Array
  def parseTestingData(line: String, columnsSet: mutable.HashSet[Int]) = {
    val fields: Array[String] = line.split(",")
    if(!fields(0).equals("SAMPLING_EVENT_ID")) {
      var index: Int = 0
      val features: Array[Double] = Array.ofDim[Double](columnsSet.size)
      var arrayIndex: Int = 1
      features(0) = fields(0).substring(1,fields(0).length).toDouble
      fields.foreach(col => {
        if (columnsSet.contains(index) && index != 26) {
          if(col.trim.equals("?") || col.trim.equals("X")){
              features(arrayIndex) = 0.0
          }
          else{
            features(arrayIndex) = col.toDouble
          }
          arrayIndex += 1
        }
        index = index + 1
      })

      LabeledPoint(features(0), Vectors.dense(features.tail))
    }else{null}}



  def predictLabel(point: LabeledPoint, decisionTreeModel: DecisionTreeModel, logisticRegressionModel: GradientBoostedTreesModel,
                   randomForestModel: RandomForestModel, gradientBoostModel: LogisticRegressionModel) = {
    val decisionTreePrediction: Double = decisionTreeModel.predict(point.features)
    val logisticRegressionPrediction: Double = logisticRegressionModel.predict(point.features)
    val randomForestPrediction: Double = randomForestModel.predict(point.features)
    val gradientBoostPrediction: Double = gradientBoostModel.predict(point.features)

    // Add weights in future for each model to do weighted mean.
    var avgPrediction: Double = (decisionTreePrediction + logisticRegressionPrediction +
      randomForestPrediction + gradientBoostPrediction)/4
    if(avgPrediction < 0.50)
      avgPrediction = 0
    else
      avgPrediction = 1
    (point.label, avgPrediction)
  }

  def main(args: Array[String]) {


    // Spark configuration
    val conf : SparkConf= new SparkConf()
    conf.setAppName("Training")
    conf.setMaster("local")
    val sc : SparkContext = new SparkContext(conf)
    val inputRDD : RDD[String] = sc.textFile(args(0))

    // Go through each line and map with lambda as Parser provided


    val columnsSet : mutable.HashSet[Int] = new mutable.HashSet[Int]()
    val columns = Array[Int] (2,
      3,
      5,
      6,
      12,
      13,
      14,
      16,
      26,
      955,
      960,
      962,
      963,
      964,
      965,
      966,
      967)

    columns.foreach(value => columnsSet.add(value))

    val parsedData : RDD[LabeledPoint] =  inputRDD.map(line => parseTrainingData(line, columnsSet)).filter(x=> x!=null).persist()

    val splits : Array[RDD[LabeledPoint]] = parsedData.randomSplit(Array(0.7, 0.3))

    val (trainingData : RDD[LabeledPoint], testData : RDD[LabeledPoint]) = (splits(0), splits(1))


    var categoricalFeaturesInfo : Map[Int, Int] = Map[Int, Int]()
    categoricalFeaturesInfo += (2 -> 31)
    categoricalFeaturesInfo += (3 -> 367)
    categoricalFeaturesInfo += (9 -> 38)
    categoricalFeaturesInfo += (10 -> 121)

    val randomTrainingData : RDD[(Int, LabeledPoint)] = trainingData.map(line => (scala.util.Random.nextInt(4), line))
    val decisionTreeModel : DecisionTreeModel= DecisionTree.trainClassifier(randomTrainingData.filter(x => x._1!=0 || x._1 == null).map(x => x._2), 4, categoricalFeaturesInfo, "gini", 9, 7000)
    val randomForestModel : RandomForestModel = RandomForest.trainClassifier(randomTrainingData.filter(x => x._1!=1 || x._1 == null).map(x => x._2), Strategy.defaultStrategy("Classification"), 4, "auto", 12345)
    val gradientBoostModel : GradientBoostedTreesModel = GradientBoostedTrees.train(randomTrainingData.filter(x => x._1!=2 || x._1 == null).map(x => x._2), BoostingStrategy.defaultParams("Classification"))
    val logisticRegressionModel : LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(randomTrainingData.filter(x => x._1!=3 || x._1 == null).map(x => x._2))




    val EnsembleValidationRDD : RDD[(Double, Double)]= testData.map(point => predictLabel(point, decisionTreeModel,  gradientBoostModel , randomForestModel,logisticRegressionModel))
    val finalAccuracy = EnsembleValidationRDD.filter(r => r._1 == r._2).count.toDouble / testData.count()

    println("Accuracy = " + finalAccuracy)


    val predictionInput : RDD[String] = sc.textFile(args(1))
    val predictionData : RDD[LabeledPoint]= predictionInput.map(line => parseTestingData(line, columnsSet)).filter(x=> x!=null).persist()


    val EnsemblePredictionRDD : RDD[(Double, Double)]= predictionData.map { point => predictLabel(point, decisionTreeModel, gradientBoostModel, randomForestModel, logisticRegressionModel)}

    EnsemblePredictionRDD.map(line=> "S"+line._1.toInt + "," + line._2).saveAsTextFile(args(2))

    sc.stop()
  }


}