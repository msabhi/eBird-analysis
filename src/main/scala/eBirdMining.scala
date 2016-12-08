/**
  * Created by akhil0 on 10/26/16.
  */

import java.lang.System

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest, DecisionTree}
import org.apache.spark.mllib.tree.impurity.{Variance, Gini}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}

import scala.collection.mutable
import scala.collection.mutable.HashSet

object eBirdMining {

  def isAllDigits(x: String) = x forall Character.isDigit

  def doClassification(_1: Int, _2: RDD[LabeledPoint]) = {

    var categoricalFeaturesInfo = Map[Int, Int]()
    categoricalFeaturesInfo += (2 -> 31)
    categoricalFeaturesInfo += (3 -> 366)
    categoricalFeaturesInfo += (7 -> 38)
    categoricalFeaturesInfo += (8 -> 121)

    _1 match {

    case 0 =>
    DecisionTree.trainClassifier(_2, 4, categoricalFeaturesInfo, "gini", 9, 7000)
      case 1 =>
    RandomForest.trainClassifier(_2, Strategy.defaultStrategy("Classification"), 4, "auto", 12345)
      case 2 =>
     GradientBoostedTrees.train(_2, BoostingStrategy.defaultParams("Classification"))
      case 3 => new LogisticRegressionWithLBFGS().setNumClasses(10).run(_2)
   }
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


    var columnsSet = new mutable.HashSet[Int]()

    val ColumnsRDD = sc.textFile(args(1))

    ColumnsRDD.collect().foreach(value => columnsSet.add(value.split('#')(0).toInt))

    //print(columnsSet.size)


    val parsedData =  inputRDD.map(line => {
      val fields = line.split(",")
      if(!fields(0).equals("SAMPLING_EVENT_ID") && !fields(26).equals("?") && !fields(26).equals("X")) {
        var index = 0
        var features = Array.ofDim[Double](columnsSet.size)
        var arrayIndex = 1
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

        if(keep.equals(true)) {LabeledPoint(features(0), Vectors.dense(features.tail))} else {null}
      }else{null}}).filter(x=> x!=null).persist()

    //parsedData.foreach(x=>println(x))
    println(parsedData.count())


    //MLUtils.loadLibSVMFile()
    val splits = parsedData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //trainingData.foreach(x => println(x))
    //println(trainingData.count())

    //val decisionTreeRDD = trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=0 || x._1 == null).map(x => x._2).persist()
    //val logisticRegressionRDD = trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=1).map(x=>x._2)
    //val randomForestRdd = trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=2).map(x=>x._2)
    //val gradientBoostRdd = trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=3).map(x=>x._2)

    var categoricalFeaturesInfo = Map[Int, Int]()
    categoricalFeaturesInfo += (2 -> 31)
    categoricalFeaturesInfo += (3 -> 366)
    categoricalFeaturesInfo += (7 -> 38)
    categoricalFeaturesInfo += (8 -> 121)

    val decisionTreeModel = DecisionTree.trainClassifier(trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=0 || x._1 == null).map(x => x._2), 4, categoricalFeaturesInfo, "gini", 9, 7000)//doClassification(0, trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=0 || x._1 == null).map(x => x._2))
    val randomForestModel = RandomForest.trainClassifier(trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=1 || x._1 == null).map(x => x._2), Strategy.defaultStrategy("Classification"), 4, "auto", 12345)//doClassification(0, trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=1 || x._1 == null).map(x => x._2))
    val logisticRegressionModel = GradientBoostedTrees.train(trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=2 || x._1 == null).map(x => x._2), BoostingStrategy.defaultParams("Classification"))//doClassification(0, trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=2 || x._1 == null).map(x => x._2))
    val gradientBoostModel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=3 || x._1 == null).map(x => x._2))//doClassification(0, trainingData.map(line => (scala.util.Random.nextInt(4), line)).filter(x => x._1!=3 || x._1 == null).map(x => x._2))

    //.reduceByKey((a,b) => a++b)
    //.map(a => doClassification(a._1, sc.parallelize(a._2)))

    /*
    val decisionTreePreds = testData.map { point =>
      val prediction = decisionTreeModel.predict(point.features)
      (point.label, prediction)
    }

    val logisticRegressionPreds = testData.map { point =>
      val prediction = logisticRegressionModel.predict(point.features)
      (point.label, prediction)
    }

    val randomForestPreds = testData.map { point =>
      val prediction = randomForestModel.predict(point.features)
      (point.label, prediction)
    }

    val gradientBoostPreds = testData.map { point =>
      val prediction = gradientBoostModel.predict(point.features)
      (point.label, prediction)
    }
    */

    val EnsemblePredictionRDD = testData.map { point =>
      val decisionTreePrediction = decisionTreeModel.predict(point.features)
      val logisticRegressionPrediction = logisticRegressionModel.predict(point.features)
      val randomForestPrediction = randomForestModel.predict(point.features)
      val gradientBoostPrediction = gradientBoostModel.predict(point.features)
      var avgPrediction = (decisionTreePrediction + logisticRegressionPrediction + randomForestPrediction + gradientBoostPrediction)/4
      if(avgPrediction < 0.50)
        avgPrediction = 0
      else
        avgPrediction = 1
      (point.label, avgPrediction)

    }

    /*
    val decisionTreeAccuracy = decisionTreePreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
    val logisticRegressionAccuracy = logisticRegressionPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
    val randomForestAccuracy = randomForestPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
    val gradientBoostAccuracy = gradientBoostPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()



    println("Decision Tree Test Accuracy = " + decisionTreeAccuracy)
    println("Logistic Regression Test Accuracy = " + logisticRegressionAccuracy)
    println("Random Forest Test Accuracy = " + randomForestAccuracy)
    println("Gradient Boost Test Accuracy = " + gradientBoostAccuracy)
    */

    val finalAccuracy = EnsemblePredictionRDD.filter(r => r._1 == r._2).count.toDouble / testData.count()
    println("Accuracy = " + finalAccuracy)




    sc.stop()
  }


}