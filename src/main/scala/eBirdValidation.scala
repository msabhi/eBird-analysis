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


object eBirdValidation {

  // The preprocessing function to parse the training data record to convert into labeled point
  def parseTrainingData(line: String, columnsSet: mutable.HashSet[Int]) = {
    val fields: Array[String] = line.split(",")  // Split on ","

    // Ignore if its first row in the file or if the label has invalid characters
    if(!fields(0).equals("SAMPLING_EVENT_ID") && !fields(26).equals("?") && !fields(26).equals("X")) {
      var index: Int = 0
      val features: Array[Double] = Array.ofDim[Double](columnsSet.size)  // initialize a features array
      var arrayIndex: Int = 1
      features(0) = if (fields(26).toInt >= 20)  1 else 0  // if the label is having more birds, we consider the species to be present
      fields.foreach(col => {
        if (columnsSet.contains(index)  && index != 26) {
          if(col.trim.equals("?") || col.trim.equals("X")){
            features(arrayIndex) = 0.0                  // if the feature is missing, normalize it to zero
          }
          else{
            features(arrayIndex) = col.toDouble         // set the feature array
          }
          arrayIndex += 1
        }
        index = index + 1
      })

      LabeledPoint(features(0), Vectors.dense(features.tail)) // prepare the label point with Agelaius_phoeniceus
    }else{null}}




  // A generalized function for validation and testing to predict the label for LabeledPoint record
  // given all the four models
  def predictLabel(point: LabeledPoint, decisionTreeModel: DecisionTreeModel, logisticRegressionModel: GradientBoostedTreesModel,
                   randomForestModel: RandomForestModel, gradientBoostModel: LogisticRegressionModel) : (Double, Double) = {

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
    conf.setAppName("Training and Validation")
    conf.setMaster("local")
    val sc : SparkContext = new SparkContext(conf)
    val inputRDD : RDD[String] = sc.textFile(args(0))


    // Keep track of the columns that needs to be considered
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

    // Add to a hashset for easy look up across the nodes in the cluster.
    // This acts as a distributed file for the mapper function to prune the columns during the
    // preprocessing stage.
    columns.foreach(value => columnsSet.add(value))

    // RDD of the new parsed labeled points
    val parsedData : RDD[LabeledPoint] =  inputRDD.map(line => parseTrainingData(line, columnsSet)).filter(x=> x!=null).persist()

    // Split the RDD into Training data and Validation data
    val splits : Array[RDD[LabeledPoint]] = parsedData.randomSplit(Array(0.7, 0.3))
    val (trainingData : RDD[LabeledPoint], testData : RDD[LabeledPoint]) = (splits(0), splits(1))


    // Add categorical fields to be consumed by the MLLib classifiers
    // Our categorical fields are
    // Months 1-12
    // Days 1-31
    // BCR 1-38
    // OMERNIK_L3_ECOREGION 1-121
    var categoricalFeaturesInfo : Map[Int, Int] = Map[Int, Int]()
    categoricalFeaturesInfo += (2 -> 31)
    categoricalFeaturesInfo += (3 -> 367)
    categoricalFeaturesInfo += (9 -> 38)
    categoricalFeaturesInfo += (10 -> 121)


    // Prepare random sample for all the 4 models - Decisiontree, Randomforest, graidientboost, logisticregression
    val randomTrainingData : RDD[(Int, LabeledPoint)] = trainingData.map(line => (scala.util.Random.nextInt(4), line))
    val decisionTreeModel : DecisionTreeModel= DecisionTree.trainClassifier(randomTrainingData.filter(x => x._1!=0 || x._1 == null).map(x => x._2), 2, categoricalFeaturesInfo, "gini", 9, 4000)
    val randomForestModel : RandomForestModel = RandomForest.trainClassifier(randomTrainingData.filter(x => x._1!=1 || x._1 == null).map(x => x._2), Strategy.defaultStrategy("Classification"), 10, "auto", 4000)
    var boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(5)
    boostingStrategy.treeStrategy.setNumClasses(2)
    boostingStrategy.treeStrategy.setMaxDepth(9)
    val gradientBoostModel : GradientBoostedTreesModel = GradientBoostedTrees.train(randomTrainingData.filter(x => x._1!=2 || x._1 == null).map(x => x._2), boostingStrategy)
    val logisticRegressionModel : LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(randomTrainingData.filter(x => x._1!=3 || x._1 == null).map(x => x._2))



    // This is the validation step to calculate the accuracy of the validation data
    val EnsembleValidationRDD : RDD[(Double, Double)]= testData.map(point => predictLabel(point, decisionTreeModel,  gradientBoostModel , randomForestModel,logisticRegressionModel))
    val finalAccuracy = EnsembleValidationRDD.filter(r => r._1 == r._2).count.toDouble / testData.count()

    println("Accuracy = " + finalAccuracy)

    decisionTreeModel.save(sc, args(1) + "/decisionTreeModel")
    randomForestModel.save(sc, args(1) + "/randomForestModel")
    gradientBoostModel.save(sc, args(1) + "/gradientBoostModel")
    logisticRegressionModel.save(sc, args(1) + "/logisticRegressionModel")

    sc.stop()
  }


}