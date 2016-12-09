import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.tree.model.{RandomForestModel, GradientBoostedTreesModel, DecisionTreeModel}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * Created by Abhi on 12/9/16.
  */
object eBirdPrediction {

  // function to parse the test data record to convert into labeled point
  def parseTestingData(line: String, columnsSet: mutable.HashSet[Int]) = {
    val fields: Array[String] = line.split(",")   // Split on ","

    // Ignore if its first row in the file or if the label has invalid characters
    if(!fields(0).equals("SAMPLING_EVENT_ID")) {
      var index: Int = 0
      val features: Array[Double] = Array.ofDim[Double](columnsSet.size)  // initialize a features array
      var arrayIndex: Int = 1
      features(0) = fields(0).substring(1,fields(0).length).toDouble      // consider the zeroth column for output
      fields.foreach(col => {
        if (columnsSet.contains(index) && index != 26) {
          if(col.trim.equals("?") || col.trim.equals("X")){
            features(arrayIndex) = 0.0              // if the feature is missing, normalize it to zero
          }
          else{
            features(arrayIndex) = col.toDouble       // set the feature array
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

  def main(args: Array[String]): Unit = {

    // Spark configuration
    val conf : SparkConf= new SparkConf()
    conf.setAppName("Training and Validation")
    conf.setMaster("local")
    val sc : SparkContext = new SparkContext(conf)

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


    val decisionTreeModel = DecisionTreeModel.load(sc, args(1) + "/decisionTreeModel")
    val randomForestModel = RandomForestModel.load(sc, args(1) + "/randomForestModel")
    val gradientBoostModel = GradientBoostedTreesModel.load(sc, args(1) + "/gradientBoostModel")
    val logisticRegressionModel = LogisticRegressionModel.load(sc, args(1) + "/logisticRegressionModel")


    // read the test file for predictions
    val testInputRdd : RDD[String] = sc.textFile(args(0))
    // pre-process the test file exactly the way training data was preprocessed
    val predictionData : RDD[LabeledPoint]= testInputRdd.map(line => parseTestingData(line, columnsSet)).filter(x=> x!=null).persist()

    // Predict the labels for the test records
    val EnsemblePredictionRDD : RDD[(Double, Double)]= predictionData.map { point => predictLabel(point, decisionTreeModel, gradientBoostModel, randomForestModel, logisticRegressionModel)}

    // Save the predicted values to a text file in the given format
    EnsemblePredictionRDD.map(line=> "S"+line._1.toInt + "," + line._2).saveAsTextFile(args(1)+"/output")

  }

}
