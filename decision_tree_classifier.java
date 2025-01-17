
// $example on$
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class decision_tree_classifier {


	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Users\\ablasetti\\spark-3.0.0-bin-hadoop2.7\\spark-3.0.0-bin-hadoop2.7");


		SparkSession spark = SparkSession
				.builder()
				.appName("JavaDecisionTreeClassificationExample")
				.master("local[*]")
				.getOrCreate();

		System.out.println("myspark");

		//			SparkConf conf = new SparkConf().setAppName("PCA Example").setMaster("local[*]");
		//		    SparkContext sc = new SparkContext(conf);
		//		    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);


		Dataset<Row> data = spark
				.read()
				.format("libsvm")
				.load("C:\\Users\\ablasetti\\spark-3.0.0-bin-hadoop2.7\\spark-3.0.0-bin-hadoop2.7\\data\\mllib/sample_libsvm_data.txt");



		System.out.println(data.head());

		String cols[] = data.columns();
		for(String col:cols)
			System.out.println(col);


		// Index labels, adding metadata to the label column.
		// Fit on whole dataset to include all labels in index.
		StringIndexerModel labelIndexer = new StringIndexer()
				.setInputCol("label")
				.setOutputCol("indexedLabel")
				.fit(data);


		// Automatically identify categorical features, and index them.
		VectorIndexerModel featureIndexer = new VectorIndexer()
				.setInputCol("features")
				.setOutputCol("indexedFeatures")
				.setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
				.fit(data);

		System.out.println("Out columns..."+ featureIndexer.getOutputCol());

		// Split the data into training and test sets (30% held out for testing).
		Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];

		// Train a DecisionTree model.
		DecisionTreeClassifier dt = new DecisionTreeClassifier()
				.setLabelCol("indexedLabel")
				.setFeaturesCol("indexedFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
				.setInputCol("prediction")
				.setOutputCol("predictedLabel")
				.setLabels(labelIndexer.labelsArray()[0]);


		// Chain indexers and tree in a Pipeline.
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});


		// Train model. This also runs the indexers.
		PipelineModel model = pipeline.fit(trainingData);

		// Make predictions.
		Dataset<Row> predictions = model.transform(testData);

		// Select example rows to display.
		predictions.select("predictedLabel", "label", "features").show(5);

		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("indexedLabel")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("Test Error = " + (1.0 - accuracy));

		DecisionTreeClassificationModel treeModel =
				(DecisionTreeClassificationModel) (model.stages()[2]);
		System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
		// $example off$

		spark.stop();
	}




}