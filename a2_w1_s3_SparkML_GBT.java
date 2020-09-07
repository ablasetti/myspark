// A. BLASETTI 9/7/2020

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



// Try to javalize https://github.com/IBM/skillsnetwork/blob/master/coursera_ml/a2_w1_s3_SparkML_GBT.ipynb

public class a2_w1_s3_SparkML_GBT {


	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Users\\ablasetti\\spark-3.0.0-bin-hadoop2.7\\spark-3.0.0-bin-hadoop2.7");

		try {
			SparkSession spark = SparkSession
					.builder()
					.appName("JAVA_GBT")
					.master("local[*]")
					.getOrCreate();

			System.out.println("myspark");
			

//			spark.sparkContext.setLogLevel("ERROR");
//			spark.log().info("");

			Dataset<Row> data = spark
					.read()
					.format("parquet")
					.load("hmp.parquet"); //x,y,z accelometer data classified vs ACTIONS DONE by people...

			data.show(10);
			
			data.createTempView("data"); // due volte data???
			
			// Use just 2 classes
			Dataset<Row> df_two_class = spark.sql("select * from data where class in ('Use_telephone','Standup_chair')");
			df_two_class.show(5);
			
			
//			+---+---+---+--------------------+-------------+
//			|  x|  y|  z|              source|        class|
//			+---+---+---+--------------------+-------------+
//			| 30| 40| 51|Accelerometer-201...|Standup_chair|
//			| 30| 41| 51|Accelerometer-201...|Standup_chair|
//			| 31| 41| 51|Accelerometer-201...|Standup_chair|
//			| 29| 42| 51|Accelerometer-201...|Standup_chair|
//			| 30| 43| 52|Accelerometer-201...|Standup_chair|
//			+---+---+---+--------------------+-------------+
			
			
			Dataset<Row>[] splits = df_two_class.randomSplit(new double[]{0.8, 0.2});
			Dataset<Row> df_train  = splits[0];
			Dataset<Row> df_test  = splits[1];
			
			// From label to index/numbers...
			StringIndexer  indexer  = new StringIndexer()
					.setInputCol("class")
					.setOutputCol("label");
			

		    VectorAssembler vectorAssembler = new VectorAssembler()  // sparl vuole tutte le feature in un unico vettore...
		    	      .setInputCols(new String[]{"x", "y", "z"})
		    	      .setOutputCol("features");
		    
		    
		    MinMaxScaler normalizer  = new MinMaxScaler()
		    	      .setInputCol("features")
		    	      .setOutputCol("features_norm");

		    
		    
		    // GBT model definition...
		    GBTClassifier gbt = new GBTClassifier()
		      .setLabelCol("label")
		      .setFeaturesCol("features_norm")
		      .setMaxIter(10);
		    
		    
		    // set the pipeline...
			Pipeline pipeline = new Pipeline()
					.setStages(new PipelineStage[]{indexer,vectorAssembler,normalizer,gbt});
			
			
			PipelineModel model = pipeline.fit(df_train);
			Dataset<Row> prediction = model.transform(df_train);
			
			
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
					.setLabelCol("label")
					.setPredictionCol("prediction")
					.setMetricName("accuracy");
			double accuracy = evaluator.evaluate(prediction);
			System.out.println("Accuracy = "  + accuracy);
//			System.out.println("Test Error = " + (1.0 - accuracy));
			
			
			System.out.println("Run model on test data...");
			Dataset<Row> test_prediction = model.transform(df_test);
			double test_accuracy = evaluator.evaluate(test_prediction);
			System.out.println("TEST Accuracy = "  + test_accuracy);
			
			
		} catch (AnalysisException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}
}