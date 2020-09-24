// A. BLASETTI 9/7/2020


import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



// Try to javalize https://github.com/IBM/skillsnetwork/blob/master/coursera_ml/a2_w1_s3_SparkML_GBT.ipynb

public class a2_w1_s3_SparkML_NN {


	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Users\\ablasetti\\spark-3.0.0-bin-hadoop2.7\\spark-3.0.0-bin-hadoop2.7");

		try {
			
			System.out.println("Starting spark...");
			SparkSession spark = SparkSession
					.builder()
					.appName("JAVA_NN")
					.master("local[*]")
					.getOrCreate();

			// read parquet file x,y,z accellerometer data classified vs ACTIONS DONE by people...
			Dataset<Row> data = spark
					.read()
					.format("parquet")
					.load("hmp.parquet"); 

			data.show(10);

			// to use spark sql quaries...
			data.createTempView("data"); // due volte data???
			
			
			System.out.println("Number of classes...");
			spark.sql("SELECT count(class), class  from data  group by class").show();
	
			
			// Use just 2 classes
			System.out.println("Reduce to 2 classes...");
			Dataset<Row> df_two_class = spark.sql("select * from data where class in ('Use_telephone','Standup_chair')");
			df_two_class.show(5);
			
			Dataset<Row>[] splits = df_two_class.randomSplit(new double[]{0.8, 0.2});
			Dataset<Row> df_train  = splits[0];
			Dataset<Row> df_test  = splits[1];
			
			// From label to index/numbers...
			StringIndexer  indexer  = new StringIndexer()
					.setInputCol("class")
					.setOutputCol("label");
			

		    VectorAssembler vectorAssembler = new VectorAssembler()  // spark vuole tutte le features in un unico vettore...
		    	      .setInputCols(new String[]{"x", "y", "z"})
		    	      .setOutputCol("features");
		    
		    
		    MinMaxScaler normalizer  = new MinMaxScaler()
		    	      .setInputCol("features")
		    	      .setOutputCol("features_norm");
		    
		    
		    int[] layers = new int[] {3, 5, 4, 2}; //   x,y,z --> 'Use_telephone','Standup_chair' [in=3; out = 2]
			
		    
		    // create the trainer and set its parameters
		    MultilayerPerceptronClassifier nnet = new MultilayerPerceptronClassifier()
		      .setLayers(layers)
		      .setBlockSize(128)
		      .setSeed(1234L)
		      .setMaxIter(100);
		    
		    
		    // set the pipeline...
			Pipeline pipeline = new Pipeline()
					.setStages(new PipelineStage[]{indexer,vectorAssembler,normalizer,nnet});
			
			
			PipelineModel model = pipeline.fit(df_train);
			Dataset<Row> prediction = model.transform(df_train);
			
			
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
					.setLabelCol("label")
					.setPredictionCol("prediction")
					.setMetricName("accuracy");
			double accuracy = evaluator.evaluate(prediction);
			System.out.println("Accuracy = "  + accuracy);
		    
			
			spark.stop();
			
		} catch (AnalysisException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}
}