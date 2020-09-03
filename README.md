# myspark

download spark .tar.gz
put all jars in Eclipse project and build path

download winutils.exe (hadoop-common-2.2.0-bin-master) and put in the  spark --> bin/
C:\Users\Utente\Documents\ALE\spark-3.0.0-preview2-bin-hadoop2.7\bin
https://github.com/steveloughran/winutils/tree/master/hadoop-2.7.1/bin

in the .java:
1) USE LOCAL
	SparkConf conf = new SparkConf().setAppName("PCA Example").setMaster("local[*]");

2) set dir
	System.setProperty("hadoop.home.dir", "C:\\Users\\Utente\\Documents\\ALE\\spark-3.0.0-preview2-bin-hadoop2.7");
