package in.ac.iitkgp.atdc;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

public class DistributedNaive implements Serializable{

	public JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> multiply(
			JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> A, JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> B,
			int rowA, int colA, int colB) {

		JavaPairRDD<Tuple2<Integer, Integer>, String> pairRDDA = A.flatMapToPair(
				new PairFlatMapFunction<Tuple2<Tuple2<Integer, Integer>, Double>, Tuple2<Integer, Integer>, String>() {

					@Override
					public Iterator<Tuple2<Tuple2<Integer, Integer>, String>> call(
							Tuple2<Tuple2<Integer, Integer>, Double> tuple) throws Exception {
						List<Tuple2<Tuple2<Integer, Integer>, String>> list = new ArrayList<Tuple2<Tuple2<Integer, Integer>, String>>();
						int i = tuple._1._1;
						int j = tuple._1._2;
						double value = tuple._2;
						for (int k = 0; k < colB; k++) {
							Tuple2<Integer, Integer> key = new Tuple2<Integer, Integer>(i, k);
							String val = "A," + j + "," + Double.toString(value);
							list.add(new Tuple2(key, val));
						}
						return list.iterator();
					}

				});

		JavaPairRDD<Tuple2<Integer, Integer>, String> pairRDDB = B.flatMapToPair(
				new PairFlatMapFunction<Tuple2<Tuple2<Integer, Integer>, Double>, Tuple2<Integer, Integer>, String>() {

					@Override
					public Iterator<Tuple2<Tuple2<Integer, Integer>, String>> call(
							Tuple2<Tuple2<Integer, Integer>, Double> tuple) throws Exception {
						List<Tuple2<Tuple2<Integer, Integer>, String>> list = new ArrayList<Tuple2<Tuple2<Integer, Integer>, String>>();
						int j = tuple._1._1;
						int k = tuple._1._2;
						double value = tuple._2;
						for (int i = 0; k < rowA; i++) {
							Tuple2<Integer, Integer> key = new Tuple2<Integer, Integer>(i, k);
							String val = "B," + j + "," + Double.toString(value);
							list.add(new Tuple2(key, val));
						}
						return list.iterator();
					}

				});

		JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> result = pairRDDA.union(pairRDDB).groupByKey().map(
				new Function<Tuple2<Tuple2<Integer, Integer>, Iterable<String>>, Tuple2<Tuple2<Integer, Integer>, Double>>() {

					@Override
					public Tuple2<Tuple2<Integer, Integer>, Double> call(
							Tuple2<Tuple2<Integer, Integer>, Iterable<String>> tuple2) throws Exception {
						HashMap<Integer, Double> hashA = new HashMap<Integer, Double>();
						HashMap<Integer, Double> hashB = new HashMap<Integer, Double>();
						int row = tuple2._1._1;
						int col = tuple2._1._2;
						Iterator<String> it = tuple2._2.iterator();

						while (it.hasNext()) {
							String str = (String) it.next();
							String[] arr = str.split(",");
							String matName = arr[0];
							int index = Integer.parseInt(arr[1]);
							double value = Double.parseDouble(arr[2]);

							if (matName.equals("A")) {
								hashA.put(index, value);
							} else {
								hashB.put(index, value);
							}
						}

						double sum = 0;

						for (Integer i : hashA.keySet()) {
							double valA = hashA.get(i);
							double valB = hashB.get(i);
							double product = valA * valB;
							sum = sum + product;
						}
						return new Tuple2(new Tuple2(row, col), sum);
					}
				});

		return result;

	}

	public void print(JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> rdd) {
		List<Tuple2<Tuple2<Integer, Integer>, Double>> list = new ArrayList<Tuple2<Tuple2<Integer, Integer>, Double>>();
		list = rdd.collect();
		for (int i = 0; i < list.size(); i++) {
			System.out.println(list.get(i)._1._1 + ":" + list.get(i)._1._2 + "=" + list.get(i)._2);
		}
	}

	public static void main(String args[]) {

		DistributedNaive distributedNaive = new DistributedNaive();
		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);

		String fileNameA = args[0];
		String fileNameB = args[1];
		int rowA = 4;
		int colA = 4;
		int colB = 4;

		String pathA = "E:\\" + fileNameA + ".csv";
		String pathB = "E:\\" + fileNameB + ".csv";

		JavaRDD<String> linesA = sc.textFile(pathA);
		JavaRDD<String> linesB = sc.textFile(pathB);

		JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> A = linesA
				.map(new Function<String, Tuple2<Tuple2<Integer, Integer>, Double>>() {

					@Override
					public Tuple2<Tuple2<Integer, Integer>, Double> call(String str) throws Exception {
						String[] array = str.split(",");
						int row = Integer.parseInt(array[0]);
						int col = Integer.parseInt(array[1]);
						double value = Double.parseDouble(array[2]);
						return new Tuple2(new Tuple2(row, col), value);
					}
				});

		JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> B = linesB
				.map(new Function<String, Tuple2<Tuple2<Integer, Integer>, Double>>() {

					@Override
					public Tuple2<Tuple2<Integer, Integer>, Double> call(String str) throws Exception {
						String[] array = str.split(",");
						int row = Integer.parseInt(array[0]);
						int col = Integer.parseInt(array[1]);
						double value = Double.parseDouble(array[2]);
						return new Tuple2(new Tuple2(row, col), value);
					}
				});

		distributedNaive.print(distributedNaive.multiply(A, B, rowA, colA, colB));
	}

}