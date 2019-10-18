/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package in.ac.iitkgp.atdc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.jblas.DoubleMatrix;

import scala.Tuple2;

/**
 *
 * @author chandan
 */
public class DS implements Serializable {

	public int count;
	// private static final Logger logger =
	// Logger.getLogger(DistributedStrassenNew.class);

	public DS() {
		count = 0;
	}

	public static void main(String args[]) {
		// logger.setLevel(Level.WARN);
		int blockSize = 0;
		int partitions = 0;
		String file = "";
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			System.out.println("Enter the block size");
			blockSize = Integer.parseInt(br.readLine());
			System.out.println("Enter the file name");
			file = br.readLine();
			System.out.println("Enter the number of partitions");
			partitions = Integer.parseInt(br.readLine());
		} catch (IOException ex) {
			// Logger.getLogger(DistributedNaive.class.getName()).log(Level.SEVERE,
			// null, ex);
		}

		// getting the context and spark configuration
		SparkConf sparkConf = new SparkConf().setAppName("Strassen_" + file + "_" + blockSize)
				.set("spark.driver.maxResultSize", "3g");
		;
		JavaSparkContext ctx = new JavaSparkContext(sparkConf);

		// getting the input file into JavaRDD of strings
		JavaRDD<String> lines = ctx.textFile("hdfs:/root/utkarsh/stark/" + file, partitions);
		// lines.saveAsTextFile("hdfs://192.168.0.109:8020/user/research/chandanm/inputMatrix");

		int size = (int) Math.sqrt(lines.count() / 2) / blockSize;
		final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);
		// final Broadcast<Integer> bSize = ctx.broadcast(size);

		JavaRDD<String> IndexedLines = lines.map(new Function<String, String>() {

			@Override
			public String call(String line) throws Exception {
				return line + "," + "M,0";
			}
		});

		/* Prepare Matrix A as JavaRDD<Block> */
		JavaRDD<Block> A = IndexedLines.filter(new Function<String, Boolean>() {

			@Override
			public Boolean call(String t1) throws Exception {
				return t1.split(",")[0].equals("A");
			}
		}).mapToPair(new PairFunction<String, String, String>() {

			@Override
			public Tuple2<String, String> call(String t) throws Exception {
				String key = Integer.toString(Integer.parseInt(t.split(",")[1]) / bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) / bblockSize.value()) + ","
						+ t.split(",")[4] + "," + t.split(",")[5];
				String value = Integer.toString(Integer.parseInt(t.split(",")[1]) % bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) % bblockSize.value()) + ","
						+ t.split(",")[3];

				return new Tuple2(key, value);
			}
		}).groupByKey().map(new Function<Tuple2<String, Iterable<String>>, Block>() {

			@Override
			public Block call(Tuple2<String, Iterable<String>> t1) throws Exception {
				int rowIndex = Integer.parseInt(t1._1.split(",")[0]);
				int colIndex = Integer.parseInt(t1._1.split(",")[1]);
				String level = t1._1.split(",")[2] + "," + t1._1.split(",")[3];
				double[][] mat = new double[bblockSize.value()][bblockSize.value()];
				Iterator it = t1._2.iterator();
				while (it.hasNext()) {
					String str = it.next().toString();
					int row = Integer.parseInt(str.split(",")[0]);
					int column = Integer.parseInt(str.split(",")[1]);
					double value = Double.parseDouble(str.split(",")[2]);
					mat[row][column] = value;
				}
				Block block = new Block();
				block.rowIndex = rowIndex;
				block.columnIndex = colIndex;
				block.matrix = mat;
				block.matname = "A," + level;

				return block;

			}
		});

		/* Prepare Matrix B as JavaRDD<Block> */
		JavaRDD<Block> B = IndexedLines.filter(new Function<String, Boolean>() {

			@Override
			public Boolean call(String t1) throws Exception {
				return t1.split(",")[0].equals("B");
			}
		}).mapToPair(new PairFunction<String, String, String>() {

			@Override
			public Tuple2<String, String> call(String t) throws Exception {
				String key = Integer.toString(Integer.parseInt(t.split(",")[1]) / bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) / bblockSize.value()) + ","
						+ t.split(",")[4] + "," + t.split(",")[5];
				String value = Integer.toString(Integer.parseInt(t.split(",")[1]) % bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) % bblockSize.value()) + ","
						+ t.split(",")[3];

				return new Tuple2(key, value);
			}
		}).groupByKey().map(new Function<Tuple2<String, Iterable<String>>, Block>() {

			@Override
			public Block call(Tuple2<String, Iterable<String>> t1) throws Exception {
				int rowIndex = Integer.parseInt(t1._1.split(",")[0]);
				int colIndex = Integer.parseInt(t1._1.split(",")[1]);
				String level = t1._1.split(",")[2] + "," + t1._1.split(",")[3];

				double[][] mat = new double[bblockSize.value()][bblockSize.value()];
				Iterator it = t1._2.iterator();
				while (it.hasNext()) {
					String str = it.next().toString();
					int row = Integer.parseInt(str.split(",")[0]);
					int column = Integer.parseInt(str.split(",")[1]);
					double value = Double.parseDouble(str.split(",")[2]);
					mat[row][column] = value;
				}
				Block block = new Block();
				block.rowIndex = rowIndex;
				block.columnIndex = colIndex;
				block.matrix = mat;
				block.matname = "B," + level;
				return block;

			}
		});

		// System.out.println(A.count());
		// System.out.println(B.count());

		DS strassen = new DS();
		long time1 = System.currentTimeMillis();
		// System.out.println("INITIAL SIZE:" + size);
		JavaRDD<Block> result = strassen.multiply(ctx, A, B, size, blockSize);
		// result.cache();
		// result.count();
		System.out.println(result.count());
		long time2 = System.currentTimeMillis();
		System.out.println((time2 - time1) / 1000);

	}

	public JavaRDD<String> toRDDString(JavaRDD<Block> blockRDD) {

		JavaRDD<String> RDDString = blockRDD.flatMap(new FlatMapFunction<Block, String>() {

			@Override
			public Iterator<String> call(Block block) throws Exception {
				ArrayList<String> list = new ArrayList<String>();
				double[][] mat = new double[512][512];
				mat = block.matrix;
				for (int i = 0; i < mat.length; i++) {
					for (int j = 0; j < mat.length; j++) {
						String str = Integer.toString((512 * block.rowIndex) + i) + ","
								+ Integer.toString((512 * block.columnIndex) + j) + "," + mat[i][j];
					}
				}
				return list.iterator();
			}

		});

		return RDDString;
	}

	public JavaRDD<String> add(JavaSparkContext ctx, JavaRDD<String> rdd, int blockRowIndex, int blockColIndex) {

		final Broadcast<Integer> bRow = ctx.broadcast(blockRowIndex);
		final Broadcast<Integer> bCol = ctx.broadcast(blockColIndex);

		JavaPairRDD<String, Double> pairRDD = rdd.mapToPair(new PairFunction<String, String, Double>() {

			@Override
			public Tuple2<String, Double> call(String line) throws Exception {

				return new Tuple2(line.split(",")[0] + "," + line.split(",")[1],
						Double.parseDouble(line.split(",")[4]));
			}
		});

		JavaPairRDD<String, Iterable<Double>> group = pairRDD.groupByKey();

		JavaRDD<String> sum = group.map(new Function<Tuple2<String, Iterable<Double>>, String>() {

			@Override
			public String call(Tuple2<String, Iterable<Double>> arg0) throws Exception {

				Iterator it = arg0._2.iterator();

				double sum = (double) it.next() + (double) it.next();
				return Integer.toString(bRow.value()) + "," + Integer.toString(bCol.value()) + "," + arg0._1 + ","
						+ Double.toString(sum);
			}

		});

		return sum;
	}

	public JavaRDD<Block> multiplyRDDString(JavaSparkContext ctx, JavaRDD<String> left, JavaRDD<String> right,
			int blockSize, int size) {

		final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);

		JavaRDD<String> leftIndexedLines = left.map(new Function<String, String>() {

			@Override
			public String call(String line) throws Exception {
				return line + "," + "M,0";
			}
		});
		JavaRDD<Block> A = leftIndexedLines.mapToPair(new PairFunction<String, String, String>() {

			@Override
			public Tuple2<String, String> call(String t) throws Exception {
				String key = Integer.toString(Integer.parseInt(t.split(",")[1]) / bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) / bblockSize.value()) + ","
						+ t.split(",")[4] + "," + t.split(",")[5];
				String value = Integer.toString(Integer.parseInt(t.split(",")[1]) % bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) % bblockSize.value()) + ","
						+ t.split(",")[3];

				return new Tuple2(key, value);
			}
		}).groupByKey().map(new Function<Tuple2<String, Iterable<String>>, Block>() {

			@Override
			public Block call(Tuple2<String, Iterable<String>> t1) throws Exception {
				int rowIndex = Integer.parseInt(t1._1.split(",")[0]);
				int colIndex = Integer.parseInt(t1._1.split(",")[1]);
				String level = t1._1.split(",")[2] + "," + t1._1.split(",")[3];
				double[][] mat = new double[bblockSize.value()][bblockSize.value()];
				Iterator it = t1._2.iterator();
				while (it.hasNext()) {
					String str = it.next().toString();
					int row = Integer.parseInt(str.split(",")[0]);
					int column = Integer.parseInt(str.split(",")[1]);
					double value = Double.parseDouble(str.split(",")[2]);
					mat[row][column] = value;
				}
				Block block = new Block();
				block.rowIndex = rowIndex;
				block.columnIndex = colIndex;
				block.matrix = mat;
				block.matname = "A," + level;

				return block;

			}
		});

		JavaRDD<String> rightIndexedLines = right.map(new Function<String, String>() {

			@Override
			public String call(String line) throws Exception {
				return line + "," + "M,0";
			}
		});

		JavaRDD<Block> B = rightIndexedLines.mapToPair(new PairFunction<String, String, String>() {

			@Override
			public Tuple2<String, String> call(String t) throws Exception {
				String key = Integer.toString(Integer.parseInt(t.split(",")[1]) / bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) / bblockSize.value()) + ","
						+ t.split(",")[4] + "," + t.split(",")[5];
				String value = Integer.toString(Integer.parseInt(t.split(",")[1]) % bblockSize.value()) + ","
						+ Integer.toString(Integer.parseInt(t.split(",")[2]) % bblockSize.value()) + ","
						+ t.split(",")[3];

				return new Tuple2(key, value);
			}
		}).groupByKey().map(new Function<Tuple2<String, Iterable<String>>, Block>() {

			@Override
			public Block call(Tuple2<String, Iterable<String>> t1) throws Exception {
				int rowIndex = Integer.parseInt(t1._1.split(",")[0]);
				int colIndex = Integer.parseInt(t1._1.split(",")[1]);
				String level = t1._1.split(",")[2] + "," + t1._1.split(",")[3];

				double[][] mat = new double[bblockSize.value()][bblockSize.value()];
				Iterator it = t1._2.iterator();
				while (it.hasNext()) {
					String str = it.next().toString();
					int row = Integer.parseInt(str.split(",")[0]);
					int column = Integer.parseInt(str.split(",")[1]);
					double value = Double.parseDouble(str.split(",")[2]);
					mat[row][column] = value;
				}
				Block block = new Block();
				block.rowIndex = rowIndex;
				block.columnIndex = colIndex;
				block.matrix = mat;
				block.matname = "B," + level;
				return block;

			}
		});

		JavaRDD<Block> result = multiply(ctx, A, B, size, blockSize);
		result.cache();
		result.count();

		return result;
	}

	public JavaRDD<Block> multiply(JavaSparkContext ctx, JavaRDD<Block> A, JavaRDD<Block> B, int size, int blockSize) {
		List<Block> c = new ArrayList<Block>();
		JavaRDD<Block> result1 = null;
		count++;

		if (size == 1) {
			System.out.println("Naive Method");

			JavaPairRDD<String, Block> indexedMatrices = A.union(B).mapToPair(new PairFunction<Block, String, Block>() {

				@Override
				public Tuple2<String, Block> call(Block t) throws Exception {
					String key = t.matname.split(",")[1] + "," + t.matname.split(",")[2];
					Block block = new Block();
					block.matname = t.matname.split(",")[0];
					block.rowIndex = t.rowIndex;
					block.columnIndex = t.columnIndex;
					if (t.allZero == true) {
						block.matrix = null;
						block.allZero = true;
					} else {
						block.matrix = t.matrix;
						block.allZero = false;
					}

					return new Tuple2(key, block);
				}
			});

			JavaPairRDD<String, Iterable<Block>> map4 = indexedMatrices.groupByKey();

			result1 = map4.map(new Function<Tuple2<String, Iterable<Block>>, Block>() {

				@Override
				public Block call(Tuple2<String, Iterable<Block>> t1) throws Exception {

					DS strassen = new DS();
					Block blockA = new Block();
					Block blockB = new Block();
					Block blockC = new Block();
					Iterator it = t1._2.iterator();
					while (it.hasNext()) {
						Block block = new Block();
						block = (Block) it.next();
						if (block.matname.equals("A")) {
							blockA = block;
						} else {
							blockB = block;
						}
					}

					blockC = strassen.multiply(blockA, blockB, t1._1);

					return blockC;
				}
			});

		} else {
			size = size / 2;
			final Broadcast<Integer> bSize = ctx.broadcast(size);
			final Broadcast<Integer> bBlockSize = ctx.broadcast(blockSize);

			JavaRDD<Block> union = A.union(B);
			System.out.println("After Union");

			JavaPairRDD<String, Block> map1 = union.flatMapToPair(new PairFlatMapFunction<Block, String, Block>() {

				@Override
				public Iterator<Tuple2<String, Block>> call(Block t) throws Exception {

					ArrayList<Tuple2<String, Block>> list = new ArrayList<Tuple2<String, Block>>();
					String key1;
					String key2;
					String key3;
					String key4;
					if (t.matname.split(",")[0].equals("A")) {

						if (t.rowIndex / bSize.value() == 0 && t.columnIndex / bSize.value() == 0) {
							Block block1 = new Block();
							Block block2 = new Block();
							Block block3 = new Block();
							Block block4 = new Block();

							key1 = "M1," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 0));
							key2 = "M3," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 2));
							key3 = "M5," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 4));
							key4 = "M6," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 5));

							block1.matname = "A11";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "A11";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();
							block3.matname = "A11";
							block3.rowIndex = t.rowIndex % bSize.value();
							block3.columnIndex = t.columnIndex % bSize.value();
							block4.matname = "A11";
							block4.rowIndex = t.rowIndex % bSize.value();
							block4.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;
							block3.matrix = t.matrix;
							block4.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));
							list.add(new Tuple2(key3, block3));
							list.add(new Tuple2(key4, block4));
						} else if (t.rowIndex / bSize.value() == 0 && t.columnIndex / bSize.value() == 1) {
							Block block1 = new Block();
							Block block2 = new Block();

							key1 = "M5," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 4));
							key2 = "M7," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 6));

							block1.matname = "A12";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "A12";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));

						} else if (t.rowIndex / bSize.value() == 1 && t.columnIndex / bSize.value() == 0) {
							Block block1 = new Block();
							Block block2 = new Block();

							key1 = "M2," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 1));
							key2 = "M6," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 5));

							block1.matname = "A21";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "A21";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));
						} else {
							Block block1 = new Block();
							Block block2 = new Block();
							Block block3 = new Block();
							Block block4 = new Block();

							key1 = "M1," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 0));
							key2 = "M2," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 1));
							key3 = "M4," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 3));
							key4 = "M7," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 6));

							block1.matname = "A22";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "A22";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();
							block3.matname = "A22";
							block3.rowIndex = t.rowIndex % bSize.value();
							block3.columnIndex = t.columnIndex % bSize.value();
							block4.matname = "A22";
							block4.rowIndex = t.rowIndex % bSize.value();
							block4.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;
							block3.matrix = t.matrix;
							block4.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));
							list.add(new Tuple2(key3, block3));
							list.add(new Tuple2(key4, block4));
						}

					} else {
						if (t.rowIndex / bSize.value() == 0 && t.columnIndex / bSize.value() == 0) {
							Block block1 = new Block();
							Block block2 = new Block();
							Block block3 = new Block();
							Block block4 = new Block();

							key1 = "M1," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 0));
							key2 = "M2," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 1));
							key3 = "M4," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 3));
							key4 = "M6," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 5));

							block1.matname = "B11";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "B11";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();
							block3.matname = "B11";
							block3.rowIndex = t.rowIndex % bSize.value();
							block3.columnIndex = t.columnIndex % bSize.value();
							block4.matname = "B11";
							block4.rowIndex = t.rowIndex % bSize.value();
							block4.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;
							block3.matrix = t.matrix;
							block4.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));
							list.add(new Tuple2(key3, block3));
							list.add(new Tuple2(key4, block4));
						} else if (t.rowIndex / bSize.value() == 0 && t.columnIndex / bSize.value() == 1) {
							Block block1 = new Block();
							Block block2 = new Block();

							key1 = "M3," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 2));
							key2 = "M6," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 5));

							block1.matname = "B12";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "B12";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));

						} else if (t.rowIndex / bSize.value() == 1 && t.columnIndex / bSize.value() == 0) {
							Block block1 = new Block();
							Block block2 = new Block();

							key1 = "M4," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 3));
							key2 = "M7," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 6));

							block1.matname = "B21";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "B21";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block1.allZero = false;
							block2.matrix = t.matrix;
							block2.allZero = false;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));

						} else {
							Block block1 = new Block();
							Block block2 = new Block();
							Block block3 = new Block();
							Block block4 = new Block();

							key1 = "M1," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 0));
							key2 = "M3," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 2));
							key3 = "M5," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 4));
							key4 = "M7," + Integer.toString(((Integer.parseInt(t.matname.split(",")[2]) * 7) + 6));

							block1.matname = "B22";
							block1.rowIndex = t.rowIndex % bSize.value();
							block1.columnIndex = t.columnIndex % bSize.value();
							block2.matname = "B22";
							block2.rowIndex = t.rowIndex % bSize.value();
							block2.columnIndex = t.columnIndex % bSize.value();
							block3.matname = "B22";
							block3.rowIndex = t.rowIndex % bSize.value();
							block3.columnIndex = t.columnIndex % bSize.value();
							block4.matname = "B22";
							block4.rowIndex = t.rowIndex % bSize.value();
							block4.columnIndex = t.columnIndex % bSize.value();

							block1.matrix = t.matrix;
							block2.matrix = t.matrix;
							block3.matrix = t.matrix;
							block4.matrix = t.matrix;

							list.add(new Tuple2(key1, block1));
							list.add(new Tuple2(key2, block2));
							list.add(new Tuple2(key3, block3));
							list.add(new Tuple2(key4, block4));

						}
					}

					return list.iterator();
				}
			});

			JavaPairRDD<String, Iterable<Block>> map2 = map1.groupByKey();

			JavaRDD<Block> map3 = map2.flatMap(new FlatMapFunction<Tuple2<String, Iterable<Block>>, Block>() {

				@Override
				public Iterator<Block> call(Tuple2<String, Iterable<Block>> t) throws Exception {
					DS strassen = new DS();
					ArrayList<Block> list = new ArrayList<Block>();

					String key = t._1;

					if (key.split(",")[0].equals("M1")) {

						Block blockA11 = new Block();
						Block blockA22 = new Block();
						Block blockB11 = new Block();
						Block blockB22 = new Block();

						HashMap<String, Block> A11 = new HashMap<String, Block>();
						HashMap<String, Block> A22 = new HashMap<String, Block>();
						HashMap<String, Block> B11 = new HashMap<String, Block>();
						HashMap<String, Block> B22 = new HashMap<String, Block>();

						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {

							block = (Block) it.next();
							if (block.matname.equals("A11")) {
								blockA11 = block;
								A11.put(Integer.toString(blockA11.rowIndex) + ","
										+ Integer.toString(blockA11.columnIndex), blockA11);
							} else if (block.matname.equals("A22")) {
								blockA22 = block;
								A22.put(Integer.toString(blockA22.rowIndex) + ","
										+ Integer.toString(blockA22.columnIndex), blockA22);
							} else if (block.matname.equals("B11")) {
								blockB11 = block;
								B11.put(Integer.toString(blockB11.rowIndex) + ","
										+ Integer.toString(blockB11.columnIndex), blockB11);
							} else {
								blockB22 = block;
								B22.put(Integer.toString(blockB22.rowIndex) + ","
										+ Integer.toString(blockB22.columnIndex), blockB22);
							}
						}

						Set set = A11.entrySet();
						Iterator itA11 = set.iterator();
						while (itA11.hasNext()) {
							Map.Entry me = (Map.Entry) itA11.next();
							Block block1 = (Block) me.getValue();
							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = A22.get(keys);
							list.add(strassen.add(block1, block2, key, block1.matname.substring(0, 1)));

							Block block3 = new Block();
							block3 = B11.get(keys);
							Block block4 = new Block();
							block4 = B22.get(keys);
							list.add(strassen.add(block3, block4, key, block3.matname.substring(0, 1)));
						}

					} else if (key.split(",")[0].equals("M2")) {

						Block blockA21 = new Block();
						Block blockA22 = new Block();
						Block blockB11 = new Block();

						HashMap<String, Block> A21 = new HashMap<String, Block>();
						HashMap<String, Block> A22 = new HashMap<String, Block>();
						HashMap<String, Block> B11 = new HashMap<String, Block>();

						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A21")) {
								blockA21 = block;
								A21.put(Integer.toString(blockA21.rowIndex) + ","
										+ Integer.toString(blockA21.columnIndex), blockA21);
							} else if (block.matname.equals("A22")) {
								blockA22 = block;
								A22.put(Integer.toString(blockA22.rowIndex) + ","
										+ Integer.toString(blockA22.columnIndex), blockA22);
							} else if (block.matname.equals("B11")) {
								blockB11 = block;
								B11.put(Integer.toString(blockB11.rowIndex) + ","
										+ Integer.toString(blockB11.columnIndex), blockB11);
							}
						}

						Set set = A21.entrySet();
						Iterator itA21 = set.iterator();
						while (itA21.hasNext()) {
							Map.Entry me = (Map.Entry) itA21.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = A22.get(keys);
							list.add(strassen.add(block1, block2, key, block1.matname.substring(0, 1)));

							Block block3 = new Block();
							block3 = B11.get(keys);
							block3.matname = "B," + key;
							list.add(block3);

						}

					} else if (key.split(",")[0].equals("M3")) {

						Block blockA11 = new Block();
						Block blockB12 = new Block();
						Block blockB22 = new Block();

						HashMap<String, Block> A11 = new HashMap<String, Block>();
						HashMap<String, Block> B12 = new HashMap<String, Block>();
						HashMap<String, Block> B22 = new HashMap<String, Block>();
						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A11")) {
								blockA11 = block;
								A11.put(Integer.toString(blockA11.rowIndex) + ","
										+ Integer.toString(blockA11.columnIndex), blockA11);
							} else if (block.matname.equals("B12")) {
								blockB12 = block;
								B12.put(Integer.toString(blockB12.rowIndex) + ","
										+ Integer.toString(blockB12.columnIndex), blockB12);
							} else if (block.matname.equals("B22")) {
								blockB22 = block;
								B22.put(Integer.toString(blockB22.rowIndex) + ","
										+ Integer.toString(blockB22.columnIndex), blockB22);
							}
						}

						Set set = A11.entrySet();
						Iterator itA11 = set.iterator();
						while (itA11.hasNext()) {
							Map.Entry me = (Map.Entry) itA11.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();
							block1.matname = "A," + key;
							list.add(block1);

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = B12.get(keys);

							Block block3 = new Block();
							block3 = B22.get(keys);
							list.add(strassen.subtract(block2, block3, key, block2.matname.substring(0, 1)));
						}

					} else if (key.split(",")[0].equals("M4")) {

						Block blockA22 = new Block();
						Block blockB11 = new Block();
						Block blockB21 = new Block();

						HashMap<String, Block> A22 = new HashMap<String, Block>();
						HashMap<String, Block> B21 = new HashMap<String, Block>();
						HashMap<String, Block> B11 = new HashMap<String, Block>();
						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A22")) {
								blockA22 = block;
								A22.put(Integer.toString(blockA22.rowIndex) + ","
										+ Integer.toString(blockA22.columnIndex), blockA22);
							} else if (block.matname.equals("B21")) {
								blockB21 = block;
								B21.put(Integer.toString(blockB21.rowIndex) + ","
										+ Integer.toString(blockB21.columnIndex), blockB21);
							} else if (block.matname.equals("B11")) {
								blockB11 = block;
								B11.put(Integer.toString(blockB11.rowIndex) + ","
										+ Integer.toString(blockB11.columnIndex), blockB11);
							}
						}

						Set set = A22.entrySet();
						Iterator itA22 = set.iterator();
						while (itA22.hasNext()) {
							Map.Entry me = (Map.Entry) itA22.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();
							block1.matname = "A," + key;
							list.add(block1);

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = B21.get(keys);

							Block block3 = new Block();
							block3 = B11.get(keys);
							list.add(strassen.subtract(block2, block3, key, block2.matname.substring(0, 1)));
						}

					} else if (key.split(",")[0].equals("M5")) {

						Block blockA11 = new Block();
						Block blockA12 = new Block();
						Block blockB22 = new Block();

						HashMap<String, Block> A11 = new HashMap<String, Block>();
						HashMap<String, Block> A12 = new HashMap<String, Block>();
						HashMap<String, Block> B22 = new HashMap<String, Block>();
						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A11")) {
								blockA11 = block;
								A11.put(Integer.toString(blockA11.rowIndex) + ","
										+ Integer.toString(blockA11.columnIndex), blockA11);
							} else if (block.matname.equals("A12")) {
								blockA12 = block;
								A12.put(Integer.toString(blockA12.rowIndex) + ","
										+ Integer.toString(blockA12.columnIndex), blockA12);
							} else if (block.matname.equals("B22")) {
								blockB22 = block;
								B22.put(Integer.toString(blockB22.rowIndex) + ","
										+ Integer.toString(blockB22.columnIndex), blockB22);
							}
						}

						Set set = A11.entrySet();
						Iterator itA11 = set.iterator();
						while (itA11.hasNext()) {
							Map.Entry me = (Map.Entry) itA11.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = A12.get(keys);

							list.add(strassen.add(block1, block2, key, block1.matname.substring(0, 1)));
							Block block3 = new Block();
							block3 = B22.get(keys);
							block3.matname = "B," + key;
							list.add(block3);
						}

					} else if (key.split(",")[0].equals("M6")) {

						Block blockA11 = new Block();
						Block blockA21 = new Block();
						Block blockB11 = new Block();
						Block blockB12 = new Block();

						HashMap<String, Block> A21 = new HashMap<String, Block>();
						HashMap<String, Block> A11 = new HashMap<String, Block>();
						HashMap<String, Block> B11 = new HashMap<String, Block>();
						HashMap<String, Block> B12 = new HashMap<String, Block>();
						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A21")) {
								blockA21 = block;
								A21.put(Integer.toString(blockA21.rowIndex) + ","
										+ Integer.toString(blockA21.columnIndex), blockA21);
							} else if (block.matname.equals("A11")) {
								blockA11 = block;
								A11.put(Integer.toString(blockA11.rowIndex) + ","
										+ Integer.toString(blockA11.columnIndex), blockA11);
							} else if (block.matname.equals("B11")) {
								blockB11 = block;
								B11.put(Integer.toString(blockB11.rowIndex) + ","
										+ Integer.toString(blockB11.columnIndex), blockB11);
							} else {
								blockB12 = block;
								B12.put(Integer.toString(blockB12.rowIndex) + ","
										+ Integer.toString(blockB12.columnIndex), blockB12);
							}
						}

						Set set = A21.entrySet();
						Iterator itA21 = set.iterator();
						while (itA21.hasNext()) {
							Map.Entry me = (Map.Entry) itA21.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = A11.get(keys);

							list.add(strassen.subtract(block1, block2, key, block1.matname.substring(0, 1)));
							Block block3 = new Block();
							block3 = B11.get(keys);

							Block block4 = new Block();
							block4 = B12.get(keys);

							list.add(strassen.add(block3, block4, key, block3.matname.substring(0, 1)));
						}

					} else if (key.split(",")[0].equals("M7")) {
						Block blockA12 = new Block();
						Block blockA22 = new Block();
						Block blockB21 = new Block();
						Block blockB22 = new Block();

						HashMap<String, Block> A12 = new HashMap<String, Block>();
						HashMap<String, Block> A22 = new HashMap<String, Block>();
						HashMap<String, Block> B21 = new HashMap<String, Block>();
						HashMap<String, Block> B22 = new HashMap<String, Block>();
						Iterator it = t._2.iterator();
						Block block = new Block();
						while (it.hasNext()) {
							block = (Block) it.next();
							if (block.matname.equals("A12")) {
								blockA12 = block;
								A12.put(Integer.toString(blockA12.rowIndex) + ","
										+ Integer.toString(blockA12.columnIndex), blockA12);
							} else if (block.matname.equals("A22")) {
								blockA22 = block;
								A22.put(Integer.toString(blockA22.rowIndex) + ","
										+ Integer.toString(blockA22.columnIndex), blockA22);
							} else if (block.matname.equals("B21")) {
								blockB21 = block;
								B21.put(Integer.toString(blockB21.rowIndex) + ","
										+ Integer.toString(blockB21.columnIndex), blockB21);
							} else {
								blockB22 = block;
								B22.put(Integer.toString(blockB22.rowIndex) + ","
										+ Integer.toString(blockB22.columnIndex), blockB22);
							}
						}

						Set set = A12.entrySet();
						Iterator itA12 = set.iterator();
						while (itA12.hasNext()) {
							Map.Entry me = (Map.Entry) itA12.next();
							Block block1 = new Block();
							block1 = (Block) me.getValue();

							String keys = Integer.toString(block1.rowIndex) + ","
									+ Integer.toString(block1.columnIndex);
							Block block2 = new Block();
							block2 = A22.get(keys);

							list.add(strassen.subtract(block1, block2, key, block1.matname.substring(0, 1)));

							Block block3 = new Block();
							block3 = B21.get(keys);

							Block block4 = new Block();
							block4 = B22.get(keys);

							list.add(strassen.add(block3, block4, key, block3.matname.substring(0, 1)));
						}

					}

					return list.iterator();
				}

			});

			JavaRDD<Block> a = map3.filter(new Function<Block, Boolean>() {

				@Override
				public Boolean call(Block t1) throws Exception {
					return t1.matname.split(",")[0].equals("A");
				}
			});

			// a.setName("A RDD");
			// a.cache();
			// a.count();

			JavaRDD<Block> b = map3.filter(new Function<Block, Boolean>() {

				@Override
				public Boolean call(Block t1) throws Exception {
					return t1.matname.split(",")[0].equals("B");
				}
			});

			// b.setName("B RDD");
			// b.cache();
			// b.count();

			JavaRDD<Block> result = multiply(ctx, a, b, size, blockSize);

			count--;

			JavaPairRDD<String, Block> group = result.mapToPair(new PairFunction<Block, String, Block>() {

				@Override
				public Tuple2<String, Block> call(Block t) throws Exception {
					String key = Double.toString(Math.floor(Double.parseDouble(t.matname.split((","))[1]) / 7));
					// t.matname=t.matname.split(",")[0]+","+Double.toString(Double.parseDouble(t.matname.split((","))[1])
					// % 7);
					return new Tuple2(key, t);
				}
			});

			// group.setName("Group RDD after multiplication "+count);
			// group.cache();
			// group.count();

			JavaPairRDD<String, Iterable<Block>> group1 = group.groupByKey();
			result1 = group1.flatMap(new FlatMapFunction<Tuple2<String, Iterable<Block>>, Block>() {

				@Override
				public Iterator<Block> call(Tuple2<String, Iterable<Block>> t) throws Exception {
					String M = "";
					HashMap<String, Block> M1 = new HashMap<String, Block>();
					HashMap<String, Block> M2 = new HashMap<String, Block>();
					HashMap<String, Block> M3 = new HashMap<String, Block>();
					HashMap<String, Block> M4 = new HashMap<String, Block>();
					HashMap<String, Block> M5 = new HashMap<String, Block>();
					HashMap<String, Block> M6 = new HashMap<String, Block>();
					HashMap<String, Block> M7 = new HashMap<String, Block>();

					ArrayList<Block> blockList = new ArrayList<Block>();

					Iterator it = t._2.iterator();
					while (it.hasNext()) {
						Block block = new Block();
						block = (Block) it.next();
						if (block.matname.split(",")[0].equals("M1")) {
							M1.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else if (block.matname.split(",")[0].equals("M2")) {
							M2.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else if (block.matname.split(",")[0].equals("M3")) {
							M3.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else if (block.matname.split(",")[0].equals("M4")) {
							M4.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else if (block.matname.split(",")[0].equals("M5")) {
							M5.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else if (block.matname.split(",")[0].equals("M6")) {
							M6.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						} else {
							M7.put(Integer.toString(block.rowIndex) + Integer.toString(block.columnIndex), block);
						}
					}

					DS strassen = new DS();
					blockList = strassen.rearrange(M1, M2, M3, M4, M5, M6, M7);

					int mod = (int) (Double.parseDouble(t._1) % 7);
					if (mod == 0) {
						M = "M1";
					} else if (mod == 1) {
						M = "M2";
					} else if (mod == 2) {
						M = "M3";
					} else if (mod == 3) {
						M = "M4";
					} else if (mod == 4) {
						M = "M5";
					} else if (mod == 5) {
						M = "M6";
					} else {
						M = "M7";
					}

					for (int i = 0; i < blockList.size(); i++) {

						blockList.get(i).matname = M + "," + t._1;
					}

					return blockList.iterator();
				}
			});

			// result1.setName("result RDD at step "+count);
			// result1.cache();
			// result1.count();

		}

		return result1;
	}

	public Block multiply(Block block1, Block block2, String name) {
		Block block = new Block();
		block.rowIndex = block1.rowIndex;
		block.columnIndex = block1.columnIndex;
		int size = block1.matrix.length;
		double[][] mat3 = new double[block1.matrix.length][block1.matrix.length];
		DoubleMatrix mat1 = new DoubleMatrix(block1.matrix);
		DoubleMatrix mat2 = new DoubleMatrix(block2.matrix);
		mat3 = mat1.mmul(mat2).toArray2();

		block.matrix = mat3;
		block.matname = name;
		block.allZero = false;

		return block;
	}

	public Block add(Block block1, Block block2) {
		Block block = new Block();
		block.rowIndex = block1.rowIndex;
		block.columnIndex = block1.columnIndex;

		double[][] mat = new double[block1.matrix.length][block1.matrix.length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat.length; j++) {
				mat[i][j] = block1.matrix[i][j] + block2.matrix[i][j];
			}
		}
		block.matrix = mat;
		return block;
	}

	public Block subtract(Block block1, Block block2) {
		Block block = new Block();
		block.rowIndex = block1.rowIndex;
		block.columnIndex = block1.columnIndex;

		double[][] mat = new double[block1.matrix.length][block1.matrix.length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat.length; j++) {
				mat[i][j] = block1.matrix[i][j] - block2.matrix[i][j];
			}
		}
		block.matrix = mat;

		return block;
	}

	public Block add(Block block1, Block block2, String key, String matName) {
		Block block = new Block();
		block.rowIndex = block1.rowIndex;
		block.columnIndex = block1.columnIndex;

		double[][] mat3 = new double[block1.getMatrix().length][block1.getMatrix().length];
		for (int i = 0; i < block1.getMatrix().length; i++) {
			for (int j = 0; j < block1.getMatrix().length; j++) {
				mat3[i][j] = block1.getMatrix()[i][j] + block2.getMatrix()[i][j];
			}
		}
		block.matrix = mat3;
		block.matname = matName + "," + key;

		return block;
	}

	public Block subtract(Block block1, Block block2, String key, String matName) {
		Block block = new Block();
		block.rowIndex = block1.rowIndex;
		block.columnIndex = block1.columnIndex;

		double[][] mat3 = new double[block1.matrix.length][block1.matrix.length];
		for (int i = 0; i < block1.getMatrix().length; i++) {
			for (int j = 0; j < block1.getMatrix().length; j++) {
				mat3[i][j] = block1.matrix[i][j] - block2.matrix[i][j];
			}
		}
		block.matrix = mat3;
		block.matname = matName + "," + key;

		return block;
	}

	public ArrayList<Block> rearrange(HashMap<String, Block> b1, HashMap<String, Block> b2, HashMap<String, Block> b3,
			HashMap<String, Block> b4, HashMap<String, Block> b5, HashMap<String, Block> b6,
			HashMap<String, Block> b7) {
		ArrayList<Block> blockListC11 = new ArrayList<Block>();
		ArrayList<Block> blockListC12 = new ArrayList<Block>();
		ArrayList<Block> blockListC21 = new ArrayList<Block>();
		ArrayList<Block> blockListC22 = new ArrayList<Block>();

		ArrayList<Block> mergedList = new ArrayList<Block>();
		for (int i = 0; i < Math.sqrt(b1.size()); i++) {
			for (int j = 0; j < Math.sqrt(b1.size()); j++) {
				Block blockM1 = new Block();
				Block blockM2 = new Block();
				Block blockM3 = new Block();
				Block blockM4 = new Block();
				Block blockM5 = new Block();
				Block blockM6 = new Block();
				Block blockM7 = new Block();

				Block blockC11 = new Block();
				Block blockC12 = new Block();
				Block blockC21 = new Block();
				Block blockC22 = new Block();

				blockM1 = b1.get(Integer.toString(i) + Integer.toString(j));
				blockM1.rowIndex = i;
				blockM1.columnIndex = j;
				blockM2 = b2.get(Integer.toString(i) + Integer.toString(j));
				blockM2.rowIndex = i;
				blockM2.columnIndex = j;
				blockM3 = b3.get(Integer.toString(i) + Integer.toString(j));
				blockM3.rowIndex = i;
				blockM3.columnIndex = j;
				blockM4 = b4.get(Integer.toString(i) + Integer.toString(j));
				blockM4.rowIndex = i;
				blockM4.columnIndex = j;
				blockM5 = b5.get(Integer.toString(i) + Integer.toString(j));
				blockM5.rowIndex = i;
				blockM5.columnIndex = j;
				blockM6 = b6.get(Integer.toString(i) + Integer.toString(j));
				blockM6.rowIndex = i;
				blockM6.columnIndex = j;
				blockM7 = b7.get(Integer.toString(i) + Integer.toString(j));
				blockM7.rowIndex = i;
				blockM7.columnIndex = j;

				blockC11 = add(subtract(add(blockM1, blockM4), blockM5), blockM7);
				blockC12 = add(blockM3, blockM5);
				blockC21 = add(blockM2, blockM4);
				blockC22 = add(add(subtract(blockM1, blockM2), blockM3), blockM6);

				blockListC11.add(blockC11);
				blockListC12.add(blockC12);
				blockListC21.add(blockC21);
				blockListC22.add(blockC22);

			}

		}

		int size = (int) Math.sqrt(b1.size());

		for (int i = 0; i < blockListC11.size(); i++) {
			blockListC12.get(i).columnIndex = blockListC12.get(i).columnIndex * 1 + size;
			blockListC21.get(i).rowIndex = blockListC21.get(i).rowIndex * 1 + size;
			blockListC22.get(i).rowIndex = blockListC22.get(i).rowIndex * 1 + size;
			blockListC22.get(i).columnIndex = blockListC22.get(i).columnIndex * 1 + size;
		}

		mergedList.addAll(blockListC11);
		mergedList.addAll(blockListC12);
		mergedList.addAll(blockListC21);
		mergedList.addAll(blockListC22);

		return mergedList;
	}

}
