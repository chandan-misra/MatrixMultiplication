package in.ac.iitkgp.atdc;

import java.awt.print.Printable;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.rdd.RDD;
import org.jblas.Decompose;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import scala.Tuple2;

public class BlockStrassen implements Serializable {

	/**
	 * Breaks the matrix of type [[BlockMatrix]] into four equal sized sub-matrices.
	 * Each block of each sub-matrix gets a tag or key and relative index inside
	 * that sub-matrix.
	 * 
	 * @param A    The input matrix of type [[BlockMatrix]].
	 * @param ctx  The JavaSparkContext of the job.
	 * @param size size size The size of the matrix in terms of number of
	 *             partitions. If the dimension of the matrix is n and the dimension
	 *             of each block is m, the value of size is = n/m.
	 * @return PairRDD `pairRDD` of [[<String, BlockMatrix.RDD>]] type. Each tuple
	 *         consists of a tag corresponds to block's coordinate and the RDD of
	 *         blocks.
	 */

	private static JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> breakMat(BlockMatrix A, String matName,
			JavaSparkContext ctx, int size) {

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rdd = A.blocks().toJavaRDD();
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = rdd.mapToPair(
				new PairFunction<Tuple2<Tuple2<Object, Object>, Matrix>, String, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> call(
							Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {

						Tuple2<Object, Object> tuple1 = tuple._1;
						int rowIndex = tuple1._1$mcI$sp();
						int colIndex = tuple1._2$mcI$sp();
						Matrix matrix = tuple._2;
						String tag = "";
						if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = matName + "11";
						} else if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 1) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = matName + "12";
						} else if (rowIndex / bSize.value() == 1 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = matName + "21";
						} else {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = matName + "22";
						}
						return new Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>(tag,
								new Tuple2(new Tuple2(rowIndex, colIndex), matrix));
					}

				});

		return pairRDD;
	}

	/**
	 * Returns the upper-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _11(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD, String label,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals(label);
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the upper-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _12(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD, String label,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals(label);
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the lower-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _21(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD, String label,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals(label);
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	/**
	 * Returns the lower-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD   The PairRDD of a broken RDD with a tag for each block.
	 * @param ctx       The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */

	private static BlockMatrix _22(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD, String label,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals(label);
					}
				})
				.map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						return tuple._2;
					}

				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}

	private static BlockMatrix reArrange(JavaSparkContext ctx, BlockMatrix C11, BlockMatrix C12, BlockMatrix C21,
			BlockMatrix C22, int size, int blockSize) {
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C11_RDD = C11.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C12_RDD = C12.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21_RDD = C21.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22_RDD = C22.blocks().toJavaRDD();
		
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C12Arranged = C12_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp();
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21Arranged = C21_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22Arranged = C22_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> union = C11_RDD.union(C12Arranged.union(C21Arranged.union(C22Arranged)));
		BlockMatrix C = new BlockMatrix(union.rdd(), blockSize, blockSize);
		return C;
	}

	private static BlockMatrix scalerMul(JavaSparkContext ctx, BlockMatrix A, final double scalar, int blockSize) {
		final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A_RDD = A.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> B_RDD = A_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int blockSize = bblockSize.getValue();
						Tuple2<Tuple2<Object, Object>, Matrix> tuple2 = tuple;
						int rowIndex = tuple2._1._1$mcI$sp();
						int colIndex = tuple2._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						DoubleMatrix candidate = new DoubleMatrix(matrix.toArray());
						DoubleMatrix product = candidate.muli(scalar);
						matrix = Matrices.dense(blockSize, blockSize, product.toArray());
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		BlockMatrix product = new BlockMatrix(B_RDD.rdd(), blockSize, blockSize);
		return product;

	}

	public void print(BlockMatrix blockMat) {

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blockRDD = blockMat.blocks().toJavaRDD();
		List<Tuple2<Tuple2<Object, Object>, Matrix>> blockList = blockRDD.collect();

		for (int i = 0; i < blockList.size(); i++) {
			System.out.println("[" + blockList.get(i)._1._1$mcI$sp() + ":" + blockList.get(i)._1._2$mcI$sp() + "]");

			int numRows = blockList.get(i)._2.numRows();
			int numCols = blockList.get(i)._2.numCols();
			for (int j = 0; j < numRows; j++) {
				for (int k = 0; k < numCols; k++) {
					System.out.println(blockList.get(i)._2.apply(j, k));
				}
			}

		}
	}

	public BlockMatrix multiply(JavaSparkContext sc, BlockMatrix A, BlockMatrix B, int partitionSize, int blockSize) {

		if (partitionSize == 1) {
			Matrix a = A.blocks().toJavaRDD().collect().get(0)._2;
			Matrix b = B.blocks().toJavaRDD().collect().get(0)._2;
			DoubleMatrix blasA = new DoubleMatrix(blockSize, blockSize, a.toArray());
			DoubleMatrix blasB = new DoubleMatrix(blockSize, blockSize, b.toArray());
			DoubleMatrix blasC = blasA.mmul(blasB);
			Matrix product = Matrices.dense(a.numRows(), b.numCols(), blasC.toArray());
			Tuple2<Tuple2<Object, Object>, Matrix> tuple = new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2(0, 0),
					product);
			ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>> list = new ArrayList<Tuple2<Tuple2<Object, Object>, Matrix>>();
			list.add(tuple);
			BlockMatrix C = new BlockMatrix(sc.parallelize(list).rdd(), blockSize, blockSize);
			return C;
		} else {
			partitionSize = partitionSize / 2;
			JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDDA = BlockStrassen.breakMat(A, "A", sc,
					partitionSize);
			BlockMatrix A11 = BlockStrassen._11(pairRDDA, "A11", sc, blockSize);
			BlockMatrix A12 = BlockStrassen._12(pairRDDA, "A12", sc, blockSize);
			BlockMatrix A21 = BlockStrassen._21(pairRDDA, "A21", sc, blockSize);
			BlockMatrix A22 = BlockStrassen._22(pairRDDA, "A22", sc, blockSize);

			JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDDB = BlockStrassen.breakMat(B, "B", sc,
					partitionSize);
			BlockMatrix B11 = BlockStrassen._11(pairRDDB, "B11", sc, blockSize);
			BlockMatrix B12 = BlockStrassen._12(pairRDDB, "B12", sc, blockSize);
			BlockMatrix B21 = BlockStrassen._21(pairRDDB, "B21", sc, blockSize);
			BlockMatrix B22 = BlockStrassen._22(pairRDDB, "B22", sc, blockSize);

			BlockMatrix I = A11.add(A22);
			BlockMatrix II = B11.add(B22);
			BlockMatrix III = A21.add(A22);
			BlockMatrix IV = B12.subtract(B22);
			BlockMatrix V = B21.subtract(B11);
			BlockMatrix VI = A11.add(A12);
			BlockMatrix VII = A21.subtract(A11);
			BlockMatrix VIII = B11.add(B12);
			BlockMatrix IX = A12.subtract(A22);
			BlockMatrix X = B21.add(B22);

			BlockMatrix M1 = multiply(sc, I, II, partitionSize, blockSize);
			BlockMatrix M2 = multiply(sc, III, B11, partitionSize, blockSize);
			BlockMatrix M3 = multiply(sc, A11, IV, partitionSize, blockSize); print(M3);
			BlockMatrix M4 = multiply(sc, A22, V, partitionSize, blockSize); print(M4);
			BlockMatrix M5 = multiply(sc, VI, B22, partitionSize, blockSize); print(M5);
			BlockMatrix M6 = multiply(sc, VII, VIII, partitionSize, blockSize);
			BlockMatrix M7 = multiply(sc, IX, X, partitionSize, blockSize);
			
			BlockMatrix XI = M1.add(M4);
			BlockMatrix XII = XI.subtract(M5);
			print(XI);
			BlockMatrix C11 = XII.add(M7);
			print(C11);
			BlockMatrix C12 = M3.add(M5);
			print(C12);
			BlockMatrix C21 = M2.add(M4);
			print(C21);
			BlockMatrix XIII = M1.subtract(M2);
			print(XIII);
			BlockMatrix XIV = XIII.add(M3);
			print(XIV);
			BlockMatrix C22 = XIV.add(M6);
			print(C22);
			BlockMatrix C = BlockStrassen.reArrange(sc, C11, C12, C21, C22,partitionSize, blockSize);
			print(C);
			return C;

		}

	}

	public BlockMatrix getSquareMatrix(JavaSparkContext sc, String path, int blockSize) {
		JavaRDD<String> lines = sc.textFile(path);
		JavaRDD<MatrixEntry> mat = lines.map(new Function<String, MatrixEntry>() {

			@Override
			public MatrixEntry call(String line) throws Exception {
				long row = Long.parseLong(line.split(",")[0]);
				long column = Long.parseLong(line.split(",")[1]);
				double value = Double.parseDouble(line.split(",")[2]);

				MatrixEntry entry = new MatrixEntry(row, column, value);
				return entry;
			}
		});
		CoordinateMatrix coordinateMatrix = new CoordinateMatrix(mat.rdd());
		BlockMatrix matrix = coordinateMatrix.toBlockMatrix(blockSize, blockSize);
		return matrix;
	}

	public static void main(String args[]) {

		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);

		String fileNameA = args[0];
		String fileNameB = args[1];
		int blockSize = Integer.parseInt(args[2]);

		String pathA = "E:\\" + fileNameA + ".csv";
		String pathB = "E:\\" + fileNameB + ".csv";

		JavaRDD<String> lines = sc.textFile(pathA);
		int size = (int) Math.sqrt(lines.count());
		int partitionSize = size / blockSize;

		BlockStrassen blockStrassen = new BlockStrassen();
		BlockMatrix A = blockStrassen.getSquareMatrix(sc, pathA, blockSize);
		BlockMatrix B = blockStrassen.getSquareMatrix(sc, pathB, blockSize);
		BlockMatrix product = blockStrassen.multiply(sc, A, B, partitionSize, blockSize);
		// product.blocks().count();
		sc.close();
	}

}
