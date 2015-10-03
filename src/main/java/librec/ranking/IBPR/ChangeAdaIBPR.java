// Copyright (C) 2014 Zhou Ge

package librec.ranking.IBPR;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import coding.io.KeyValPair;
import coding.io.Lists;
import coding.io.Strings;
import coding.math.Randoms;
import coding.math.Stats;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;

/**
 * 
 * 
 * <p>
 * Related Work:
 * <ul>
 * 按照 wsdm 2014年的文章，改进IBPR
 * <li></li>
 * </ul>
 * </p>
 * 
 * @author zhouge
 * 
 */
public class ChangeAdaIBPR extends IterativeRecommender {

	double basis_reg = 0.01;
	/* adaptive factor and and one dim for bias */
	private final int loopNumber_Item, loopNumber_User;
	private double[] var_Item, var_User;
	private int[][] factorRanking_Item, factorRanking_User;
	private double[] RankingPro_Item, RankingPro_User;
	private final double lamda_Item, lamda_User;

	int countIter_user = 0, countIter_Item = 0;

	public ChangeAdaIBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix,
			int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		userBias.init(0.01);
		itemBias.init(0.01);

		// set for adaptive
		loopNumber_Item = (int) (numItems * Math.log(numItems));
		loopNumber_User = (int) (numUsers * Math.log(numUsers));
		lamda_Item = cf.getDouble("lamda_Item_ratio") * numItems;
		lamda_User = cf.getDouble("lamda_User_ratio") * numUsers;

		var_Item = new double[numFactors + 1];
		var_User = new double[numFactors + 1];
		factorRanking_Item = new int[numFactors + 1][numItems];
		factorRanking_User = new int[numFactors + 1][numUsers];
		double sum = 0;

		RankingPro_Item = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			double temp = Math.exp(-(i + 1) / lamda_Item);
			RankingPro_Item[i] = temp;
			sum += temp;
		}
		for (int i = 0; i < numItems; i++) {
			RankingPro_Item[i] /= sum;
		}
		sum = 0;
		RankingPro_User = new double[numUsers];
		for (int i = 0; i < numUsers; i++) {
			double temp = Math.exp(-(i + 1) / lamda_User);
			RankingPro_User[i] = temp;
			sum += temp;
		}
		for (int i = 0; i < numUsers; i++) {
			RankingPro_User[i] /= sum;
		}
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			// iterative in user

			loss = 0;
			errs = 0;
			update();
			if (isConverged(iter))
				break;

		}
	}

	public void update() {
		for (int s = 0, smax = numUsers * 100; s < smax; s++) {

			// randomly draw (u,v, i, j)
			int u = 0, v = 0, i = 0, j = 0;
			
			if (countIter_Item % loopNumber_Item == 0) {
				updateRanking_Item();
				updateRanking_User();
				countIter_Item = 0;
				
			}
			countIter_Item++;
			
			while (true) {
				u = Randoms.uniform(trainMatrix.numRows());
				SparseVector pu = trainMatrix.row(u);

				if (pu.getCount() == 0)
					continue;

				int[] is = pu.getIndex();
				i = is[Randoms.uniform(is.length)];
				
				SparseVector pi = trainMatrix.column(i);
				
				do {
					// randoms get a r by exp(-r/lamda)
					int randomJIndex = 0;
					do {
						randomJIndex = Randoms.discrete(RankingPro_User);
					} while (randomJIndex > numUsers);

					// randoms get a f by p(f|c)
					double[] pfc = new double[numFactors + 1];
					double sumfc = 0;
					for (int index = 0; index <= numFactors; index++) {
						double temp = 0;
						if (index == numFactors) {
							temp = 1;
						} else {
							temp = Math.abs(Q.get(i, index));
						}
						double var = temp * var_User[index];
						sumfc += var;
						pfc[index] = var;
					}
					for (int index = 0; index <= numFactors; index++) {
						pfc[index] /= sumfc;
					}
					int f = Randoms.discrete(pfc);

					// get the r-1 in f item
					if (f == numFactors) {
						v = factorRanking_User[f][randomJIndex];
					} else {
						if (Q.get(i, f) > 0) {
							v = factorRanking_User[f][randomJIndex];
						} else {
							v = factorRanking_User[f][numUsers - randomJIndex
									- 1];
						}
					}

				} while (pi.contains(v));
				
				do {
					// randoms get a r by exp(-r/lamda)
					int randomJIndex = 0;
					do {
						randomJIndex = Randoms.discrete(RankingPro_Item);
					} while (randomJIndex > numItems);

					// randoms get a f by p(f|c)
					double[] pfc = new double[numFactors + 1];
					double sumfc = 0;
					for (int index = 0; index <= numFactors; index++) {
						double temp = 0;
						if (index == numFactors) {
							temp = 1;
						} else {
							temp = Math.abs(P.get(u, index));
						}
						double var = temp * var_Item[index];
						sumfc += var;
						pfc[index] = var;
					}
					for (int index = 0; index <= numFactors; index++) {
						pfc[index] /= sumfc;
					}
					int f = Randoms.discrete(pfc);

					// get the r-1 in f item
					if (f == numFactors) {
						j = factorRanking_Item[f][randomJIndex];
					} else {
						if (P.get(u, f) > 0) {
							j = factorRanking_Item[f][randomJIndex];
						} else {
							j = factorRanking_Item[f][numItems - randomJIndex
									- 1];
						}
					}

				} while (pu.contains(j));

				break;
			}
			// System.out.println(trainMatrix.get(u, i));
			// update parameters
			double xui = predict(u, i);
			double xuj = predict(u, j);
			double xvi = predict(v, i);
			double xuij = xui - xuj;
			double xiuv = xui - xvi;
			double vals = -Math.log(g(xuij));
			loss += vals;
			errs += vals;

			double cmguij = g(-xuij);
			double cmgiuv = g(-xiuv);

			for (int f = 0; f < numFactors; f++) {
				double puf = P.get(u, f);
				double pvf = P.get(v, f);
				double qif = Q.get(i, f);
				double qjf = Q.get(j, f);

				P.add(u, f, lRate
						* (cmguij * (qif - qjf) + cmgiuv * qif - regU * puf));
				P.add(v, f, lRate * (cmgiuv * (-qif) - regU * pvf));

				Q.add(i, f, lRate
						* (cmguij * puf + cmgiuv * (puf - pvf) - regI * qif));
				Q.add(j, f, lRate * (cmguij * (-puf) - regI * qjf));

				loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
			}

			itemBias.add(i, lRate * (cmguij - regI * itemBias.get(i)));
			itemBias.add(j, lRate * (-cmguij - regI * itemBias.get(j)));
			userBias.add(u, lRate * (cmgiuv - regU * userBias.get(u)));
			userBias.add(v, lRate * (-cmgiuv - regU * userBias.get(v)));
		}

	}

	/**
	 * update Ranking in item
	 * 
	 * @create_time：2014年12月25日上午9:34:36
	 * @modifie_time：2014年12月25日 上午9:34:36
	 */
	public void updateRanking_Item() {
		// echo for each factors
		for (int factorIndex = 0; factorIndex < numFactors; factorIndex++) {
			DenseVector factorVector = Q.column(factorIndex).clone();
			List<KeyValPair<Integer>> sort = sortByDenseVectorValue(factorVector);
			double[] valueList = new double[numItems];
			for (int i = 0; i < numItems; i++) {
				factorRanking_Item[factorIndex][i] = sort.get(i).getKey();
				valueList[i] = sort.get(i).getValue();
			}
			// get
			var_Item[factorIndex] = Stats.var(valueList);

		}

		// set for item bias
		DenseVector factorVector = itemBias.clone();
		List<KeyValPair<Integer>> sort = sortByDenseVectorValue(factorVector);
		for (int i = 0; i < numItems; i++) {
			factorRanking_Item[numFactors][i] = sort.get(i).getKey();
		}
		var_Item[numFactors] = Stats.var(itemBias.getData());

	}

	/**
	 * update Ranking in user
	 * 
	 * @create_time：2014年12月25日上午9:34:39
	 * @modifie_time：2014年12月25日 上午9:34:39
	 */
	public void updateRanking_User() {
		// echo for each factors
		for (int factorIndex = 0; factorIndex < numFactors; factorIndex++) {
			DenseVector factorVector = P.column(factorIndex).clone();
			List<KeyValPair<Integer>> sort = sortByDenseVectorValue(factorVector);
			double[] valueList = new double[numUsers];
			for (int i = 0; i < numUsers; i++) {
				factorRanking_User[factorIndex][i] = sort.get(i).getKey();
				valueList[i] = sort.get(i).getValue();
			}
			// get
			var_User[factorIndex] = Stats.var(valueList);
		}

		// Set for user bias
		DenseVector factorVector = userBias.clone();
		List<KeyValPair<Integer>> sort = sortByDenseVectorValue(factorVector);
		for (int i = 0; i < numUsers; i++) {
			factorRanking_User[numFactors][i] = sort.get(i).getKey();
		}
		var_User[numFactors] = Stats.var(userBias.getData());

	}

	public List<KeyValPair<Integer>> sortByDenseVectorValue(DenseVector vector) {
		Map<Integer, Double> keyValPair = new HashMap<>();
		for (int i = 0; i < vector.getSize(); i++) {
			keyValPair.put(i, vector.get(i));
		}
		List<KeyValPair<Integer>> sorted = Lists.sortMap(keyValPair, true);
		return sorted;
	}

	@Override
	public String toString() {
		return Strings.toString(
				new Object[] { binThold, numFactors, initLRate, regU, regI,
						numIters, cf.getDouble("lamda_Item_ratio"),
						cf.getDouble("lamda_User_ratio") }, ",");
	}

	@Override
	public double predict(int u, int j) {
		return userBias.get(u) + itemBias.get(j)
				+ DenseMatrix.rowMult(P, u, Q, j);
	}
}
