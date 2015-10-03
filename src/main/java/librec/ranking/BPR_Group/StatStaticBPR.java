// Copyright (C) 2014 zhouge
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//
package librec.ranking.BPR_Group;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import coding.io.KeyValPair;
import coding.io.Lists;
import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;

/**
 * 
 * Rendle S, Improving pairwise learning for item recommendation from implicit
 * feedback[C] WSDM 2014
 * 
 * 
 * @author zhouge
 * 
 */
public class StatStaticBPR extends IterativeRecommender {
	private int[] popularityRanking;
	private double[] RankingPro;
	private final double lamda_Item;

	public StatStaticBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix,
			int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;

		lamda_Item = (int) (cf.getDouble("lamda_Item_ratio") * numItems);
		// set for this alg
		popularityRanking = new int[numItems];
		double sum = 0;
		RankingPro = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			double temp = Math.exp(-(i + 1) / lamda_Item);
			RankingPro[i] = temp;
			sum += temp;
		}
		for (int i = 0; i < numItems; i++) {
			RankingPro[i] /= sum;
		}

		// sort by PopularityValue
		sortByPopularityValue();
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			errs = 0;
			double poscount=0,negcount=0,count=0;
			for (int s = 0, smax = numUsers; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = s, j = 0;
				SparseVector pu = trainMatrix.row(u);
				if (pu.getCount() == 0)
					continue;
				int[] is = pu.getIndex();
				DenseVector puitem=new DenseVector(numFactors);
				for (int i : is) {
					puitem=puitem.add(Q.row(i));
				}
				puitem=puitem.scale(1.0/is.length);
				
				for (int i : is) {
					do {
						// randoms get a r by exp(-r/lamda)
						int randomJIndex = 0;
						randomJIndex = Randoms.discrete(RankingPro);
						j = popularityRanking[randomJIndex];
					} while (pu.contains(j));
					count++;
					double mult=puitem.inner(Q.row(j));
					if (mult>0) {
						poscount++;
					}else if (mult<0) {
						negcount++;
					}
					// update parameters
					double xui = predict(u, i);
					double xuj = predict(u, j);
					double xuij = xui - xuj;

					double vals = -Math.log(g(xuij));
					loss += vals;
					errs += vals;

					double cmg = g(-xuij);

					for (int f = 0; f < numFactors; f++) {
						double puf = P.get(u, f);
						double qif = Q.get(i, f);
						double qjf = Q.get(j, f);

						P.add(u, f, lRate
								* (cmg * (qif - qjf) - regU * puf));
						Q.add(i, f, lRate * (cmg * puf - regI * qif));
						Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

						loss += regU * puf * puf + regI * qif * qif + regI * qjf
								* qjf;
					}
				}
			}
			System.out.println(iter+" "+"  "+(negcount/count));
//			if (isConverged(iter))
//				break;

		}
	}

	public void sortByPopularityValue() {
		Map<Integer, Double> keyValPair = new HashMap<>();
		for (int i = 0; i < numItems; i++) {
			SparseVector pi = trainMatrix.column(i);
			keyValPair.put(i, pi.getCount() * 1.0);
		}
		List<KeyValPair<Integer>> sorted = Lists.sortMap(keyValPair, true);
		for (int i = 0; i < numItems; i++) {
			popularityRanking[i] = sorted.get(i).getKey();
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate,
				regU, regI, numIters, cf.getDouble("lamda_Item_ratio") }, ",");
	}
}
