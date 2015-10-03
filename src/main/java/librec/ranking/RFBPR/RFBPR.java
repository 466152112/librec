// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General User_factorublic License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A User_factorARTICULAR User_factorURUser_factorOSE. See the
// GNU General User_factorublic License for more details.
//
// You should have received a copy of the GNU General User_factorublic License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.ranking.RFBPR;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import coding.io.Strings;
import coding.math.Randoms;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.VectorEntry;
import librec.intf.SocialRecommender;

/**
 * Social Bayesian User_factorersonalized Ranking (SBUser_factorR)
 * 
 * <p>
 * Zhao et al., <strong>Leveraging Social Connections to Improve
 * User_factorersonalized Ranking for Collaborative Filtering</strong>, CIKM
 * 2014.
 * </p>
 * 
 * @author guoguibing
 * 
 */
public class RFBPR extends SocialRecommender {

	private double[] RankingPro;

	public RFBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		double sum = 0;

		RankingPro = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			double temp = Math.exp(-(i + 1) / 500);
			RankingPro[i] = temp;
			sum += temp;
		}
		for (int i = 0; i < numItems; i++) {
			RankingPro[i] /= sum;
		}
	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();
		P.init(smallValue);
		Q.init(smallValue);
		itemBias = new DenseVector(numItems);
		itemBias.init();

	}

	@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {
			updateUI();
			updateFriend();
			if (isConverged(iter))
				break;
		}

	}

	protected void updateUI() {
		errs = 0;
		for (int u = 0; u < numUsers; u++) {

			// uniformly draw (u, i, j, v, k)
			// 其中 i,j 为商品。 u为当前目标用户， v,为u 的朋友， k 为u非朋友
			int i = 0, j = 0;

			// u
			SparseVector pu = null;
			do {
				u = Randoms.uniform(trainMatrix.numRows());
				pu = trainMatrix.row(u);
			} while (pu.getCount() == 0);

			// i
			int[] is = pu.getIndex();
			i = is[Randoms.uniform(is.length)];

			double xui = predict(u, i);

			// j
			do {
				j = Randoms.uniform(numItems);
			} while (pu.contains(j));
			// u 的所有朋友
			double xuj = predict(u, j);
			// if no social neighbors, the same as BPR
			double xuij = xui - xuj;
			double vals = -Math.log(g(xuij));
			errs += vals;
			loss += vals;

			double cij = g(-xuij);

			// update bi, bj
			double bi = itemBias.get(i);
			itemBias.add(i, lRate * (cij - regB * bi));
			loss += regB * bi * bi;

			double bj = itemBias.get(j);
			itemBias.add(j, lRate * (-cij - regB * bj));
			loss += regB * bj * bj;

			// update User_factor, Item_factor
			for (int f = 0; f < numFactors; f++) {
				double puf = P.get(u, f);
				double qif = Q.get(i, f);
				double qjf = Q.get(j, f);

				P.add(u, f, lRate * (cij * (qif - qjf) - regU * puf));
				Q.add(i, f, lRate * (cij * puf - regI * qif));
				Q.add(j, f, lRate * (cij * (-puf) - regI * qjf));

				loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
			}
		}

	}

	protected void updateFriend() {
		for (int u = 0; u < numUsers; u++) {
			// uniformly draw (u, i, j, v, k)
			// 其中 i,j 为商品。 u为当前目标用户， v,为u 的朋友， k 为u非朋友
			// u
			SparseVector pu = null;
			do {
				u = Randoms.uniform(trainMatrix.numRows());
				pu = trainMatrix.row(u);
			} while (pu.getCount() == 0);
		
			// u 的所有朋友
			SparseVector Fu = socialMatrix.row(u);
			if (Fu.size() > 5) {
				// if having social neighbors
				// random v
				int[] indextemp = Fu.getIndex();
				int v = (int) Fu.get(Randoms.uniform(indextemp.length));

				double yuv = DenseMatrix
						.rowMult(P, u, P, v);
				// random k
				int k = 0;
				do {
					k = Randoms.uniform(numUsers);
				} while (Fu.contains(k) || u == k);

				double yuk = DenseMatrix
						.rowMult(P, u, P, k);

				double yuvk = yuv - yuk;

				double valsuser = -Math.log(g(yuvk));

				double  cvk = g(-valsuser);


				// update User_factor, Item_factor
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double pvf = P.get(v, f);
					double pkf = P.get(k, f);

					double delta_puf = cvk * (pvf - pkf);

					P.add(u, f, lRate * (delta_puf - regU * puf));

					P.add(v, f, lRate * (cvk * puf - regU * pvf));

					P.add(k, f, lRate * (cvk * (-puf) - regU * pkf));
				}
			} else {
				// if no social neighbors, the same as BPR
			}
		}
	}

	@Override
	protected double predict(int u, int j) {
		return itemBias.get(j)
				+ DenseMatrix.rowMult(P, u, Q, j);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate,
				maxLRate, regU, regI, regB, numIters });
	}

}
