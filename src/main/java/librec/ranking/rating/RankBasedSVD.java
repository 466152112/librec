// Copyright (C) 2015 
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

package librec.ranking.rating;

import coding.eval.RankEvaluator;
import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.intf.RankingRecommender;

/**   
*    
* project_name：librec_zhouge   
* type_name：RankBasedSVD   
* type_description：   
* creator：zhoug_000   
* create_time：2015年5月3日 下午11:15:27   
* modification_user：zhoug_000   
* modification_time：2015年5月3日 下午11:15:27   
* @version    
*    
*/
public class RankBasedSVD extends RankingRecommender {
	private static final long serialVersionUID = 4001;
	private static int[] s_u;
	

	public RankBasedSVD(SparseMatrix trainMatrix, SparseMatrix testMatrix,
			int fold) {
		super(trainMatrix, testMatrix, fold);

	}

	@Override
	protected void initModel() throws Exception {
		// initialization
		super.initModel();

		itemBias = new DenseVector(numItems);
		itemBias.init();

		// Preparing data structure:
		s_u = new int[numUsers];
		for (int u = 0; u < numUsers; u++) {
			s_u[u] = 0;
			int[] itemList = trainMatrix.row(u).getIndex();
			if (itemList != null) {
				for (int i : itemList) {
					for (int j : itemList) {
						if (trainMatrix.get(u, i) > trainMatrix.get(u, j)) {
							s_u[u]++;
						}
					}
				}
			}
		}

	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 0; iter < numIters; iter++) {
			// Gradient Descent:
			for (int u = 0; u < numUsers; u++) {
				SparseVector pu = trainMatrix.row(u);
				DenseVector uchange = new DenseVector(numFactors);
				int[] itemIndexList = pu.getIndex();
				if (itemIndexList != null) {
					// item-pair
					for (int i : itemIndexList) {
						double pred_i = predict(u, i);
						DenseVector ichange = new DenseVector(numFactors);
						double Mui = pu.get(i);
						for (int j : itemIndexList) {
							double pred_j = predict(u, j);
							double Muj = pu.get(j);

							if (Mui > Muj) {
								double dg = RankEvaluator.lossDiff(Mui, Muj,pred_i, pred_j, lossCode);
								uchange=uchange.add(Q.row(i).minus(Q.row(j)).scale(dg));
								ichange=ichange.add(P.row(u).scale(dg));
							}

							else if (Mui < Muj) {
								double dg = RankEvaluator.lossDiff(Muj, Mui, pred_j, pred_i, lossCode);
								ichange=ichange.minus(P.row(u).scale(dg));
							}
						}
						ichange = ichange.scale(1.0 / (numUsers * s_u[u]));
						for (int factor = 0; factor < numFactors; factor++) {
							Q.add(i, factor,-lRate* (ichange.get(factor) + 2 * regI* Q.get(i, factor)));
						}
					}
				}
				uchange = uchange.scale(1.0 / (numUsers * s_u[u]));
				for (int factor = 0; factor < numFactors; factor++) {
					P.add(u,factor,	-lRate* (uchange.get(factor) + 2 * regU* P.get(u, factor)));
				}
			}

			if (isConverged(iter))
				break;
		}
	}

	/**
	 * default prediction method
	 */
	@Override
	protected double predict(int u, int j) {
		return DenseMatrix.rowMult(P, u, Q, j);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate,
				regU, regI, numIters }, ",");
	}
}
