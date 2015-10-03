// Copyright (C) 2014 Guibing Guo
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

package librec.ranking;


import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;

/**
 * 
 * Rendle et al., <strong>BPR: Bayesian Personalized Ranking from Implicit Feedback</strong>, UAI 2009.
 * 
 * <p>
 * Related Work:
 * <ul>
 * <li>Gantner et al., Learning Attribute-to-Feature Mappings for Cold-Start Recommendations, ICDM 2010.</li>
 * </ul>
 * </p>
 * 
 * @author zhouge
 * 
 */
public class MMMF extends IterativeRecommender {
	final double basis_reg;
	public MMMF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		basis_reg=cf.getDouble("val.reg.bias");
		itemBias = new DenseVector(numItems);
		itemBias.init(0.01);
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			errs = 0;
			for (int s = 0, smax = numUsers * 100; s < smax; s++) {

				// randomly draw (u, i, j)
				int u = 0, i = 0, j = 0;

				while (true) {
					u = Randoms.uniform(trainMatrix.numRows());
					SparseVector pu = trainMatrix.row(u);

					if (pu.getCount() == 0)
						continue;

					int[] is = pu.getIndex();
					i = is[Randoms.uniform(is.length)];

					do {
						j = Randoms.uniform(numItems);
					} while (pu.contains(j));

					break;
				}

				// update parameters
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double xuij = itemBias.get(i)-itemBias.get(j)+xui - xuj;

				double vals = -Math.log(g(xuij));
				loss += vals;
				errs += vals;

				double cmg =xuij < 0 ? 1 : 0;
				//basis update
				itemBias.set(i, itemBias.get(i)+lRate*(cmg-basis_reg*itemBias.get(i)));
				itemBias.set(j, itemBias.get(j)+lRate*(-cmg-basis_reg*itemBias.get(j)));
				    
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);

					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
			}

			if (isConverged(iter))
				break;

		}
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, regU, regI,basis_reg, numIters }, ",");
	}
}
