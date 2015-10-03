// Copyright (C) 2014-2005 Guibing Guo
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
package librec.ranking.temporal;


import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.RatingContext;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import coding.io.Logs;
import coding.io.Strings;
import coding.math.Randoms;


/**
 * Koren, <strong>Collaborative Filtering with Temporal Dynamics</strong>, KDD 2009.
 * 
 * @author zhouge
 * 
 */
public class TimeRanking extends TemporalRecommender {
	// item's implicit influence
		protected DenseMatrix T;
	// 
		protected static DenseVector lambdau;
		
		protected static double lambdasuper=0.95;

	// read context information
	static {
		try {
			readContext();
		} catch (Exception e) {
			Logs.error(e.getMessage());
			e.printStackTrace();
		}
	}

	public TimeRanking(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		algoName = "timeRanking";
		System.out.println(lambdasuper);
	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();

		userBias = new DenseVector(numUsers);
		userBias.init();
		lambdau=new DenseVector(numUsers, lambdasuper);
		numBins=100;
		T = new DenseMatrix(numBins, numFactors);
		T.init();
		
	}

	@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {
			errs = 0;
			loss = 0;
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
				//System.out.println(trainMatrix.get(u, i));
				// update parameters
				RatingContext rc=ratingContexts.get(u, i);
				int t=bin(days(rc.getTimestamp(),minTimestamp));
				double xui = predict(u, i,t);
				double xuj = predict(u, j,t);
				double xuij = xui - xuj;
				
				
				
				double vals = -Math.log(g(xuij));
				loss += vals;
				errs += vals;

				double cmg = g(-xuij);
				double lambda=lambdau.get(u);
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);
					double Ttf = T.get(t, f);
					P.add(u, f, lRate * (cmg * (lambda*qif - lambda*qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg *( lambda*puf+(1-lambda)*Ttf )- regI * qif));
					Q.add(j, f, lRate * (cmg * (-lambda*puf-(1-lambda)*Ttf) - regI * qjf));
					T.add(t, f, lRate * (cmg * ((1-lambda)*qif-(1-lambda)*qif) - regI * Ttf));
					
					loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
				}
			}
			
			if (isConverged(iter))
				break;
		}
	}

	protected double predict(int u, int i,int t)  {
		double result=0;
		double lambda=lambdau.get(u);
		result=lambda*DenseMatrix.rowMult(P, u, Q, i)+(1-lambda)*DenseMatrix.rowMult( Q, i,T, t);
		return result;
	}

	@Override
	public String toString() {
		return super.toString() + "," + Strings.toString(new Object[] {  numBins,lambdasuper });
	}
	
}
