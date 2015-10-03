// Copyright (C) 2014 Zhou Ge

package librec.ranking.IBPR;


import coding.io.Strings;
import coding.math.Randoms;
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
 * <li></li>
 * </ul>
 * </p>
 * 
 * @author zhouge
 * 
 */
public class StatIBPR extends IterativeRecommender {
	
	final double basis_reg;
		
	public StatIBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		
		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		userBias.init(0.01);
		itemBias.init(0.01);
		basis_reg=cf.getDouble("val.reg.bias");
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			// iterative in user

			loss = 0;
			errs = 0;
			userUpdate();
			itemUpdate();
			// iterative in item
			if (isConverged(iter))
				break;

		}
	}

	public void userUpdate(){
		
		for (int u = 0, smax = numUsers ; u < smax; u++) {

			SparseVector pu = trainMatrix.row(u);
			if (pu.getCount() == 0)
					continue;
			int[] is = pu.getIndex();
			for (int i : is) {
				int j=0;
				do {
					j = Randoms.uniform(numItems);
				} while (pu.contains(j));


				// update parameters
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double ii_basis=itemBias.get(i);
				double ji_basis=itemBias.get(j);
				double xuij = ii_basis+xui - xuj-ji_basis;
				
				double vals = -Math.log(g(xuij));
				loss += vals;
				errs += vals;

				double cmg = g(-xuij);

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);

					P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf - regI * qif));
					Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

					loss += regU * puf * puf + regI * qif * qif + regI * qjf
							* qjf;
				}
				//basis update
				itemBias.set(i, ii_basis+lRate*(cmg-basis_reg*ii_basis));
				itemBias.set(j, ji_basis+lRate*(-cmg-basis_reg*ji_basis));
			}
		}
	}
	
	public void itemUpdate(){
		
		for (int i = 0, smax = numItems ; i < smax; i++) {
			SparseVector pi = trainMatrix.column(i);

			if (pi.getCount() == 0)
				continue;

			int[] us = pi.getIndex();
			for (int u1 : us) {
				int u2=0;
				do {
					u2 = Randoms.uniform(numUsers);
				} while (pi.contains(u2));
				// update parameters
				double xu1 = predict(u1, i);
				double xu2 = predict(u2, i);
				double u1_basis=userBias.get(u1);
				double u2_basis=userBias.get(u2);
				
				double xu12 = u1_basis+xu1 - xu2-u2_basis;

				double vals = -Math.log(g(xu12));
				loss += vals;
				errs += vals;

				double cmg = g(-xu12);

				for (int f = 0; f < numFactors; f++) {

					double qif = Q.get(i, f);
					double pu1f = P.get(u1, f);
					double pu2f = P.get(u2, f);

					Q.add(i, f, lRate * (cmg * (pu1f - pu2f) - regI * qif));
					P.add(u1, f, lRate * (cmg * qif - regU * pu1f));
					P.add(u2, f, lRate * (cmg * (-qif) - regU * pu2f));

					loss += regI * qif * qif + regU * pu1f * pu1f + regI * pu2f
							* pu2f;
				}
				userBias.set(u1, u1_basis+lRate*(cmg-basis_reg*u1_basis));
				userBias.set(u2, u2_basis+lRate*(-cmg-basis_reg*u2_basis));
			}
			}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate,
				regU, regI, numIters }, ",");
	}
	
	@Override
	public double  predict(int u, int j){
		return userBias.get(u)+itemBias.get(j)+DenseMatrix.rowMult(P, u, Q, j);
	}
}
