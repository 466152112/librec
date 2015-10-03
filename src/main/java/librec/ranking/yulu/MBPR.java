package librec.ranking.yulu;

import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
//MBPR, LearnRatio = 0.006, Reg = 0.05
//在 Epinions 上
//在其他数据上， LearnRatio = 0.005, Reg = 0.05

//MutuUpdateIBPR -> MBPR
//RankMutIBPR -> RankMBPR
public class MBPR extends IterativeRecommender {
	
	public MBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		
		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		userBias.init(0.01);
		itemBias.init(0.01);
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			// iterative in user

			loss = 0;
			errs = 0;
			Update();
			if (isConverged(iter))
				break;

		}
	}

	public void Update(){

			loss = 0;
			errs = 0;
			// randomly draw (u,v, i, j)
			int u = 0, v = 0, j = 0;
			
			for (; u < numUsers; u++){
				
				SparseVector pu = trainMatrix.row(u);

				if (pu.getCount() == 0)
					continue;
				
				int[] is = pu.getIndex();
				
				for(int i : is){

					SparseVector pi=trainMatrix.column(i);
					
					
					while (true) {
						
						do {
							v = Randoms.uniform(numUsers);
						} while (pi.contains(v));

						do {
							j = Randoms.uniform(numItems);
						} while (pu.contains(j));

						break;
					}
					//System.out.println(trainMatrix.get(u, i));
					// update parameters
					double xui = predict(u, i);
					double xuj = predict(u, j);
					double xvi = predict(v, i);
					double xuij = xui - xuj;
					double xiuv = xui-xvi;
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

						P.add(u, f, lRate * (cmguij * (qif - qjf) + cmgiuv*qif - regU * puf));
						P.add(v, f, lRate * (cmgiuv*(-qif) - regU * pvf));
						
						Q.add(i, f, lRate * (cmguij * puf + cmgiuv*(puf-pvf)- regI * qif));
						Q.add(j, f, lRate * (cmguij * (-puf) - regI * qjf));
						
						loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
					}
					
					itemBias.add(i,lRate*(cmguij-regI*itemBias.get(i)));
					itemBias.add(j,lRate*(-cmguij-regI*itemBias.get(j)));
					userBias.add(u,lRate*(cmgiuv-regU*userBias.get(u)));
					userBias.add(v,lRate*(-cmgiuv-regU*userBias.get(v)));
					
					
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
