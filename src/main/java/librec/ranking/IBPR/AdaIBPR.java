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
 * <ul> 按照 wsdm 2014年的文章，改进IBPR
 * <li></li>
 * </ul>
 * </p>
 * 
 * @author zhouge
 * 
 */
public class AdaIBPR extends IterativeRecommender {
	
	double basis_reg=0.01;
	/* adaptive factor and and one dim for bias */
	private final int loopNumber_Item,loopNumber_User;
	private double[] var_Item,var_User;
	private int[][] factorRanking_Item,factorRanking_User;
	private double[] RankingPro_Item,RankingPro_User;
	private final double lamda_Item,lamda_User;
	
	int countIter_user=0, countIter_Item=0;
	
			
	public AdaIBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		
		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		userBias.init(0.01);
		itemBias.init(0.01);
		

		//set for adaptive
		loopNumber_Item=(int)(numItems*Math.log(numItems));
		loopNumber_User=(int)(numUsers*Math.log(numUsers));
		lamda_Item=cf.getDouble("lamda_Item_ratio")*numItems;
		lamda_User=cf.getDouble("lamda_User_ratio")*numUsers;
		
		var_Item=new double[numFactors+1];var_User=new double[numFactors+1];
		factorRanking_Item=new int[numFactors+1][numItems];factorRanking_User=new int[numFactors+1][numUsers];
		double sum=0;
		
		RankingPro_Item=new double[numItems];
		for (int i = 0; i < numItems; i++) {
			double temp=Math.exp(-(i+1)/lamda_Item);
			RankingPro_Item[i]=temp;
			sum+=temp;
		}
		for (int i = 0; i < numItems; i++) {
			RankingPro_Item[i]/=sum;
		}
		sum=0;
		RankingPro_User=new double[numUsers];
		for (int i = 0; i < numUsers; i++) {
			double temp=Math.exp(-(i+1)/lamda_User);
			RankingPro_User[i]=temp;
			sum+=temp;
		}
		for (int i = 0; i < numUsers; i++) {
			RankingPro_User[i]/=sum;
		}
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			// iterative in user

			loss = 0;
			errs = 0;
			System.out.print(iter );
			userUpdate();
			itemUpdate();
			// iterative in item
//			if (isConverged(iter))
//				break;

		}
	}

	public void userUpdate(){
		double poscount=0,negcount=0,count=0;
		for (int u = 0, smax = numUsers ; u < smax; u++) {

			SparseVector pu = trainMatrix.row(u);
			if (pu.getCount() == 0)
					continue;
			int[] is = pu.getIndex();
			DenseVector puitem=new DenseVector(numFactors);
			double avgbias=0;
			for (int i : is) {
				puitem=puitem.add(Q.row(i));
				avgbias+=itemBias.get(i);
			}
			puitem=puitem.scale(1.0/is.length);
			avgbias/=is.length;
			for (int i : is) {
				
				if (countIter_Item%loopNumber_Item==0) {
					updateRanking_Item();
					countIter_Item=0;
				}
				countIter_Item++;
				
				int j=0;
				do {
					//randoms get a r by exp(-r/lamda)
					int randomJIndex=0;
					do {
						randomJIndex=Randoms.discrete(RankingPro_Item);
					} while (randomJIndex>numItems);
					
					//randoms get a f by p(f|c)
					double[] pfc=new double[numFactors+1];
					double sumfc=0;
					for (int index = 0; index <= numFactors; index++) {
						double temp=0;
						if (index==numFactors) {
							temp=1;
						}else {
							temp=Math.abs(P.get(u, index));
						}
						double var=temp*var_Item[index];
						sumfc+=var;
						pfc[index]=var;
					}
					for (int index = 0; index <= numFactors; index++) {
						pfc[index]/=sumfc;
					}
					int f=Randoms.discrete(pfc);
					
					//get the r-1 in f item
					if (f==numFactors) {
						j=factorRanking_Item[f][randomJIndex];
					}else {
						if (P.get(u, f)>0) {
							j=factorRanking_Item[f][randomJIndex];
						}else {
							j=factorRanking_Item[f][numItems-randomJIndex-1];
						}
					}
					
				} while (pu.contains(j));

				count++;
				double mult=puitem.inner(Q.row(j))+avgbias*itemBias.get(j);
				if (mult>0) {
					poscount++;
				}else if (mult<0) {
					negcount++;
				}

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
		System.out.println(" "+"  "+(negcount/count));
	}
	
	public void itemUpdate(){
		
		for (int i = 0, smax = numItems ; i < smax; i++) {
			SparseVector pi = trainMatrix.column(i);

			if (pi.getCount() == 0)
				continue;

			int[] us = pi.getIndex();
			for (int u1 : us) {
				
				if (countIter_user%loopNumber_User==0) {
					updateRanking_User();
					countIter_user=0;
				}
				countIter_user++;
				
				int u2=0;
				do {
					//randoms get a r by exp(-r/lamda)
					int randomJIndex=0;
					do {
						randomJIndex=Randoms.discrete(RankingPro_User);
					} while (randomJIndex>numUsers);
					
					//randoms get a f by p(f|c)
					double[] pfc=new double[numFactors+1];
					double sumfc=0;
					for (int index = 0; index <=numFactors; index++) {
						double temp=0;
						if (index==numFactors) {
							temp=1;
						}else {
							temp=Math.abs(Q.get(i, index));
						}
						double var=temp*var_User[index];
						sumfc+=var;
						pfc[index]=var;
					}
					for (int index = 0; index <= numFactors; index++) {
						pfc[index]/=sumfc;
					}
					int f=Randoms.discrete(pfc);
					
					//get the r-1 in f item
					if (f==numFactors) {
						u2=factorRanking_User[f][randomJIndex];
					}else {
						if (Q.get(i, f)>0) {
							u2=factorRanking_User[f][randomJIndex];
						}else {
							u2=factorRanking_User[f][numUsers-randomJIndex-1];
						}
					}
					
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
	
	/**
	 * update Ranking in item
	 *@create_time：2014年12月25日上午9:34:36
	 *@modifie_time：2014年12月25日 上午9:34:36
	  
	 */
	public void updateRanking_Item(){
		//echo for each factors
		for (int factorIndex = 0; factorIndex < numFactors; factorIndex++) {
			DenseVector factorVector=Q.column(factorIndex).clone();
			List<KeyValPair<Integer>> sort=sortByDenseVectorValue(factorVector);
			double[] valueList=new double[numItems];
			for (int i = 0; i < numItems; i++) {
				factorRanking_Item[factorIndex][i]=sort.get(i).getKey();
				valueList[i]=sort.get(i).getValue();
			}
			//get 
			var_Item[factorIndex]=Stats.var(valueList);
			
		}
		
		//set for item bias
		DenseVector factorVector=itemBias.clone();
		List<KeyValPair<Integer>> sort=sortByDenseVectorValue(factorVector);
		for (int i = 0; i < numItems; i++) {
			factorRanking_Item[numFactors][i]=sort.get(i).getKey();
		}
		var_Item[numFactors]=Stats.var(itemBias.getData());
		
	}
	
	/**
	 * update Ranking in user
	 *@create_time：2014年12月25日上午9:34:39
	 *@modifie_time：2014年12月25日 上午9:34:39
	  
	 */
	public void updateRanking_User(){
		//echo for each factors
		for (int factorIndex = 0; factorIndex < numFactors; factorIndex++) {
			DenseVector factorVector=P.column(factorIndex).clone();
			List<KeyValPair<Integer>> sort=sortByDenseVectorValue(factorVector);
			double[] valueList=new double[numUsers];
			for (int i = 0; i < numUsers; i++) {
				factorRanking_User[factorIndex][i]=sort.get(i).getKey();
				valueList[i]=sort.get(i).getValue();
			}
			//get 
			var_User[factorIndex]=Stats.var(valueList);
		}
		
		// Set for user bias
		DenseVector factorVector=userBias.clone();
		List<KeyValPair<Integer>> sort=sortByDenseVectorValue(factorVector);
		for (int i = 0; i < numUsers; i++) {
			factorRanking_User[numFactors][i]=sort.get(i).getKey();
		}
		var_User[numFactors]=Stats.var(userBias.getData());
		
	}
	
	public List<KeyValPair<Integer>> sortByDenseVectorValue(DenseVector vector){
		Map<Integer, Double> keyValPair=new HashMap<>(); 
		for (int i = 0; i < vector.getSize(); i++) {
			keyValPair.put(i, vector.get(i));
		}
		List<KeyValPair<Integer>> sorted = Lists.sortMap(keyValPair, true);
		return sorted;
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate,
				regU, regI, numIters,cf.getDouble("lamda_Item_ratio"),cf.getDouble("lamda_User_ratio") }, ",");
	}
	
	@Override
	public double  predict(int u, int j){
		return userBias.get(u)+itemBias.get(j)+DenseMatrix.rowMult(P, u, Q, j);
	}
}
