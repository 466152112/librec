

// Copyright (C) 2014 Guibing Guo
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
package librec.ranking.tag;

import java.util.ArrayList;
import java.util.List;

import coding.io.Strings;
import coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;

/**
 * 
 * Rendle et al., <strong>Pairwise Interaction Tensor Factorization for Personalized Tag Recommendation</strong>, WSDM 2010
 * 
 * @author zhouge
 * 
 */
public class CD_BPR extends TagRecommender {

	public CD_BPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		tag_factor=new DenseMatrix(numtag,numFactors);
		tag_factor.init(1.0);
	}

	@Override
	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {

			loss = 0;
			errs = 0;
			for (int s = 0, smax = numUsers*numtag * 100; s < smax; s++) {

				// randomly draw (u, i, ta,tb)
				int u = 0, i = 0,j=0, t = 0;
				String userstr,itemstri,itemstrj,tagstr;
				while (true) {
					//random user
					u = Randoms.uniform(numUsers);
					userstr=rateDao.getUserId(u);
					
					//random tag
					List<String> tagSet=new ArrayList<>(trainMap.get(userstr).keySet());
					tagstr=tagSet.get(Randoms.uniform(tagSet.size()));
					t=getTagids(tagstr);
					
					//random itemi
					List<String> itemset=new ArrayList<>(trainMap.get(userstr).get(tagstr));
					itemstri=itemset.get(Randoms.uniform(itemset.size()));
					i=rateDao.getItemId(itemstri);
					
					//random itemj
					do {
						j=Randoms.uniform(numItems);
						itemstrj=rateDao.getItemId(j);
					} while (trainMap.get(userstr).get(tagstr).contains(itemstrj));

					break;
				}

				// update parameters
				double xuit = predict(u,i,t);
				double xujt= predict(u,j,t);
				double xuijt = xuit - xujt;

				double vals = -Math.log(g(xuijt));
				loss += vals;
				errs += vals;

				double cmg = g(-xuijt);

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qif = Q.get(i, f);
					double qjf = Q.get(j, f);
					double tf = tag_factor.get(t, f);
					
					P.add(u, f, lRate * (cmg * tf*(qif-qjf) - regU * puf));
					Q.add(i, f, lRate * (cmg * puf*tf  - regI * qif));
					Q.add(j, f, lRate * (cmg * puf*tf*(-1) - regI * qjf));
					tag_factor.add(t, f, lRate * (cmg * puf*(qif-qjf) - regI * tf));
				}
			}

			if (isConverged(iter))
				break;

		}
	}
	
	protected double predict(int user, int item, int tag) {
		DenseVector uvectro=P.row(user);
		DenseVector itemvector=Q.row(item);
		DenseVector tagvector=tag_factor.row(tag);
		double result=0;
		for (int i = 0; i < numFactors; i++) {
			result+=uvectro.get(i)*itemvector.get(i)*tagvector.get(i);
		}
		
		return result;
	}
	
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
	}
}
