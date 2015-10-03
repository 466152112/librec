package librec.ranking.tag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import coding.io.KeyValPair;
import coding.io.Lists;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.data.VectorEntry;
import librec.intf.TagRecommender;

public class FolkRank extends TagRecommender {
	// use for
	final static double lambda = 0.85;
	// max iter
	final static int maxiter = 100;
	// use for check if convergence
	final static double minerror = 0.001;
	// totalnode=user+tag+item
	static int totalNode;
	Map<Integer, Map<Integer, Double>> transfer;
	static DenseVector p0;

	public FolkRank(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true;
		initByNorm = false;
		totalNode = numUsers + numItems + numtag;

	}

	@Override
	protected void initModel() throws Exception {
		super.initModel();
		//

		// first construct co-frequency matrix
		// the order of entry in matrix is user, tag, item
		SymmMatrix A = new SymmMatrix(totalNode);
		for (String userstr : trainMap.keySet()) {
			int userid = rateDao.getUserId(userstr);
			for (String tagstr : trainMap.get(userstr).keySet()) {
				int tagid = getTagids(tagstr) + numUsers;
				// (u,t)
				A.set(userid, tagid, 1.0 * trainMap.get(userstr).get(tagstr)
						.size());
				for (String itemstr : trainMap.get(userstr).get(tagstr)) {
					int itemid = rateDao.getItemId(itemstr) + numUsers + numtag;
					// (u,i) (t,i)
					A.set(userid, itemid, A.get(userid, itemid) + 1);
					A.set(tagid, itemid, A.get(tagid, itemid) + 1);
				}
			}
		}

		// then construct the transfer matrix
		SymmMatrix tempMatrix = A;
		
		double[] degree = new double[totalNode];
		for (int row = 0; row < totalNode; row++) {
			SparseVector degrevector = tempMatrix.row(row);
			for (VectorEntry entry : degrevector) {
				degree[row] += entry.get();
			}
		}
		transfer=new HashMap<>();
		for (int row = 0; row < totalNode; row++) {
			for (int column =0; column < totalNode; column++) {
				double val = tempMatrix.get(row, column);
				if(val!=0){
					if (!transfer.containsKey(row)) {
						Map<Integer, Double> temp=new HashMap<>();
						transfer.put(row, temp);
					}
					transfer.get(row).put(column, val / degree[row]);
				}
			}
		}

		// init p0
		p0 = new DenseVector(totalNode, 1.0 / totalNode);
	}

	@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {
			if (isConverged(iter))
				break;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see librec.intf.TagRecommender#getTopkItem(java.util.List, int, int)
	 */
	public List<String> getTopkItem(final List<String> candItems, final int u,
			final int tag) {
		//System.out.println("begin getTopkItem"+u+" "+tag);
		// init
		DenseVector p1 = new DenseVector(totalNode, 1.0 / (2 * numUsers + 2
				* numtag + numItems));
		p1.set(u, (1.0 + numUsers) / (2 * numUsers + 2 * numtag + numItems));
		p1.set(tag + numUsers, (1.0 + numtag)
				/ (2 * numUsers + 2 * numtag + numItems));
		
		DenseVector w0 = new DenseVector(totalNode), w1 = new DenseVector(
				totalNode);
		for (int row = 0; row < numUsers; row++) {
			w0.set(row, 1.0 /(3*numUsers) );
			w1.set(row, 1.0 / (3*numUsers));
		}
		for (int row = 0; row < numtag; row++) {
			w0.set(row + numUsers, 1.0 / (3*numtag));
			w1.set(row + numUsers, 1.0 / (3*numtag));
		}
		for (int row = 0; row < numItems; row++) {
			w0.set(row + numUsers + numtag, 1.0 /(3*numItems) );
			w1.set(row + numUsers + numtag, 1.0 / (3*numItems));
		}

		// repeat until converage
		int count = 0;
		DenseVector w0bef = null, w1bef = null;
		do {
			count++;
			w0bef = new DenseVector(w0);
			w1bef = new DenseVector(w1);
			w0 = sparsMultVector(w0bef).scale(lambda).add(p0.scale(1 - lambda));
			w1 = sparsMultVector(w1bef).scale(lambda).add(p1.scale(1 - lambda));
			if (diff(w1, w1bef) < minerror) {
				break;
			}
		} while (count < maxiter);

		DenseVector w = w1.minus(w0);

		List<String> rankedItems = new ArrayList<>();
		Map<Integer, Double> itemScores = new HashMap<>();
		for (int index = 0; index < numItems; index++) {
			String itemStr = rateDao.getItemId(index);
			if (candItems.contains(itemStr)) {
				itemScores.put(index, w.get(index + numUsers + numtag));
			}
		}
		if (itemScores.size() > 0) {
			List<KeyValPair<Integer>> sorted = Lists.sortMap(itemScores, true);
			List<KeyValPair<Integer>> recomd = (numRecs < 0 || sorted.size() <= numRecs) ? sorted
					: sorted.subList(0, numRecs);

			for (KeyValPair<Integer> kv : recomd)
				rankedItems.add(rateDao.getItemId(kv.getKey()));
		}
		//System.out.println("finish getTopkItem"+u+" "+tag+" iter:"+count+" diff："+diff(w1, w1bef));
		return rankedItems;
		
	}
	
	private DenseVector sparsMultVector(DenseVector vector){
		DenseVector res = new DenseVector(vector.getSize());
		for (int row = 0; row < vector.getSize(); row++) {
			double val = 0;
			Set<Integer> nozero=transfer.get(row).keySet();
			for (Integer index : nozero) {
				val += transfer.get(row).get(index) * vector.get(index);
				assert transfer.get(row).get(index)==transfer.get(index).get(row);
			}
			res.set(row, val);
		}

		return res;
	}
	private double diff(DenseVector pre, DenseVector compare) {
		double result = 0;
		for (int i = 0; i < pre.getSize(); i++) {
			result += Math.pow(pre.get(i) - compare.get(i), 2);
		}
		result = Math.sqrt(result);
		return result;
	}
}
