package librec.ranking.temporal;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import coding.io.FileIO;
import coding.io.KeyValPair;
import coding.io.Lists;
import coding.io.Logs;
import coding.io.Strings;
import coding.math.Measures;
import coding.math.Stats;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import librec.data.MatrixEntry;
import librec.data.RatingContext;
import librec.data.SparseMatrix;
import librec.intf.ContextRecommender;

/**
 * Generic recommenders where contextual information is used. The context can be user-, item- and rating-related.
 * 
 * @author guoguibing
 * 
 */
public class TemporalRecommender extends ContextRecommender {
	// the span of days of rating timestamps
	protected static int numDays;

	// minimum/maximum rating timestamp
	protected static long minTimestamp, maxTimestamp;
	
	protected Table<Integer, Integer, List<Integer>> testContext;
	protected Table<Integer, Integer, List<Integer>> trainContext;
	// number of bins over all the items
	protected int numBins;
	
	
	// time unit may depend on data sets, e.g. in MovieLens, it is unix seconds
	protected final static TimeUnit secs = TimeUnit.MILLISECONDS;
	
	// initialization
	static {

		// read context information here
	}

	public TemporalRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}
	
	@Override
	protected Map<Measure, Double> evalRankings() {
		
		if (testContext==null) {
			testContext= HashBasedTable.create();
			// for each test user
			for (MatrixEntry me: testMatrix) {
				int u=me.row();
				int item=me.column();
				RatingContext rc=ratingContexts.get(u, item);
				int t=bin(days(rc.getTimestamp(),minTimestamp));
				if (testContext.contains(u, t)) {
					testContext.get(u, t).add(item);
				}else {
					List<Integer> items=new ArrayList<Integer>();
					items.add(item);
					testContext.put(u, t, items);
				}
				
			}
			
			trainContext= HashBasedTable.create();
			// for each test user
			for (MatrixEntry me: trainMatrix) {
				int u=me.row();
				int item=me.column();
				RatingContext rc=ratingContexts.get(u, item);
				int t=bin(days(rc.getTimestamp(),minTimestamp));
				if (testContext.contains(u, t)) {
					testContext.get(u, t).add(item);
				}else {
					List<Integer> items=new ArrayList<Integer>();
					items.add(item);
					testContext.put(u, t, items);
				}
				
			}
		}
		
		Map<Measure, Double> measures = new HashMap<>();

		List<Double> ds5 = new ArrayList<>();
		List<Double> ds10 = new ArrayList<>();

		List<Double> precs5 = new ArrayList<>();
		List<Double> precs10 = new ArrayList<>();
		List<Double> recalls5 = new ArrayList<>();
		List<Double> recalls10 = new ArrayList<>();
		List<Double> aps = new ArrayList<>();
		List<Double> rrs = new ArrayList<>();
		List<Double> aucs = new ArrayList<>();
		List<Double> ndcgs = new ArrayList<>();

		List<Double> maes = new ArrayList<>();
		List<Double> rmses = new ArrayList<>();

		
		
		// run in parallel
		ExecutorService executor = Executors.newFixedThreadPool(numberOfCoreForMeasure);
		List<Future<Map<Measure, Double>>> results=new ArrayList<>();
		// for each test user
		for (int userid : testContext.rowKeySet()) {
			for (int time : testContext.row(userid).keySet()) {
				// candidate items for all users: here only training items
				List<Integer> candItems = trainMatrix.columns();
				candItems.retainAll(trainContext.get(userid, time));
				results.add(executor.submit(new rankingMeasure(candItems,userid,time)));
			}
		}
		
		executor.shutdown();
		for (Future<Map<Measure, Double>> result : results) {
          try {
        	Map<Measure, Double> resultMap=result.get();
			
			if (resultMap!=null) {
			//	System.out.println(resultMap.keySet().size());
				maes.add(resultMap.get(Measure.MAE));
				rmses.add(resultMap.get(Measure.RMSE));
				if (isDiverseUsed) {
					ds5.add(resultMap.get(Measure.D5));
					ds10.add(resultMap.get(Measure.D10));
				}
				precs5.add(resultMap.get(Measure.Pre5));
				precs10.add(resultMap.get(Measure.Pre10));
				recalls5.add(resultMap.get(Measure.Rec5));
				recalls10.add(resultMap.get(Measure.Rec10));
				aucs.add(resultMap.get(Measure.AUC));
				aps.add(resultMap.get(Measure.MAP));
				rrs.add(resultMap.get(Measure.MRR));
				ndcgs.add(resultMap.get(Measure.NDCG));
			}
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
        }  
		measures.put(Measure.D5, isDiverseUsed ? Stats.mean(ds5) : 0.0);
		measures.put(Measure.D10, isDiverseUsed ? Stats.mean(ds10) : 0.0);
		measures.put(Measure.Pre5, Stats.mean(precs5));
		measures.put(Measure.Pre10, Stats.mean(precs10));
		measures.put(Measure.Rec5, Stats.mean(recalls5));
		measures.put(Measure.Rec10, Stats.mean(recalls10));
		measures.put(Measure.AUC, Stats.mean(aucs));
		measures.put(Measure.NDCG, Stats.mean(ndcgs));
		measures.put(Measure.MAP, Stats.mean(aps));
		measures.put(Measure.MRR, Stats.mean(rrs));

		measures.put(Measure.MAE, Stats.mean(maes));
		measures.put(Measure.RMSE, Stats.mean(rmses));

		return measures;
	}

	private class rankingMeasure implements Callable<Map<Measure, Double>> {
		private final List<Integer> candItems;
		private final int u;
		private final int t;
		
		/**
		 * @param candItems
		 * @param u
		 * @param t
		 */
		rankingMeasure(List<Integer> candItems, int u,int t) {
			this.candItems = candItems;
			this.u = u;
			this.t=t;
		}

		@Override
		public Map<Measure, Double> call() {
			// make a copy: candidate items for each user
			List<Integer> pCandItems = new ArrayList<>(candItems);
			Map<Measure, Double> measures = new HashMap<>();
			
			// get positive items from testContext data
			List<Integer> tv=testContext.get(u, t);
			List<Integer> correctItems = new ArrayList<>();

			// get overall MAE and RMSE -- not preferred for ranking
			for (Integer j : tv) {
				// intersect with the candidate items
				if (pCandItems.contains(j))
					correctItems.add(j);

				double pred = predict(u, j, t);
				if (!Double.isNaN(pred)) {
					double rate = 1;
					double euj = rate - pred;
					measures.put(Measure.MAE,Math.abs(euj) );
					measures.put(Measure.RMSE,euj * euj );
				}
			}
			
			if (correctItems.size() == 0)
				return null; // no testing data for user u
			// remove rated items from candidate items
			
			// number of candidate items for this user
			int numCand = pCandItems.size();

			// predict the ranking scores of all candidate items
			Map<Integer, Double> itemScores = ranking(u, pCandItems,t);

			// order the ranking scores from highest to lowest
			List<Integer> rankedItems = new ArrayList<>();
			
			if (itemScores.size() > 0) {

				List<KeyValPair<Integer>> sorted = Lists.sortMap(itemScores,
						true);
				List<KeyValPair<Integer>> recomd = (numRecs < 0 || sorted
						.size() <= numRecs) ? sorted : sorted.subList(0,
						numRecs);

				for (KeyValPair<Integer> kv : recomd)
					rankedItems.add(kv.getKey());
			}

			if (rankedItems.size() == 0)
				return null; // no recommendations available for user u

			int numDropped = numCand - rankedItems.size();
			double AUC = Measures.AUC(rankedItems, correctItems, numDropped);
			double AP = Measures.AP(rankedItems, correctItems);
			double nDCG = Measures.nDCG(rankedItems, correctItems);
			double RR = Measures.RR(rankedItems, correctItems);

			if (isDiverseUsed) {
				double d5 = diverseAt(rankedItems, 5);
				double d10 = diverseAt(rankedItems, 10);
				measures.put(Measure.D5,d5 );
				measures.put(Measure.D10,d10);
			}

			List<Integer> cutoffs = Arrays.asList(5, 10);
			Map<Integer, Double> precs = Measures.PrecAt(rankedItems,
					correctItems, cutoffs);
			Map<Integer, Double> recalls = Measures.RecallAt(rankedItems,
					correctItems, cutoffs);
			
			measures.put(Measure.Pre5,precs.get(5) );
			measures.put(Measure.Pre10,precs.get(10));
			measures.put(Measure.Rec5,recalls.get(5));
			measures.put(Measure.Rec10,recalls.get(10));
			
			measures.put(Measure.AUC,AUC);
			measures.put(Measure.MAP,AP);
			measures.put(Measure.MRR,RR);
			measures.put(Measure.NDCG,nDCG);
			
			return measures;
		}
	}
	
	/**
	 * compute ranking scores for a list of candidate items
	 * 
	 * @param u
	 *            user id
	 * @param candItems
	 *            candidate items
	 * @return a map of {item, ranking scores}
	 */
	protected Map<Integer, Double> ranking(int u, Collection<Integer> candItems,int time) {

		Map<Integer, Double> itemRanks = new HashMap<>();
		for (Integer j : candItems) {
			double rank = ranking(u, j,time);
			if (!Double.isNaN(rank))
				itemRanks.put(j, rank);
		}

		return itemRanks;
	}
	
	/**
	 * predict a ranking score for user u on item j: default case using the
	 * unbounded predicted rating values if is userBasis no null then add
	 * userBasis and ItemBasis
	 * 
	 * @param u
	 *            user id
	 * 
	 * @param j
	 *            item id
	 * @return a ranking score for user u on item j
	 */
	protected double ranking(int u, int j,int time) {
		return predict(u, j, time);
	}
	
	protected double predict(int u, int i,int timeinterval)  {
		return 0;
	}
	/***************************************************************** Functional Methods *******************************************/
	/**
	 * Read rating timestamps
	 */
	protected static void readContext() throws Exception {
		String contextPath = cf.getPath("dataset.social");
		Logs.debug("Context dataset: {}", Strings.last(contextPath, 38));

		ratingContexts = HashBasedTable.create();
		BufferedReader br = FileIO.getReader(contextPath);
		String line = null;
		

		minTimestamp = Long.MAX_VALUE;
		maxTimestamp = Long.MIN_VALUE;
		
		while ((line = br.readLine()) != null) {
			String[] data = line.split("[ \t,]");
			String user = data[0];
			String item = data[1];

			// convert to million seconds
			long timestamp = secs.toMillis(Long.parseLong(data[3]));

			int userId = rateDao.getUserId(user);
			int itemId = rateDao.getItemId(item);
			RatingContext rc = null;
			rc = new RatingContext(userId, itemId);
			rc.setTimestamp(timestamp);
			if (userId==1787&&itemId==3151) {
				System.out.println();	
			}
			ratingContexts.put(userId, itemId, rc);

			if (minTimestamp > timestamp)
				minTimestamp = timestamp;
			if (maxTimestamp < timestamp)
				maxTimestamp = timestamp;
		}

		numDays = days(maxTimestamp, minTimestamp) + 1;
	}
	/**
	 * @return the bin number (starting from 0..numBins-1) for a specific timestamp t;
	 */
	protected int bin(int day) {
		return (int) (day / (numDays + 0.0) * numBins);
	}

	/**
	 * @return number of days for a given time difference
	 */
	protected static int days(long diff) {
		return (int) TimeUnit.MILLISECONDS.toDays(diff);
	}

	/**
	 * @return number of days between two timestamps
	 */
	protected static int days(long t1, long t2) {
		return days(Math.abs(t1 - t2));
	}
	
}
