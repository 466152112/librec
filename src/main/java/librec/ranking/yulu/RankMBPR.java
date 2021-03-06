package librec.ranking.yulu;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import coding.io.KeyValPair;
import coding.io.Lists;
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
 * </ul>RankIBPR learning ratio = 0.0005, reg = 0.35
 * </p>
 * 
 * 
 */
public class RankMBPR extends IterativeRecommender {
	
	private int sFixedSteps = 200;
	
	public RankMBPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
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
			int u = 0;
			
			for (; u < numUsers; u++){
				
				SparseVector pu = trainMatrix.row(u);

				if (pu.getCount() == 0)
					continue;
				
				int[] is = pu.getIndex();
				
				for(int i : is){

//					SparseVector pi=trainMatrix.column(i);
					// 运行 RankBPR 算法获取预估排序值
					List<Integer> rankTuple = itemStepEstimate(u, i);
					int steps = rankTuple.get(0);
					double rLoss = rankLoss(steps);
					int j = rankTuple.get(1);
					
					// 运行 RankBPR 算法获取预估排序值
					rankTuple = userStepEstimate(u, i);
					steps = rankTuple.get(0);
					double usrLoss = rankLoss(steps);
					int v = rankTuple.get(1);
					
//					while (true) {
//						
//						do {
//							v = Randoms.uniform(numUsers);
//						} while (pi.contains(v));
//
//						do {
//							j = Randoms.uniform(numItems);
//						} while (pu.contains(j));
//
//						break;
//					}
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

					double cmguij = g(-xuij) * rLoss;
					double cmgiuv = g(-xiuv) * usrLoss;
					
					for (int f = 0; f < numFactors; f++) {
						double puf = P.get(u, f);
						double pvf = P.get(v, f);
						double qif = Q.get(i, f);
						double qjf = Q.get(j, f);

						P.add(u, f, lRate * (cmguij * (qif - qjf) + cmgiuv*qif - regU * puf));
						P.add(v, f, lRate * (cmgiuv*(-qif) - regU * pvf));
						
						Q.add(i, f, lRate * (0.5 * cmguij * puf + 0.5 * cmgiuv *(puf-pvf)- regI * qif));
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
	
	/*
	在给定目标用户 uid 的情况下，该函数用于预估物品i的排序
	如果商品池中的商品数量非常庞大的话，每次迭代计算所有商品排序会消耗巨大资源
	严重影响算法模型的训练。因此，我们通过采样的方式估计商品的大致排序，根据 i
	的预估排序定义一个与排序相关的损失函数，用于动态调整 feature vectors 更新
	速率。
	算法步骤:
	1. 进行一定数量的采样次数预估商品正样本排名
	2. 根据预估排名计算与排序相关的损失值，动态调整模型的更新速率，排名估计值越大
	   更新幅度越大，反之成立
	*/
	public List<Integer> itemStepEstimate(int uid, int iid){
		
		// 获取用户 u 点击序列
		SparseVector pu = trainMatrix.row(uid);
	
		// 计算 u 对 i 的偏好值
		double xui = predict(uid, iid);
	
		// 固定采样次数
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// 存放返回结果
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int j=0;
		double xuji;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				j = Randoms.uniform(numItems);
			} while (pu.contains(j));
	
			double xuj = predict(uid, j);
	
			xuji = 1.0 + xuj - xui;
			sampleBuffer.put(j, xuji);
	
			if (xuji > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
	
			// 这里计算公式为 er = (商品总数 - 1) / (sFixedSteps - _sFixedSteps)
			//rank = (int) math.floor( er ) 对 er 进行向下取整
			// 如 er = 3.4，floor(er) = 3.0
			 rank=(numItems-1)/(sFixedSteps -_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	public List<Integer> userStepEstimate(int uid, int iid){
		
		// 获取商品 i 点击序列
		SparseVector pi = trainMatrix.column(iid);
	
		// 计算 u 对 i 的偏好值
		double xui = predict(uid, iid);
	
		// 固定采样次数
		int _sFixedSteps = sFixedSteps;
		boolean susFlag = true;
	
		// 存放返回结果
		List<Integer> resultsList = new ArrayList<Integer>();
	
		// 初始化样本缓存，用于存储中间采样时获取的商品序列
		Map<Integer, Double> sampleBuffer = new HashMap<Integer, Double>();
		int v=0;
		double xiuv;
		// 进行采样获取偏好值大于 i 的商品 j 并记录下采样次数
		while ( _sFixedSteps > 0 && susFlag){
			do {
				v = Randoms.uniform(numUsers);
			} while (pi.contains(v));
	
			double xiv = predict(v, iid);
	
			xiuv = 1.0 + xiv - xui;
			sampleBuffer.put(v, xiuv);
	
			if (xiuv > 0) susFlag = false;
	
			_sFixedSteps -= 1;
		}
		
		// 根据采样序列中每个样本得分从高到低对sampleBuffer进行排序，获取第一个元素
		// 这里需要你来写，排序部分我不是很懂
		// sampleBuffer.sort(key = x[1])，x[1] 为sampleBuffer 元素的第二个值
		// int selectedItemID = sampleBuffer[0][0];
		List<KeyValPair<Integer>> sorted = Lists.sortMap(sampleBuffer,
				true);
		int selectedItemID = sorted.get(0).getKey();
		// 获取预估采样步数
		int rank = 0;
	
		// 开始预估正样本 i 的当前排名
		if (sFixedSteps <= 0) rank = 1;
		else{
	
			// 这里计算公式为 er = (商品总数 - 1) / (sFixedSteps - _sFixedSteps)
			//rank = (int) math.floor( er ) 对 er 进行向下取整
			// 如 er = 3.4，floor(er) = 3.0
			 rank=(numItems-1)/(sFixedSteps-_sFixedSteps);
		}
	
		resultsList.add(rank);
		resultsList.add(selectedItemID);
	
		return resultsList; 
	}
	
	// 根据预估商品排序值计算采样收益
	// loss = SUM_i 1.0 / rank_i
	public double rankLoss(int steps){
		double loss = 0.0;
		for(int i = 1; i < (steps + 1); i++){
			loss += 1.0 / i;
		}
		return loss;
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
