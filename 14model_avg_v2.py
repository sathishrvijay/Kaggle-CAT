"""
	__Author__: Vijay Sathish
	__Comp__	: Kaggle CAT
	__Date__	: 07/11/2015
	- Key idea: Takes ensembling concept and achieves ultra-performance by diversifying the features on which each model predicts
		- Some models predict the bracketed and non-bracketed separately
		- Some models predict on comp_type, while others predict on comp_name and comp_id
		- Some models predict on raw spec while others predict on TFIDF+SVD spec
		- Some models predict on different combinations of the same sets
		- To add even more randomness, we also change the random seed, max_depth, learning_rate and n_estimators of each model in the mix

"""

import pandas as pd
import numpy as np

"""
 -  Replace each of the previous 9 models with tuned models with new tubes_v2 to basically kick some serious ass and get into 0.22 realm 
 - Goal is to get all tuned models below 0.24 and 2m models below 0.245
 - Note, weak models most likely because they are too deep. Try depth 6 or 7 to improve scores
 - As of 07/13, we can improve about 6 models, so we have some more upside left hopefully
 - 07/14 - 10model_tuned_avg_gbr_07_14_15.csv -> Subscore: 0.224811
 - 07/28 - 14model_tuned_avg_gbr_xgb_07_28_15.csv -> Subscore: 0.224153
"""

weighted_avg = True


# LB score: 0.235269; AUG
m1 = pd.read_csv('D:/Kaggle/CAT/results/tuned_nontrans_spec84_nontrans_compname_gbr_cv5_v1.csv')
# LB score: 0.236367; AUG (from 0.240142)
m2 = pd.read_csv('D:/Kaggle/CAT/results/tuned_nontrans_spec84_comptype_gbr_cv5_v1.csv')
# v2 *** LB score: 0.241433; AUG (from closest 0.239738) (v2 was worse)
# LB score: 0.238726; AUG (from closest 0.239738)
m3 = pd.read_csv('D:/Kaggle/CAT/results/tuned_nontrans_spec84_compname200_gbr_cv5_v1.csv')
# LB score: 0.238773 AUG (from 0.244342)
m4 = pd.read_csv('D:/Kaggle/CAT/results/tuned_spec84_comptype_gbr_cv5_v2.csv')
# LB score: 0.238830; AUG <- (from 0.242881)
m5 = pd.read_csv('D:/Kaggle/CAT/results/tuned_spec84_compname200_gbr_cv5_v3.csv')
# LB score: 0.242444; AUG (from 0.248916)
m6 = pd.read_csv('D:/Kaggle/CAT/results/tuned_spec84_bom200_comptype_gbr_cv5_v3.csv')

# LB score:  0.242211; AUG (from 0.246247)
m7 = pd.read_csv('D:/Kaggle/CAT/results/tuned_2m_nontrans_spec84_bom100_compname200_gbr_cv5_v2.csv')
# LB score: 0.243735; AUG (from 0.248916)
m8 = pd.read_csv('D:/Kaggle/CAT/results/tuned_2m_spec84_comptype_gbr_cv5_v1.csv')
# LB score: 0.244680; AUG (from 0.247012)
m9 = pd.read_csv('D:/Kaggle/CAT/results/tuned_2m_spec84_bom75_compname100_gbr_cv5_v2.csv')
# LB score: 0.247694 
m10 =	pd.read_csv('D:/Kaggle/CAT/results/tuned_nontrans_spec84_bom500_gbr_cv5_v2.csv')

### XGBoost models
# LB score: 0.237647
mx1 = pd.read_csv('D:/Kaggle/CAT/results/xgb_nontrans_spec84_comptype_v1.csv')
# LB score: 0.237025
mx2 = pd.read_csv('D:/Kaggle/CAT/results/xgb_spec84_nontrans_compname_v3.csv')
# LB score:  0.237636
mx3 = pd.read_csv('D:/Kaggle/CAT/results/xgb_nontrans_spec84_compname250_v1.csv')
# LB score: 0.242479
mx4 = pd.read_csv('D:/Kaggle/CAT/results/xgb_nontrans_spec84_bom200_comptype_v2.csv')

if (weighted_avg) :
	scores = [0.235269, 0.236367, 0.238726, 0.238773, 0.238830, 0.242444, 0.242211, 0.243735, 0.244680, 0.247694, 0.237647, 0.237025, 0.237636, 0.242479]
	numerator = np.ones(len(scores))
	weights = np.divide(numerator.astype(float), np.square(np.square(scores)))
	print ("Weights: ", weights)
	preds = np.sum(np.column_stack([m1['cost']*weights[0], m2['cost']*weights[1], m3['cost']*weights[2], 
					m4['cost']*weights[3], m5['cost']*weights[4], m6['cost']*weights[5], 
					m7['cost']*weights[6], m8['cost']*weights[7], m9['cost']*weights[8], m10['cost']*weights[9], 
					mx1['cost']*weights[10], mx2['cost']*weights[11], mx3['cost']*weights[12], mx4['cost']*weights[13]]), axis = 1)
	preds = preds / np.sum(weights)
else :
	preds = np.mean(np.column_stack([m1['cost'], m2['cost'], m3['cost'], m4['cost'], m5['cost'], m6['cost'], 
					m7['cost'], m8['cost'], m9['cost'], m10['cost'], 
					mx1['cost'], mx2['cost'], mx3['cost'], mx4['cost']]), axis = 1)
ids = m1['id']
df = pd.DataFrame({'id': ids, 'cost': preds})
df.to_csv("D:/Kaggle/CAT/results/14model_tuned_invQuadWeighted_gbr_xgb_07_28_15.csv", index = False)

