"""
Caterpillar @ Kaggle

__author__ : Vijay Sathish

- Initial Data Wrangling for some files performed in R and then imported to Python 

"""

import pandas as pd
import numpy as np
import time
import math
import re
import sys
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, LabelEncoder)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


## Global flags
basePath = 'D:/Kaggle/CAT'
dataPath = ('/').join([basePath, 'data'])
resultsPath = ('/').join([basePath, 'results'])
interPath = ('/').join([basePath, 'inter'])

# randomState = 1234
# randomState = 4321
# randomState = 8192
# randomState = 4096
randomState = 512
# randomState = 1638
cvFolds = 5
mergeBomData = False
mergeCompTypeData = False
mergeCompNameData = True
gridSearch = False
kfObject = None								# K-Fold object used in GridSearch
trainTwoModels = False					# Train bracketed and non-bracketed pricing separately using two models
transformSpecData = False			# Indicate whether to perform TFIDF + SVD transform on spec data
transformCompNameData = True	# Indicate whether to perform TFIDF + SVD transform on spec data
specNumSVDComponents = [84]		# Relevant only if we decide to transform the comp_name data
bomNumSVDComponents = [500]
compnameNumSVDComponents = [250]

dumpIntermediateResults = True		# Dump the intermediate train and test sets to be used by xgboost in R


def preprocess_train_test_sets (train, test) :
	" date features addition and drop unwanted variables for train, test datasets "
	train['year'] = train.quote_date.dt.year
	train['month'] = train.quote_date.dt.month
	train['dayofyear'] = train.quote_date.dt.dayofyear
	train['dayofweek'] = train.quote_date.dt.dayofweek
	train['day'] = train.quote_date.dt.day

	test['year'] = test.quote_date.dt.year
	test['month'] = test.quote_date.dt.month
	test['dayofyear'] = test.quote_date.dt.dayofyear
	test['dayofweek'] = test.quote_date.dt.dayofweek
	test['day'] = test.quote_date.dt.day

	## fill NAs in material_id
	train['material_id'].fillna('SP-9999', inplace = True)
	test['material_id'].fillna('SP-9999', inplace = True)

	## drop useless columns and create labels and test ids
	test_ids = np.array(test.id.values.astype(int))
	print ("test_ids.shape ", test_ids.shape)
	labels = np.array(train.cost.values)
	print ("labels.shape ", labels.shape)

	## Transform labels to log(1 + x)
	labels = np.log1p(labels)

	test = test.drop(['id', 'quote_date'], axis = 1)
	train = train.drop(['quote_date', 'cost'], axis = 1)
	print ("train.shape: ", train.shape)
	print ("test.shape: ", test.shape)
	return train, labels, test, test_ids


def label_encode_train_test_sets (train, test) :
	" Label encode 'supplier' and 'bracket_pricing' features for both train and test set "
	test_suppliers = np.sort(pd.unique(test.supplier.ravel()))
	print ("Test suppliers shape & elements: ", test_suppliers.shape, test_suppliers)
	train_suppliers = np.sort(pd.unique(train.supplier.ravel()))
	print ("Train suppliers shape & elements: ", train_suppliers.shape, train_suppliers)
	
	## Merge 'supplier' for both datasets first because we want encoding to be consistent across both
	# http://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
	supplier_ids = []
	supplier_ids.extend(train_suppliers)
	supplier_ids.extend(test_suppliers)
	supplier_ids = np.sort(np.unique(supplier_ids))
	print ("Merged supplier_ids.shape: ", supplier_ids.shape)
	# print ("supplier_ids.elements: ", supplier_ids)

	## Perform label encoding fit on the merged array and then individually transform for train and test sets
	print ("Performing label encoding on supplier column...")
	label_e = LabelEncoder()
	label_e.fit(supplier_ids)
	train['supplier'] = label_e.transform(train['supplier'])
	test['supplier'] = label_e.transform(test['supplier'])

	## Perform label encoding on 'bracket_pricing'
	print ("Performing label encoding on bracket_pricing column...")
	train['bracket_pricing'] = label_e.fit_transform(train['bracket_pricing'])
	test['bracket_pricing'] = label_e.fit_transform(test['bracket_pricing'])

	return train, test


def tfidf_plus_svd_transform (df_X, num_svd_components, input_type = 'spec') :
	" TFIDF transform the features and re-densify sparse features using TruncatedSVD "
	ids = df_X['tube_assembly_id']
	X = np.array(df_X.drop(['tube_assembly_id'], axis = 1), dtype = int)
	#X = np.array(df_X)
	#ids = X[:, 0]								# First column is tube_assembly_id
	#X = X[:, 1:]					# Remaining columns for TfidfTransform
	#X = np.array(X, dtype = int)
	print ("%s.shape: " %(input_type), X.shape)
	print ("%s_ids.shape: " %(input_type), ids.shape)

	tft = TfidfTransformer(sublinear_tf = True)
	print ("Performing Tfidf Transform on %s data... " %(input_type))
	X_sparse = tft.fit_transform(X)

	for num_comp in num_svd_components :
		svd = TruncatedSVD(n_components = num_comp, random_state = randomState)
		print ("Performing Truncated SVD Transform on %s data... " %(input_type))
		X_svd = svd.fit_transform(X_sparse)
		print ("SVD explained variance for %s n_components = %d: %.5f" %(input_type, num_comp, svd.explained_variance_ratio_.sum())) 
		print ("%s_svd.shape: " %(input_type), X_svd.shape)

	return X_svd, ids

def split_data_on_bracket_pricing (train, labels, test, test_ids) :
	" Split training and test data into two separate datasets based on 'bracket_pricing' variable "
	
	## 1. Add back labels into training set and test_ids into test set so they can be split too
	train['cost'] = labels
	test['ids'] = test_ids

	## 2. Perform the split based on 'bracket_pricing'
	train_b = train.loc[train['bracket_pricing'] == 1]
	labels_b =  train_b['cost']
	train_b = train_b.drop(['cost'], axis = 1)
	train_b = train_b.drop(['bracket_pricing'], axis = 1)

	train_nb = train.loc[train['bracket_pricing'] == 0]
	labels_nb =  train_nb['cost']
	train_nb = train_nb.drop(['cost'], axis = 1)
	train_nb = train_nb.drop(['bracket_pricing'], axis = 1)
	
	test_b = test.loc[test['bracket_pricing'] == 1]
	test_ids_b = test_b['ids']
	test_b = test_b.drop(['ids'], axis = 1)
	test_b = test_b.drop(['bracket_pricing'], axis = 1)

	test_nb = test.loc[test['bracket_pricing'] == 0]
	test_ids_nb = test_nb['ids']
	test_nb = test_nb.drop(['ids'], axis = 1)
	test_nb = test_nb.drop(['bracket_pricing'], axis = 1)

	print ("train_b.rows: %d, train_nb.rows: %d, labels_b.len: %d, labels_nb.len: %d,  test_b.rows: %d, test_nb.rows: %d " %(train_b.shape[0], train_nb.shape[0], len(labels_b), len(labels_nb), test_b.shape[0], test_nb.shape[0]))
	return train_b, train_nb, labels_b, labels_nb, test_b, test_nb, test_ids_b, test_ids_nb

def reconstruct_df_from_transformed_arrays (X, ids, num_components, input_type = 'spec') :
	" reconstruct dataframe for spec and bom data post TFIDF+SVD transform " 
	## Enumerate column names as S_xx
	col_names = []
	for i in range(num_components) :
		col_names.extend(["%s_%d" %(input_type, i)])
	# print ("%s feature names: " %(input_type), col_names)
	df = pd.DataFrame (data = X, columns = col_names)

	## Add 'tube_assembly_id' back to the dataframe
	# http://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
	df['tube_assembly_id'] = ids
	print ("%s transformed df.shape: " %(input_type), df.shape)
	return df

def train_and_predict_rfr (train, labels, test) :
	if (gridSearch) :
		#params_dict = {'n_estimators':[100, 300, 500], 'max_depth':[None], 'max_features' : [None, 0.95, 0.9]}
		params_dict = {'n_estimators':[500, 1000], 'max_depth':[None], 'max_features' : [0.8, 0.85, 0.9]}
		model = GridSearchCV(RandomForestRegressor(random_state = randomState, n_jobs = -1), params_dict, n_jobs = 1, cv = kfObject, verbose = 10, scoring = 'mean_squared_error')
	else :
		model = RandomForestRegressor(random_state = randomState, n_jobs = 4, n_estimators = 400, max_depth = None, verbose = 10, oob_score = True, max_features = 0.95)
	
	model.fit(train, labels)

	if (gridSearch) :
		print ("Best estimator: ", model.best_estimator_)
		print ("Best grid scores: %.4f" %(model.best_score_))
	else :
		print ("OOB score: %.5f " %(model.oob_score_))
	return model.predict(test)

def train_and_predict_gbr (train, labels, test, feature_names = None) :
	### Train estimator (initially only on final count
	if (gridSearch) :
		params_dict = {'n_estimators':[2000, 3000], 'max_depth':[6, 8], 'subsample': [1, 0.95], 'max_features' : [None], 'learning_rate' : [0.07]} 
		# params_dict = {'n_estimators':[3000], 'max_depth':[8], 'subsample': [1], 'max_features' : [None], 'learning_rate' : [0.05]} 
		#params_dict = {'n_estimators':[1000], 'max_depth':[7], 'learning_rate': [0.1], 'subsample': [0.8], 'max_features': [120]}
		model = GridSearchCV(GradientBoostingRegressor(random_state = randomState, learning_rate = 0.1, verbose = 0), params_dict, n_jobs = 3, cv = kfObject, verbose = 10, scoring = 'mean_squared_error')
	else :
		#model = GradientBoostingRegressor(random_state = randomState, learning_rate = 0.05, n_estimators = 3000, max_depth = 5, subsample = 1, max_features = None, verbose = 10)
		#model = GradientBoostingRegressor(random_state = randomState, learning_rate = 0.07, n_estimators = 2000, max_depth = 8, subsample = 0.95, max_features = None, verbose = 10)
		#model = GradientBoostingRegressor(random_state = randomState, learning_rate = 0.05, n_estimators = 3500, max_depth = 5, subsample = 0.95, max_features = None, verbose = 10)
		model = GradientBoostingRegressor(random_state = randomState, learning_rate = 0.07, n_estimators = 4500, max_depth = 5, subsample = 0.95, max_features = 0.9, verbose = 10)

	model.fit(train, labels)

	if (gridSearch) :
		print ("Best estimator: ", model.best_estimator_)
		print ("Best MSLE scores: %.4f" %(model.best_score_))
		print ("Best RMSLE score: %.4f" %(math.sqrt(-model.best_score_)))
	else :
		float_formatter = lambda x: "%.4f" %(x)
		print ("Feature importances: ", sorted(zip([float_formatter(x) for x in model.feature_importances_], feature_names), reverse=True))
	return model.predict(test)


def train_and_predict_adab_stacked_gbr (train, labels, test, feature_names = None) :
	" Attmept with SVR ... "
	print ("Training ADABoost with GBR as base model")
	t0 = time.clock()
	if (gridSearch) :
		params_dict = {'adab__learning_rate' : [0.1, 0.3]} 
		#model = GridSearchCV(regr, params_dict, n_jobs = 3, cv = kfObject, verbose = 10, scoring = 'mean_squared_error')
	else :
		base =  GradientBoostingRegressor(random_state = randomState, learning_rate = 0.1, n_estimators = 1500, max_depth = 6, subsample = 0.95, max_features = 1, verbose = 10)
		model = AdaBoostRegressor(random_state = randomState, base_estimator = base, n_estimators = 3, learning_rate = 0.005)

	model.fit(train, labels)
	print ("Model fit completed in %.3f sec " %(time.clock() - t0))

	if (gridSearch) :
		print ("Best estimator: ", model.best_estimator_)
		print ("Best MSLE scores: %.4f" %(model.best_score_))
		print ("Best RMSLE score: %.4f" %(math.sqrt(-model.best_score_)))
	else :
		float_formatter = lambda x: "%.4f" %(x)
		print ("Feature importances: ", sorted(zip([float_formatter(x) for x in model.feature_importances_], feature_names), reverse=True))
	
	return model.predict(test)


def train_and_predict_SVR (train, labels, test, feature_names = None) :
	" Attmept with SVR ... "
	print ("Training SVR model...")
	t0 = time.clock()
	if (gridSearch) :
		#params_dict = {'krr_alpha' : [0.05, 0.1, 0.3, 1]} 
		params_dict = {'svr__C' : [10], 'svr__gamma' : [0.1, 0.33, 1]} 
		regr = Pipeline([('scl', StandardScaler()), 
						#('krr', KernelRidge(gamma = 'auto', kernel = 'rbf', verbose = 10))])
						('svr', SVR(kernel = 'rbf', C = 10, cache_size = 512, verbose = 0))])
		model = GridSearchCV(regr, params_dict, n_jobs = 3, cv = kfObject, verbose = 10, scoring = 'mean_squared_error')
	else :
		#regr = KernelRidge(alpha = 0.05, gamma = 'auto', kernel = 'rbf')
		regr = SVR(kernel = 'rbf', C = 10, cache_size = 1024, verbose = 10)
		model = Pipeline([('scl', StandardScaler()), ('krr', regr)])

	model.fit(train, labels)
	print ("Model fit completed in %.3f sec " %(time.clock() - t0))

	if (gridSearch) :
		print ("Best estimator: ", model.best_estimator_)
		print ("Best MSLE scores: %.4f" %(model.best_score_))
		print ("Best RMSLE score: %.4f" %(math.sqrt(-model.best_score_)))
	
	return model.predict(test)

## Dump intermediate results after merging so that it can be picked up by XGBoost model in R
def dump_intermediate_results (train, labels, test, test_ids) :
	## Initialize and set tags used to create dump path
	spec_prefix = ''
	compname_prefix = ''
	comptype_prefix = ''
	bom_prefix = ''

	spec_prefix = 'spec' + str(specNumSVDComponents[0])
	if (transformSpecData == False) : 
		spec_prefix = 'nontrans_' + spec_prefix
	if mergeCompNameData :
		if (transformCompNameData == False) : 
			compname_prefix = '_nontrans_compname'
		else :
			compname_prefix = '_compname' + str(compnameNumSVDComponents[0])
	if mergeBomData :
		bom_prefix = '_bom' + str(bomNumSVDComponents[0]) 
	if mergeCompTypeData :
		comptype_prefix = '_comptype'
	
	dump_path = ('/').join([interPath, spec_prefix])
	dump_path = dump_path + bom_prefix + compname_prefix + comptype_prefix
	dump_path_train = dump_path + '_train.csv'
	dump_path_test = dump_path + '_test.csv'
	print ('Intermediate data dump_path train: ', dump_path_train)
	print ('Intermediate data dump_path test: ', dump_path_test)
	dump_train = train
	dump_train['cost'] = labels
	dump_test = test
	dump_test['ids'] = test_ids
	dump_train.to_csv(dump_path_train, index = False)
	dump_test.to_csv(dump_path_test, index = False)

	return

if __name__ == '__main__' :
	## Print all the paths
	print ("dataPath: ", dataPath)
	print ("resultsPath: ", resultsPath)
	print ("interPath: ", interPath)

	## Load training and test datasets
	print ("Reading train, test datasets...")
	train = pd.read_csv(('/').join([dataPath, 'train_set.csv']), parse_dates=[2,]).fillna("")
	test = pd.read_csv(('/').join([dataPath, 'test_set.csv']), parse_dates=[3,]).fillna("")

	## Load other helper tables
	#df_tubes = pd.read_csv(('/').join([dataPath, 'transformed', 'tubes_transformed.csv'])).fillna(-1)
	df_tubes = pd.read_csv(('/').join([dataPath, 'transformed', 'tubes_transformed_v2.csv'])).fillna(-1)
	df_bom = pd.read_csv(('/').join([dataPath, 'transformed', 'bom_wide.csv'])).fillna(-1)
	df_specs = pd.read_csv(('/').join([dataPath, 'transformed', 'spec_wide.csv'])).fillna(-1)
	df_comptype = pd.read_csv(('/').join([dataPath, 'transformed', 'comp_type_wide.csv'])).fillna(-1)
	df_compname = pd.read_csv(('/').join([dataPath, 'transformed', 'comp_name_wide.csv'])).fillna(-1)

	## Merge train and test sets with the tubes data
	train = pd.merge(train, df_tubes, on ='tube_assembly_id', how = 'inner')
	test = pd.merge(test, df_tubes, on = 'tube_assembly_id', how = 'inner')

	## Label encode 'supplier' ids and 'bracket_pricing'
	train, test = label_encode_train_test_sets (train, test)

	## Pre-process the data
	print ("Preprocessing train, test datasets...")
	train, labels, test, test_ids = preprocess_train_test_sets (train, test)

	## Whether to transform using TFIDF + SVD or use as is
	if (transformSpecData) :
		## Transform Spec data
		spec_X, spec_ids = tfidf_plus_svd_transform (df_specs, specNumSVDComponents, 'spec')
		## Reconstruct spec datafrane
		spec = reconstruct_df_from_transformed_arrays (spec_X, spec_ids, specNumSVDComponents[0], 'spec')
	else :
		spec = df_specs	

	## Merge spec dataset
	train = pd.merge(train, spec, on ='tube_assembly_id', how = 'inner')
	test = pd.merge(test, spec, on = 'tube_assembly_id', how = 'inner')

	## Transform Bill of Materials data
	if (mergeBomData) :
		bom_X, bom_ids = tfidf_plus_svd_transform (df_bom, bomNumSVDComponents, 'bom')
		## Reconstruct bom datafrane
		print ("WARNING!!! Merging feature heavy BOM data!")
		bom = reconstruct_df_from_transformed_arrays (bom_X, bom_ids, bomNumSVDComponents[0], 'bom')
		## Merge bom dataset
		train = pd.merge(train, bom, on ='tube_assembly_id', how = 'inner')
		test = pd.merge(test, bom, on = 'tube_assembly_id', how = 'inner')
	else :
		print ("NOT MERGING BOM DATA!")

	## Transform Bill of Materials data
	if (mergeCompNameData) :
		if (transformCompNameData) :
			print ("WARNING!!! Merging feature heavy comp_name data!")
			comp_name_X, comp_name_ids = tfidf_plus_svd_transform (df_compname, compnameNumSVDComponents, 'comp_name')
			## Reconstruct comp_name datafrane
			comp_name = reconstruct_df_from_transformed_arrays (comp_name_X, comp_name_ids, compnameNumSVDComponents[0], 'comp_name')
		else :
			print ("WARNING!!! Merging feature heavy comp_name data without TFIDF+SVD transform!")
			comp_name = df_compname
		## Merge comp_name dataset
		train = pd.merge(train, comp_name, on ='tube_assembly_id', how = 'inner')
		test = pd.merge(test, comp_name, on = 'tube_assembly_id', how = 'inner')
	else :
		print ("NOT MERGING comp_name data!")

	## Merge all tables together in preparation for training and prediction
	if (mergeCompTypeData) :
		print ("Merging comp_type data")
		# Merge comp_type dataframe
		train = pd.merge(train, df_comptype, on ='tube_assembly_id', how = 'inner')
		test = pd.merge(test, df_comptype, on = 'tube_assembly_id', how = 'inner')
	else :
		print ("NOT MERGING comp_type data!")

	
	## Print first few rows to make sure they are in 'tube_assembly_id' order
	"""
	tube_ass_ids = test.tube_assembly_id.values
	print ("First few train tube_assembly_ids after merging: ", tube_ass_ids[0:20])
	"""
	## Dump intermediate results for xgboost
	# Note: Not planning to do this for the 2 model results
	if (dumpIntermediateResults) :
		print ("Dumping intermediate results and exiting simulation...")
		dump_intermediate_results (train, labels, test, test_ids) 
		sys.exit()

	## Drop 'tube_assembly_id' from train, test now that all joins are complete
	test = test.drop(['tube_assembly_id'], axis = 1)
	train = train.drop(['tube_assembly_id'], axis = 1)
	print ("Final(post joins) train.shape: ", train.shape)
	print ("Final(post_joins) test.shape: ", test.shape)

	## Get list of feature names from dataframe
	feature_names = list(train)		

	
	## Split the two models based on 'bracket_pricing'
	if (trainTwoModels) :
		train_b, train_nb, labels_b, labels_nb, test_b, test_nb, test_ids_b, test_ids_nb = split_data_on_bracket_pricing (train, labels, test, test_ids)

	## Convert dataframes to numpy arrays for predictor
	if (trainTwoModels) :
		train_b = np.array(train_b, dtype = np.float)
		test_b = np.array(test_b, dtype = np.float)
		train_nb = np.array(train_nb, dtype = np.float)
		test_nb = np.array(test_nb, dtype = np.float)
	else :
		train = np.array(train, dtype = np.float)
		test = np.array(test, dtype = np.float)

	## Setup KFold cross-validation
	## Train and predict
	"""
	Note: 
		- We want to setup train-test folds based on 'tube_assembly_id' without shuffle, otherwise, we grossly over-estimate MSE. 
		- Since tube_assembly_id in training set is grouped and in order, pure KFold without shuffling should be close enough to achieving the split we want
	"""

	if (trainTwoModels) :
		kfObject = KFold(n = train.shape[0], n_folds = cvFolds, shuffle = False)
		print ("Train and predict bracket_pricing model...")
		preds_b = train_and_predict_gbr (train_b, labels_b, test_b, feature_names)
		print ("Train and predict non-bracket_pricing model...")
		preds_nb = train_and_predict_gbr (train_nb, labels_nb, test_nb, feature_names)

		## Combine predictions of both models
		preds = []
		preds.extend(preds_b)
		preds.extend(preds_nb)
		test_ids = []
		test_ids.extend(test_ids_b)
		test_ids.extend(test_ids_nb)
		print ("preds.shape: %d, ids.shape: %d" %(len(preds), len(test_ids)))

	else :
		kfObject = KFold(n = train.shape[0], n_folds = cvFolds, shuffle = False)
		preds = train_and_predict_gbr (train, labels, test, feature_names)
		#preds = train_and_predict_rfr (train, labels, test)
		#preds = train_and_predict_SVR (train, labels, test)
		#preds = train_and_predict_adab_stacked_gbr (train, labels, test, feature_names)

	# Convert predictions and dump them
	preds = np.expm1(preds)
	df_preds = pd.DataFrame({"id": test_ids, "cost": preds})
	## Sort by id so that it is easy to compare with other results
	# http://stackoverflow.com/questions/17618981/how-to-sort-pandas-data-frame-using-values-from-several-columns
	df_preds = df_preds.sort(['id'], ascending = True)

	df_preds.to_csv(('/').join([resultsPath, 'tuned_nontrans_spec84_bom500_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'nontrans_spec84_compname150_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'nontrans_spec84_nontrans_compname_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'tuned_spec84_bom200_comptype_gbr_cv5_v3.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'tuned_nontrans_spec84_compname200_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'tuned_2m_spec84_bom75_compname100_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'tuned_2m_nontrans_spec84_bom100_compname200_gbr_cv5_v2.csv']), index=False)
	#df_preds.to_csv(('/').join([resultsPath, 'tuned_nontrans_spec84_comptype_gbr_cv5_v1.csv']), index=False)


