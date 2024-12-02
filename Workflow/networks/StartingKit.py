from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class Model:

    def __init__(self, get_train_set=None, systematics=None):

        self.get_train_set = get_train_set
        self.systematics = systematics
        self.re_train = True
                
        self.model = XGBClassifier(eval_metric=["error", "logloss", "rmse"],)
        self.name = "model_XGB"
        self.scaler = StandardScaler()
        

    def fit(self):
        """
        Trains the model.

        Params:
            None

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """


        train_set = self.get_train_set() # train_set is a dictionary with data, labels, and weights
        
        training_set, holdout_set = train_test_split(
            train_set, test_size=0.5, random_state=42, reweight=True
        )
        
        del train_set
        
        training_set = self.systematics(training_set)

        weights_train = training_set["weights"].copy()
        train_labels = training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        training_set["weights"] = weights_train
                
        self.scaler.fit_transform(training_set["data"])

        X_train_data = self.scaler.transform(training_set["data"])
        self.model.fit(X_train_data,training_set["labels"], training_set["weights"])
        
        holdout_set = self.systematics(holdout_set)
        
        self.saved_info = calculate_saved_info(self.model, holdout_set)

        holdout_score = self.model.predict(holdout_set["data"])
        holdout_results = compute_mu(
            holdout_score, holdout_set["weights"], self.saved_info
        )
            
        print("Holdout Results: ")
        for key in holdout_results.keys():
            print("\t", key, " : ", holdout_results[key])


    def predict(self, test_set):
        """
        Predicts the values for the test set.

        Parameters:
            test_set (dict): A dictionary containing the test data, and weights.

        Returns:
            dict: A dictionary with the following keys:
            * 'mu_hat': The predicted value of mu.
            * 'delta_mu_hat': The uncertainty in the predicted value of mu.
            * 'p16': The lower bound of the 16th percentile of mu.
            * 'p84': The upper bound of the 84th percentile of mu.
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        test_data = self.scaler.transform(test_data)
        predictions = self.model.predict_proba(test_data)[:, 1]
    
        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
