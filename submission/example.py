from Model import Model

# run fit
m = Model(get_train_set=None, systematics=None)
results = m.predict(test_toy)
print(results)
