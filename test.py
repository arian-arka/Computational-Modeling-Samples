from AllModels import AllModels

obj = AllModels()
obj.reportData()
obj.randomForest()
obj.report()
obj.plot('confusion matrix','')
obj.plot('roc','')
obj.save('')
# AllModels().reportData().randomForest(0.2).report().plot('confusion matrix',True).plot('roc').save('')
# AllModels().reportData().knn(2).report().report().plot('confusion matrix').plot('roc')
# AllModels().reportData().linearSVM().report().report().plot('confusion matrix').plot('roc')
# AllModels().reportData().nonLinearSVM().report().report().plot('confusion matrix').plot('roc')