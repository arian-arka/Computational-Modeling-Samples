from ast import Not
from html2text import re
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from urllib3 import Retry
import getopt, sys,os

class AllModels:

    def __init__(self,dataset = None) -> None:
        self.dataset = datasets.load_digits() if dataset is None else dataset
        self.name('No Name Model')
    
    def dataset(self,dataset = None):
        if dataset is None:
            return self._dataset
        else:
            self._dataset = dataset
        return self

    def data(self):
        return self.dataset['data']

    def target(self):
        return self.dataset['target']

    def trained(self, name = 'X'):
        return self.X_train if name in ['X','x'] else self.y_train

    def tested(self, name = 'X'):
        return self.X_test if name in ['X','x'] else self.y_test

    def y_pred(self,y_pred = None):
        if y_pred is None:
            return self._y_pred
        else:
            self._y_pred = y_pred
        return self

    def model(self,model = None):
        if model is None:
            return self._model
        else:
            self._model = model
        return self

    def clf(self,clf = None):
      
        if clf is None:
            return self._clf
        else:
            self._clf = clf
        return self

    def train(self,random_state = None,test_size=0.2,report = True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data(), self.target(), random_state=random_state,test_size=test_size)
        if report:
            print('Report train : ')
            print('Train :',self.X_train.shape, self.y_train.shape)
            print('Test :',self.X_test.shape, self.y_test.shape)
            print('****************************************')
        return self

    def getScore(self):
        return metrics.accuracy_score(self.tested('y'), self.y_pred())
    
    def confusionMatrix(self):
        return metrics.confusion_matrix(self.tested('y'),self.y_pred())

    def classificationReport(self):
        return metrics.classification_report(self.tested('y'),self.y_pred())

    def predict(self):
        return self.model().predict_proba(self.tested('X'))

    def report(self):
        print('Report : ')
        print('----')
        print("Accuracy score after training on existing dataset", self.getScore())
        print('----')
        print('Classification Report : ')
        print(self.classificationReport())
        print('****************************************')
        return self

    def reportData(self):
        print('Report Dataset : ')
        print('--')
        print('Shape : ',self.data().shape)
        print('****************************************')

        return self
    
    def reportALl(self):
        self.report()
        self.reportData()
        return self

    def plotConfusionmatrix(self,save = None):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(self.tested('y'),self.y_pred())
        disp.figure_.suptitle("Confusion Matrix for "+self.name())
        if save != None:
            plt.savefig(save+'confusion_matrix.png', dpi=300, bbox_inches='tight')
      
        plt.show()
        return self
    
    def plotRoc(self,save = None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_score = self.predict()
        y_test_bin = label_binarize(self.tested('y'), classes=[x for x in range(10)])
        n_classes = y_test_bin.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr[i], tpr[i], lw=2)
            print('AUC for Class {}: {}'.format(i+1, metrics.auc(fpr[i], tpr[i])))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc for '+self.name())
        if save != None:
            plt.savefig(save+'roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        return self

    def plotSamples(self, save = None):
        # y_train_df['class'].value_counts().plot(kind = 'bar', colormap = 'Paired')
        # plt.xlabel('Class')
        # plt.ylabel('Number of samples for each category')
        # plt.title('Training set')
        # if save != None:
        #     plt.savefig(save+'samples.png', dpi=300, bbox_inches='tight')
        # plt.show()
        return self

    def plot(self,type,save = False):
        type = type.lower()
        print('type:',type,'- save:',save)
        if type == 'confusion matrix':
            self.plotConfusionmatrix(save)
        elif type == 'roc':
            self.plotRoc(save)
        elif type == 'samples':
            self.plotSamples(save)
        else:
            print('Invalid Plot')
        return self
        
    def name(self,name = None):
        if name is None:
            return self._name
        else:
            self._name = name
        return self

    def randomForest(self, test_size =0.2, random_state = None ):
        if test_size != None:
            test_size = float(test_size)
        if random_state != None:
            random_state = float(random_state)
        self.name('Random Forest')
        self.train(random_state, test_size)
        self.clf(RandomForestClassifier(random_state=random_state))
        self.model(self.clf().fit(self.trained('X'), self.trained('y')))
        self.y_pred(self.clf().predict(self.tested('X')))
        return self
    
    def knn(self, knn, test_size =0.2, random_state = None):
        if test_size != None:
            test_size = float(test_size)
        if random_state != None:
            random_state = float(random_state)
        if knn != None:
            knn = int(knn)
        self.name('KNN')
        self.train(random_state)
        self.model(KNeighborsClassifier(n_neighbors=knn))
        self.model().fit(self.trained('X'), self.trained('y'))
        self.y_pred(self.model().predict(self.tested('X')))
        return self
    
    def linearSVM(self, test_size =0.2, random_state = None):
            if test_size != None:
                test_size = float(test_size)
            if random_state != None:
                random_state = float(random_state)
            self.name('Linear SVM')
            self.train(random_state)
            self.model(svm.SVC(kernel='linear',probability=True))
            self.model().fit(self.trained('X'), self.trained('y'))
            self.y_pred(self.model().predict(self.tested('X')))
            return self
    
    def nonLinearSVM(self, test_size =0.2, random_state = None):
            if test_size != None:
                test_size = float(test_size)
            if random_state != None:
                random_state = float(random_state)
            self.name('NON linear SVM')
            self.train(random_state)
            self.model(svm.SVC(kernel='rbf',probability=True))
            self.model().fit(self.trained('X'), self.trained('y'))
            self.y_pred(self.model().predict(self.tested('X')))
            return self    

    def models(model,args=[]):
        model=model.lower()
        if model == 'random forest' or model == 'randomforest':
            return AllModels().randomForest(*args)
        if model == 'knn':
            return AllModels().knn(*args)
        if model == 'linearsvm' or model == 'linear svm':
            return AllModels().linearSVM(*args)
        if model == 'nonlinearsvm' or model == 'non linear svm':
            return AllModels().nonLinearSVM(*args)

    def __str__(self) -> str:
        data='Report Dataset : ' + "\r\n"
        data+='--' + "\r\n"
        data+='Shape : ' + str(self.data().shape) + "\r\n"
        data+='****************************************' + "\r\n"
        data+="Accuracy score after training on existing dataset"+ str(self.getScore()) + "\r\n"
        data+='----' + "\r\n"
        data+='Classification Report : ' + "\r\n"
        data+=str(self.classificationReport()) + "\r\n"
        return data

    def save(self,dir):
        if dir is None:
            return self
        f=open(dir+'report.txt','w')
        f.write(str(self))
        f.close()
        return self

    def commandLine():
        plots = []
        arguments = []
        report = True
        output = None
        print(sys.argv[1:])
        for arg in sys.argv[1:]:
            splited = arg.split('=')
            if splited[0] == 'model':
                model = splited[1].strip('"')
            elif splited[0] == 'report':
                report = splited[1].strip('"').lower()
                report = report == 'true'
            elif splited[0] == 'plot':
                plots.append(splited[1].strip('"'))
            elif splited[0] == 'arg':
                arguments.append(splited[1].strip('"'))
            elif splited[0] == 'dir' : 
                output = splited[1].strip('"')
        print('arguments : ',arguments)
        obj = AllModels.models(model,arguments)

        if report:
              obj.reportALl()  

        if output != None and os.path.isdir(output) == False:
            os.makedirs(output)

        for plot in plots:
            obj.plot(plot,output)
            
        obj.save(output)
        



