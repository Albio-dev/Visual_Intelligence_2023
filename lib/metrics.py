from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
class metrics:

    def __init__(self, y_true, y_pred, lab_classes) -> None:
        self.classes = lab_classes
        _, self.y_pred = y_pred.max(1)
        self.y_true = y_true

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true, self.y_pred)
        self.f1 = f1_score(self.y_true, self.y_pred)

        self.confmat = confusion_matrix(self.y_true, self.y_pred, labels=list(range(0,len(self.classes))))
        fpr, tpr, threshold = roc_curve(self.y_true, self.y_pred)
        self.roc = (fpr, tpr)

    def __str__(self) -> str:
        return f'Accuracy:\t\t{self.accuracy}\nPrecision:\t\t{self.precision}\nRecall:\t\t\t{self.recall}\nF1:\t\t\t\t{self.f1}'
    
    def printMetrics(self, type = None):
        if type is not None:
            print(f'{type} metrics: \n{self}')
        else:
            print(f'Metrics: \n{self}')
        
    def getMetrics(self, type = None):
        if type is not None:
            return f'{type} metrics: \n{self}'
        else:
            return f'Metrics: \n{self}'


    def rocDisplay(self):
        return RocCurveDisplay(*self.roc)

    def confMatDisplay(self):
        return ConfusionMatrixDisplay(self.confmat, display_labels=self.classes)

    def precisionRecallDisplay(self):
        return PrecisionRecallDisplay(self.precision, self.recall)
