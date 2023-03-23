from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from matplotlib import pyplot as plt
class metrics:

    def __init__(self, y_true, y_pred, lab_classes) -> None:
        # Set class variables
        self.classes = lab_classes
        _, self.y_pred = y_pred.max(1)
        self.y_true = y_true

        # Calculate metrics
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true, self.y_pred)
        self.f1 = f1_score(self.y_true, self.y_pred)

        # Calculate confusion matrix
        self.confmat = confusion_matrix(self.y_true, self.y_pred, labels=list(range(0,len(self.classes))))
        fpr, tpr, threshold = roc_curve(self.y_true, self.y_pred)
        self.roc = (fpr, tpr)

    # Method to turn metrics into string
    def __str__(self) -> str:
        return f'Accuracy:\t\t{self.accuracy}\nPrecision:\t\t{self.precision}\nRecall:\t\t\t{self.recall}\nF1:\t\t\t\t{self.f1}'
    
    # Wrapper to print metrics
    def printMetrics(self, type = None):
        if type is not None:
            print(f'{type} metrics: \n{self}')
        else:
            print(f'Metrics: \n{self}')
        
    # Wrapper to return metrics string
    def getMetrics(self, type = None):
        if type is not None:
            return f'{type} metrics: \n{self}'
        else:
            return f'Metrics: \n{self}'

    # ROC display wrapper
    def rocDisplay(self):
        return RocCurveDisplay(*self.roc)

    # Confusion matrix display wrapper
    def confMatDisplay(self):
        return ConfusionMatrixDisplay(self.confmat, display_labels=self.classes)

    # Precision recall display wrapper
    def precisionRecallDisplay(self):
        return PrecisionRecallDisplay(self.precision, self.recall)

    # Helper to plot training data
    def plotTraining( data, axs = None):   
        if axs is None:
            fig, axes = plt.subplots(1,2, figsize=(15,5))
            fig.show()
            
        x_scale = range(len(data['loss']))
        x_scale_val = [i*10 for i in range(len(data['loss_val']))]
        axs[0].plot(x_scale, data['loss'], label='training_loss')
        axs[0].plot(x_scale_val, data['loss_val'], label='validation_loss')
        axs[0].set_title('Loss')
        axs[0].legend()
        axs[1].plot(x_scale, data['accuracy'], label='training_accuracy')        
        axs[1].plot(x_scale_val, data['accuracy_val'], label='validation_accuracy')
        axs[1].set_title('Accuracy')
        axs[1].legend()
        