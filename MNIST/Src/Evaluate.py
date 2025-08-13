from sklearn.metrics import  accuracy_score, classification_report, r2_score, mean_squared_error

def r2_scoree(y_test,y_pred):
    r2score=r2_score(y_test,y_pred)
    print(f'R2 Score {r2score}')

def mse(y_test,y_pred):    
    mse=mean_squared_error(y_test,y_pred)
    print(f'MSE = {mse}')

def print_accuracy(y_test,y_pred):
    acc=accuracy_score(y_test,y_pred)
    print(f"Accuracy= {acc}")

def ClassificationReport(y_test,y_pred):
    report=classification_report(y_test,y_pred)
    print(f"Classification Report \n",report)