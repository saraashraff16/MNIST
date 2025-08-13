from sklearn.linear_model import LogisticRegression

def logistic_regression_model(x_train,x_test,y_train):
    
    log_reg=LogisticRegression(max_iter=5000)

    log_reg.fit(x_train,y_train)

    y_pred=log_reg.predict(x_test)

    return y_pred 
