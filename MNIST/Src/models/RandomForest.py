from sklearn.ensemble import RandomForestClassifier

def Random_Forest_model(x_train,x_test,y_train):
    r_forest=RandomForestClassifier(n_estimators=100,random_state=42)

    r_forest.fit(x_train,y_train)

    y_pred=r_forest.predict(x_test)

    return y_pred 