from sklearn.neighbors import KNeighborsClassifier

def KNN_model(x_train,x_test,y_train):
    knn=KNeighborsClassifier(n_neighbors=6)

    knn.fit(x_train,y_train)

    y_pred = knn.predict(x_test)

    return y_pred 