from sklearn.neural_network import MLPClassifier

def MLP_model(x_train,x_test,y_train):
    mlp=MLPClassifier(hidden_layer_sizes=100,activation='relu',solver='adam',max_iter=500,random_state=42)

    mlp.fit(x_train,y_train)

    y_pred=mlp.predict(x_test)

    return y_pred 