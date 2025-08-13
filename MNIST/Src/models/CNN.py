from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def CNN_model(X_train ,X_test,Y_train,Y_test):
    cnn=models.Sequential([
      layers.Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(8,8,1)),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2,2)),
      layers.Dropout(0.4),

      layers.Conv2D(128,(3,3),activation='relu',padding='same'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2,2)),

      layers.Flatten(),
      layers.Dense(128,activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(10,activation='softmax') 

    ])

    cnn.compile(optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',metrics=['accuracy'])

    cnn.fit(X_train ,Y_train ,epochs=8 ,batch_size=32 ,validation_split=0.1, verbose=1)

    acc_cnn=cnn.evaluate(X_test,Y_test,verbose=0)[1]
    print(f'CNN Accuracy = {acc_cnn:.4f}')

    cnn_pred=cnn.predict(X_test)

    return cnn_pred