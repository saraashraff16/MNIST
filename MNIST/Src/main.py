from Src.data_utils import load_data,explore_data,scale_data,split_data,data_for_cnn
from Src.Evaluate import  r2_scoree,mse,print_accuracy,ClassificationReport
from Src.models.LogisticReg import logistic_regression_model
from Src.models.KNN import  KNN_model
from Src.models.RandomForest import Random_Forest_model
from Src.models.MLP import MLP_model
from Src.models.CNN import CNN_model
#----------------------------------------------------------

def main():

    x, y, digits = load_data()
    
    explore_data(digits, x, y)

    x_scaled = scale_data(x)

    x_train, x_test, y_train, y_test = split_data(x_scaled, y)

    print("\n--- Logistic Regression ---")
    y_pred = logistic_regression_model(x_train, x_test, y_train)
    print_accuracy(y_test, y_pred)
    mse(y_test, y_pred)
    r2_scoree(y_test, y_pred)
    ClassificationReport(y_test, y_pred)

    print("\n--- KNN ---")
    y_pred = KNN_model(x_train, x_test, y_train)
    print_accuracy(y_test, y_pred)
    mse(y_test, y_pred)
    r2_scoree(y_test, y_pred)
    ClassificationReport(y_test, y_pred)

    print("\n--- Random Forest ---")
    y_pred = Random_Forest_model(x_train, x_test, y_train)
    print_accuracy(y_test, y_pred)
    mse(y_test, y_pred)
    r2_scoree(y_test, y_pred)
    ClassificationReport(y_test, y_pred)

    print("\n--- MLP ---")
    y_pred = MLP_model(x_train, x_test, y_train)
    print_accuracy(y_test, y_pred)
    mse(y_test, y_pred)
    r2_scoree(y_test, y_pred)
    ClassificationReport(y_test, y_pred)

    x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn = data_for_cnn()
    print("\n--- CNN ---")
    y_pred = CNN_model(x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn)
    print_accuracy(y_test_cnn, y_pred)
    mse(y_test_cnn, y_pred)
    r2_scoree(y_test_cnn, y_pred)
    ClassificationReport(y_test_cnn, y_pred)

if __name__ == "__main__":
    main()
