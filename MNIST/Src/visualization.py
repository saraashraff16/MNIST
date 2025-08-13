import matplotlib.pyplot as plt

def visualize_results(df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    axs[0].scatter(df['models'],df['MSE'], color='skyblue',s=100)
    axs[0].set_title('Mean Squared Error (MSE)')
    axs[0].set_ylabel('MSE')

    axs[1].scatter(df['models'],df['R2 Score'], color='lightgreen',s=100)
    axs[1].set_title('R² Score')
    axs[1].set_ylabel('R²')

    axs[2].scatter(df['models'],df['Accuracy'],color='salmon',s=100)
    axs[2].set_title('Accuracy')
    axs[2].set_ylabel('Accuracy')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()