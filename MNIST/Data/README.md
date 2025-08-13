# Project Data

The MNIST-like dataset is downloaded automatically when running the code using **Scikit-learn**.

Code:
```python
from sklearn.datasets import load_digits

digits = load_digits()

x = digits.data
y = digits.target
