import numpy as np
import pandas as pd
import tensorflow as tf
# In[7]:
import sklearn
# In[8]:
# Створення вектора розміром 10 елементів, який містить числа від 0 до 9
vector = np.arange(10)
# Створення матриці 3x3, заповненої випадковими числами в діапазоні [0, 1)
matrix = np.random.rand(3, 3)
# Виведення на екран створених даних
print(vector)
print(matrix)
# In[9]
scalar = 2
vector_plus_scalar = vector + scalar
print(vector_plus_scalar)
# In[11]:
vector = []
matrix = matrix.dot([2, 3, 4])
print(matrix)
matrix = []
# Створення dataframe зі списку кортежів
data = [('John', 25, 'M', 'Developer'),
        ('Lisa', 28, 'F', 'Engineer'),
        ('Lily', 32, 'F', 'Manager'),
        ('David', 42, 'M', 'Director'),
        ('Mark', 35, 'M', 'CEO')]
df1 = pd.DataFrame(data, columns=['Name', 'Age', 'Gender', 'Job'])
# Створення dataframe зі словника
data = {'Name': ['John', 'Lisa', 'Lily', 'David', 'Mark'],
        'Age': [25, 28, 32, 42, 35],
        'Gender': ['M', 'F', 'F', 'M', 'M'],
        'Job': ['Developer', 'Engineer', 'Manager', 'Director', 'CEO']}
df2 = pd.DataFrame(data)
# Використання методів head() та describe()
print("Перші 2 рядки з df1:")
print(df1.head(2))
print("\nСтатистичні показники df2:")
print(df2.describe())
# Використання методу iloc[]
print("\nРядки 2-3 та стовпці 1-2 з df1:")
print(df1.iloc[1:3, 0:2])
# Використання методу loc[]
print("\nРядки з іменем 'John' та 'David' з df2:")
print(df2.loc[df2['Name'].isin(['John', 'David'])])
# In[19]:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv

# Генеруємо дані для лінійної функції
x_ = np.linspace(0, 10, 100)
y_ = 3*x_ + 1

# Додаємо помилки вимірювання
np.random.seed(0)
y_noisy = y_ + np.random.normal(0, 1, len(x_))

# Побудова графіку функції та згенерованих даних
plt.plot(x_, y_, label='y')
plt.scatter(x_, y_noisy, label='y_noisy')
plt.legend()
plt.show()
# Обчислення похибки за метриками MAE, MSE
mae = np.mean(np.abs(y_ - y_noisy))
mse = np.mean((y_ - y_noisy)**2)

# Вивід значень похибок
print('MAE: ', mae)
print('MSE: ', mse)

# Запис результатів у CSV-файл
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X', 'Y', 'Y_hat', 'mAE', 'mSE'])
    for i in range(len(x_)):
        writer.writerow([x_[i], y_[i], y_noisy[i], mae, mse])

# In[26]:
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
# Генеруємо датасет регресії
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10)
# Генеруємо датасет класифікації
X_cls, y_cls = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0)
# In[21]:
# Для регресії
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2)
# Для класифікації
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2)
# In[24]:
# Використовуємо лінійну регресію
reg_model = LinearRegression()
# Тренуємо модель
reg_model.fit(X_reg_train, y_reg_train)
# Оцінюємо точність моделі
reg_score = reg_model.score(X_reg_test, y_reg_test)
# Виводимо точність моделі
print(f"Регресія. Точність моделі: {reg_score:.2f}")
# Отримуємо прогнозовані значення
y_reg_pred = reg_model.predict(X_reg_test)
# Обчислюємо MAE та MSE
mae = mean_absolute_error(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
# Виводимо MAE та MSE
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
# In[29]:
# Використовуємо лінійну регресію
rf_cl = RandomForestClassifier()
# Тренуємо модель
rf_cl.fit(X_cls_train, y_cls_train)

# Отримуємо прогнозовані значення
y_reg_pred = rf_cl.predict(X_cls_test)

accuracy_score = accuracy_score(y_reg_pred, y_cls_test)
print("Accuracy: ", accuracy_score)



# In[42]:


import tensorflow as tf
import numpy as np
import pandas as pd

# Створення набору даних розміру 10х3
data = tf.random.uniform(shape=(10, 3), minval=0, maxval=100, dtype=tf.float32)
print("Набір даних:\n", data.numpy())

# CRUD операції
# Створення нового запису
new_record = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
data = tf.concat([data, new_record], axis=0)
print("Набір даних з новим записом:\n", data.numpy())

# Оновлення існуючого запису
data = tf.tensor_scatter_nd_update(data, indices=[[3, 1]], updates=[99.0])
print("Набір даних з оновленим записом:\n", data.numpy())

# Видалення запису
data = tf.concat([data[:3], data[4:]], axis=0)
print("Набір даних з видаленим записом:\n", data.numpy())

data = tf.reshape(data, [30])
print("Набір даних з новою формою:\n", data.numpy())

# Інтеграція з np.array
np_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
data = tf.constant(np_array, dtype=tf.float32)
print("Набір даних, інтегрований з np.array:\n", data.numpy())

# Інтеграція з pd.DataFrame
df = pd.DataFrame(np_array, columns=['A', 'B'])
data = tf.constant(df.values, dtype=tf.float32)
print("Набір даних, інтегрований з pd.DataFrame:\n", data.numpy())


# In[ ]:





# In[ ]:




