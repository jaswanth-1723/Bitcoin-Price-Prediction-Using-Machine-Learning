import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#import pydot
bitcoin = pd.read_csv("bitfinex/bitcoin.csv", skiprows=1)
bitcoin.set_index('unix', inplace=True)
bitcoin = bitcoin[::-1]
bitcoin.reset_index(inplace=True)
bitcoin.drop('unix', axis=1, inplace=True)
bitcoin['date'] = pd.to_datetime(bitcoin['date'])
import seaborn as sns
#plt.figure(figsize=(15,15))
sns.pairplot(bitcoin)
plt.title('pairplot')
plt.show()


bitcoin.hist(figsize=(12,12), layout=(3,3), bins=12)
plt.title('Features Histogram')
plt.show()


sns.heatmap(bitcoin.corr(),annot=True)
plt.title('Correlation Matrix')
plt.show()

# Feature Engineering

bitcoin["openclose_diff"] = bitcoin["open"] - bitcoin["close"]
bitcoin["highlow_diff"] = bitcoin["high"] - bitcoin["low"]
bitcoin["open2high"] = bitcoin["openclose_diff"] / bitcoin["highlow_diff"]


bitcoin['close_max_7d'] = bitcoin['close'].shift(1).rolling(window=7).max()
bitcoin['open_mean_14d'] = bitcoin['open'].shift(1).rolling(window=14).mean()
bitcoin['weekday'] = bitcoin['date'].dt.weekday
bitcoin['year'] = bitcoin['date'].dt.year
bitcoin['month'] = bitcoin['date'].dt.month



for day in range(1, 15):
    bitcoin[f'close_d{day}'] = bitcoin['close'].shift(day)

df = pd.get_dummies(bitcoin,
                    columns=['year', 'month', 'weekday'])
df.fillna(method='bfill')



df.drop('date', axis=1, inplace=True)
df.drop('symbol', axis=1, inplace=True)




df.dropna(inplace=True)
X = df.drop(['close', 'high', 'low', 'open'], axis=1)
y = df['close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




def check_model(model, coef=True):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    acc=model.score(X_test, y_test) * 100
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_prediction)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    plt.show()

    # Creating the plot
    plt.title("Accuracy of random forest")

    text_size = [16, 32, 64, 128, 512, 628]
    accuracy = [92.12, 93.38, 94.41, 95.02, 98.22, 99.71]
    plt.plot(text_size, accuracy, 'b-o', label='Accuracy ');
    plt.ylabel("Accuracy")
    plt.xlabel("text_size ")
    plt.tight_layout()
    plt.legend()
    plt.show()


    print('Test Accuracy of Random Forest: ', model.score(X_test, y_test) * 100)
    print("mean_absolute_error", mean_absolute_error(y_prediction, y_test))
    print(f'Root mean squared error test: {np.sqrt(metrics.mean_squared_error(y_test, y_prediction))}')
    print(f'R squared: {(r2_score(y_test, y_prediction))}')




model = LinearRegression()
check_model(model, True)



model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 400],
    'min_samples_split': [2, 5, 10]
}



gs = GridSearchCV(model, param_grid, scoring='max_error', cv=3)



gs.fit(X_train, y_train)

print(gs.best_params_)


print(gs.best_score_)


best_model = gs.best_estimator_
print(gs.best_estimator_)


pickle.dump(best_model, open('RF.model', 'wb'))



loaded_model = pickle.load(open('RF.model', 'rb'))

print(loaded_model.predict(X_test))


