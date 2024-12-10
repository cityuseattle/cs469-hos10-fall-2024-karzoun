from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

neighbor_values = [1, 3, 5, 7, 9]

for n_neighbors in neighbor_values:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"n_neighbors={n_neighbors}: Accuracy = {accuracy}")
    print(y_pred)
    print(y_test)
    print()
