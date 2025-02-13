# Linear Regression flow process

1. **Get the data** *(CSV, database, etc.)*  
2. **Split into `X` (features) and `y` (target)**  
3. **Scale data if needed** *(only for models sensitive to scaling, e.g., SVR, not necessary for Linear Regression)*  
4. **Split into `X_train`, `X_test`, `y_train`, `y_test`**  
5. **Reshape if single feature (`X = X.reshape(-1, 1)`)**  
6. **Load the model (`LinearRegression()`)**  
7. **Train the model (`model.fit(X_train, y_train)`)**  
8. **Predict (`y_pred = model.predict(X_test)`)**  
9. **Evaluate (`model.score(X_test, y_test)`)**  

### 🚨 **Note:**  
- **`reshape(-1, 1)`** is needed only for **single-feature arrays**.  
- **Scaling** is optional for linear regression unless features are on drastically different scales.  

