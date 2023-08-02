import copy, math
import numpy as np
import csv

# Read the CSV file
with open('train85percent.csv', 'r') as file:
    csv_reader = csv.reader(file)

    # Skip the header if it exists
    header = next(csv_reader, None)

    # Initialize an empty list to store the instances
    instances = []

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Assuming each row represents an instance
        instances.append(row)

# Convert the list of instances to a NumPy array
instances_array = np.array(instances)
instances_without_last = instances_array[:, :-1]

# Create a separate array for the last elements
last_elements_array = instances_array[:, -1]
# print(instances_without_last)
# print(last_elements_array)


X_train = instances_without_last  #(m,n)
y_train = last_elements_array
X_train = np.round(X_train.astype(float), 3)
y_train = np.round(y_train.astype(float), 3)
#w_tmp = np.zeros_like(X_train[0])
#w_tmp = np.zeros(8)
# print("X_train[0] = ", X_train[0])
# print("w_tmp = ", w_tmp)

#array_shape = X_train.shape

# Print the shape of the array
# print("Array shapex:", array_shape)
#array_shape = y_train.shape

# Print the shape of the array
# print("Array shapey:", array_shape)
#

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

X_train, X_mu, X_sigma = zscore_normalize_features(X_train)
maxi = np.max(X_train, axis = 0)
mini = np.min(X_train, axis = 0)
# trshldmax = [3.5, 3, 3, 4.5, 5.5, 4, 5.5, 3.5]
# trshldmin = [-3, -3.5, -3.5, -3, -3, -3, -3, -3]
# row_outliers = np.any(X_train > 3 , axis=1)
#result2 = np.any(arr, axis=0)
# X_train = [~row_outliers]
# print("now xtra : ", X_train)
# maxi :  [3.97111701 2.45414661 2.36295626 4.91744177 5.86334462 4.58028651 5.98271576 4.02233536]
# mini :  [-1.12758411 -3.83671601 -3.65165051 -1.30109578 -0.7122354  -4.21628797 -1.21359227 -1.02490713]

n,m = X_train.shape
# print("nmmm, : ", n , m)
###########filtering outlined datas
X_filtered = np.empty((0, m), dtype=float)
for row in X_train:
    flagf = 0
    for cell in row:
        if (cell > 7 or cell < -7):
            flagf = 1
            #break
    if flagf == 0:
        X_filtered = np.append(X_filtered, [row], axis=0)
        #print("!!!!! ", X_filtered)
#print("XXXXXX: ", X_filtered.shape)


X_train = X_filtered

# print("NORmu: " , X_mu)
# print("NORma : " , X_sigma)
# print("maxi : " , maxi)
# print("mini : " , mini)


def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost = cost / m
    return cost


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    # if (1 + np.exp(-z)) == 0:
    #     print("ZZZZZZ = " , z);
    #     return 0
    g = 1 / (1 + np.exp(-z))

    return g



def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        # print("x[i] = ", X[i])
        # print("w = ", w)
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw



def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """

    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        #if i < 100000:  # prevent resource exhaustion
        if i % math.ceil(num_iters / 10) == 0:
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###
    # Loop over each example
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_wb)
        print(i, "- " , z_wb)
        if f_wb_i >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    ### END CODE HERE ###
    return p





w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

with open('test15percent.csv', 'r') as file:
    csv_reader = csv.reader(file)

    # Skip the header if it exists
    header = next(csv_reader, None)

    # Initialize an empty list to store the instances
    instances = []

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Assuming each row represents an instance
        instances.append(row)

# Convert the list of instances to a NumPy array
instances_array = np.array(instances)
instances_without_last = instances_array[:, :-1]

# Create a separate array for the last elements
last_elements_array = instances_array[:, -1]
# print(instances_without_last)
# print(last_elements_array)

y_test = last_elements_array
X_test = instances_without_last  #(m,n)
X_test = np.round(X_test.astype(float), 3)
X_test = (X_test - X_mu) / X_sigma

pp = predict(X_test, w_out, b_out)
print("predicted : ", pp)
cnt = 0
for i in range(pp.size):
    print(i, "- ", pp[i], y_test[i])
    if int(pp[i]) == int(y_test[i]):
        cnt += 1

ev = cnt / pp.size * 100
print("evaluation : ", ev)

while True:
    Xp = np.zeros((1, 8))
    print("pls enter your infromation to predict:")
    print("pls enter Pregnancies:")
    Xp[0][0] = input()
    print("pls enter Glucose:")
    Xp[0][1] = input()
    print("pls enter BloodPressure:")
    Xp[0][2] = input()
    print("pls enter SkinThickness:")
    Xp[0][3] = input()
    print("pls enter Insulin:")
    Xp[0][4] = input()
    print("pls enter BMI:")
    Xp[0][5] = input()
    print("pls enter DiabetesPedigreeFunction:")
    Xp[0][6] = input()
    print("pls enter Age:")
    Xp[0][7] = input()
    Xp = (Xp - X_mu) / X_sigma
    pre = predict(Xp, w_out, b_out)
    if pre[0]:
        print("We predict that it is Positive!!!\n")
    else:
        print("We predict that it is Negative!!!\n")

    print("\nfor another test press 1 or if you are done, press 0: ")
    tmp = input()
    if(tmp == 0):
        break