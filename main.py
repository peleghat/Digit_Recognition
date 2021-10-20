import numpy as np
import matplotlib.pyplot as plt
import loadMNIST_V2


def target(x, labels):  # labels = [c1, c2]
    m = np.shape(x)[1]
    c1 = labels[0]
    c2 = labels[1]

    def objective_func(w):
        return -1 / m * (c1.T @ np.log(sigmoid(x, w)) + c2.T @ np.log(1 - sigmoid(x, w)))

    def gradient(w):
        return 1 / m * x @ (sigmoid(x, w) - c1)

    def hessian(w):
        D = np.diag(sigmoid(x, w)) @ np.diag((1 - sigmoid(x, w)))
        return 1 / m * x @ D @ x.T

    return objective_func, gradient, hessian


def sigmoid(x, w):
    return 1 / (1 + np.exp(- x.T @ w))


def gradTest(x, w, labels):
    epsilon = np.arange(2, 0, -0.1)
    # epsilon = np.arange(0, 0.5, 0.025)
    noGrad = np.zeros(20)
    objective_w = target(x, labels)[0]
    gradient = target(x, labels)[1]
    hessian = target(x, labels)[2]
    objective_w_val = objective_w(w)
    gradient_val = gradient(w)
    hessian_val = hessian(w)
    withGrad = np.zeros(20)
    noHessian = np.zeros(20)
    withHessian = np.zeros(20)
    d = np.random.rand(np.size(w))
    for i in range(20):
        noGrad[19 - i] = abs(objective_w(w + epsilon[i] * d) - objective_w_val)
        withGrad[19 - i] = abs(objective_w(w + epsilon[i] * d) - objective_w_val - epsilon[i] * d.T @ gradient_val)

        grad_w = gradient(w + epsilon[i] * d)
        noHessian[19 - i] = np.linalg.norm(grad_w - gradient_val)
        withHessian[19 - i] = np.linalg.norm(grad_w - gradient_val - epsilon[i] * hessian_val @ d)

    plt.plot(epsilon, noGrad, label="GradTest without gradient")
    plt.plot(epsilon, withGrad, label="GradTest with gradient")

    # plt.legend()
    # plt.title('Gradient Tests')
    # plt.show()

    plt.plot(epsilon, noHessian, label="JacobianTest without Hessian")
    plt.plot(epsilon, withHessian, label="JacobianTest with Hessian")

    plt.legend()
    plt.title('Jacobian Tests for Hessian')
    plt.show()


"""
x = np.random.rand(3, 3)
labels = np.array([[0, 1, 0], [1, 0, 1]])
w = np.random.rand(3)
gradTest(x, w, labels)
"""
#### Question 4C
data = loadMNIST_V2.MnistDataloader("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte")
(x_train, y_train), (x_test, y_test) = data.load_data()
train_size = len(x_train)  # len(x_train)
test_size = len(x_test)  # len(x_test)
imageList = list()
print(train_size, " ", test_size)


def filter_0_1(x_train, y_train):
    filteredImageList = []
    for i in range(3000):  # 30,000
        if y_train[i] == 0 or y_train[i] == 1:
            filteredImageList.append((x_train[i], y_train[i]))
    return filteredImageList

def filter_8_9(x_train, y_train):
    filteredImageList = []
    for i in range(10000):  # 30,000
        if y_train[i] == 8 or y_train[i] == 9:
            filteredImageList.append((x_train[i], y_train[i]))
    return filteredImageList

def makeLabels_0_1(filteredImageList):
    length = len(filteredImageList)
    c1 = []
    c2 = []
    for i in range(length):
        c1.append(filteredImageList[i][1])  # [0 1 1 0 0]
        c2.append(1 - filteredImageList[i][1])  # [1 0 0 1 1]
    return np.asarray([c1, c2])

def makeLabels_8_9(filteredImageList):
    length = len(filteredImageList)
    c1 = []
    c2 = []
    for i in range(length):
        if filteredImageList[i][1] == 9:
            c1.append(1)  # [0 1 1 0 0]
            c2.append(0)  # [1 0 0 1 1]
        else:
            c1.append(0)  # [0 1 1 0 0]
            c2.append(1)  # [1 0 0 1 1]
    return np.asarray([c1, c2])


def makeXdata(filteredImageList):
    # [img1.flatten() img2.flatte() …. imgN.flatten()]
    length = len(filteredImageList)
    Xdata = []
    for i in range(length):
        image = np.array(filteredImageList[i][0]).flatten()
        for k in range(784):
            image[k] = image[k] / 255
        Xdata.append(image)
    return np.asarray(Xdata)


def Armijo(w, objective_f, gradient_f_w, d, maxiter, alpha_0, beta, c):
    alpha_j = alpha_0
    fw = objective_f(w)
    for j in range(maxiter):
        alpha_j = alpha_0 * (beta ** j)
        if objective_f(w + alpha_j * d) <= fw + c * alpha_j * np.inner(gradient_f_w, d):
            break
        else:
            alpha_j = beta * alpha_j
    return alpha_j


# Steepest(or Gradient) Descent (SD)
def SD(maxiter, epsilon, x_data, labels):
    w_initial = np.zeros(784)
    w_arr = [w_initial]
    obj, grad, hes = target(x_data, labels)
    f_result = [obj(w_initial)]
    w_curr = w_initial
    grad_w_init = grad(w_initial)
    for i in range(maxiter):
        w_prev = w_curr
        alpha = Armijo(w_prev, obj, grad(w_prev), -grad(w_prev), 10, 1, 1 / 2, 1e-4)
        w_curr = w_prev + alpha * -grad(w_prev)
        w_curr = np.clip(w_curr, -1, 1)
        w_arr.append(w_curr)
        f_result.append(obj(w_curr))
        if i > 0 and np.linalg.norm(grad(w_curr)) / np.linalg.norm(grad_w_init) < epsilon:
            break
    return w_curr, f_result, w_arr


def newton(maxiter, epsilon, x_data, labels):
    w_initial = np.ones(784)
    w_arr = [w_initial]
    obj, grad, hes = target(x_data, labels)
    f_result = [obj(w_initial)]
    w_curr = w_initial
    grad_w_init = grad(w_initial)
    for i in range(maxiter):
        print(i)
        w_prev = w_curr
        d = - np.linalg.inv(hes(w_prev) + 0.001 * np.eye(np.shape(hes(w_prev))[0])) @ grad(w_prev)
        alpha = Armijo(w_prev, obj, grad(w_prev), d, 10, 1, 1 / 2, 1e-4)
        w_curr = w_prev + alpha * d
        w_curr = np.clip(w_curr, -1, 1)
        w_arr.append(w_curr)
        f_result.append(obj(w_curr))
        if i > 0 and np.linalg.norm(grad(w_curr)) / np.linalg.norm(grad_w_init) < epsilon:
            break
    print(np.shape (w_curr))
    return w_curr, f_result, w_arr



def makeGraph(w, f_result_train, f_result_test, x_train, labels_train, x_test, labels_test):
    obj_train = target(x_train, labels_train)[0]
    obj_test = target(x_test, labels_test)[0]
    f_w_star = obj_train(w)
    f_w_star_test = obj_test(w)
    output_train = []
    output_test = []
    for i in range(100):
        output_train.append(np.abs(f_result_train[i] - f_w_star))
        output_test.append(np.abs(f_result_test[i] - f_w_star_test))
    plt.title("SD 8_9")
    plt.semilogy(output_train, label="train")
    plt.semilogy(output_test, label="test")
    plt.legend()
    plt.show()


# Train Data - SD
filteredImageList = filter_8_9(x_train, y_train)
x_data = makeXdata(filteredImageList)
labels = makeLabels_8_9(filteredImageList)
wfinal, f_result, w_arr = SD(100, 1e-3, x_data.T, labels)

# Test Data - SD
filteredImageListTest = filter_8_9(x_test, y_test)
x_data_test = makeXdata(filteredImageListTest)
labels_test = makeLabels_8_9(filteredImageListTest)
f_test_arr = []
obj_test = target(x_data_test.T, labels_test)[0]
for i in range(100):
    f_test_arr.append(obj_test(w_arr[i]))
makeGraph(wfinal, f_result, f_test_arr, x_data.T, labels, x_data_test.T, labels_test)


"""
# Train Data - Newton
filteredImageList = filter_0_1(x_train, y_train)
x_data = makeXdata(filteredImageList)
labels = makeLabels_0_1(filteredImageList)
wfinal, f_result, w_arr = SD(100, 1e-3, x_data.T, labels)

# Test Data - Newton
filteredImageListTest = filter_0_1(x_test, y_test)
x_data_test = makeXdata(filteredImageListTest)
labels_test = makeLabels_0_1(filteredImageListTest)
f_test_arr = []
obj_test = target(x_data_test.T, labels_test)[0]
for i in range(100):
    f_test_arr.append(obj_test(w_arr[i]))
makeGraph(wfinal, f_result, f_test_arr, x_data.T, labels, x_data_test.T, labels_test)
"""