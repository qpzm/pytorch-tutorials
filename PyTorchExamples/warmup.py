import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Wrong
# w1 = np.random.randn(H, D_in)
# w2 = np.random.randn(D_out, H)

# x data matrix (batch size, input dim)
# tutorial에서는 아래와 같이 정의
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(1000):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Computer and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop
    grad = {}
    grad['y_pred'] = 2 * (y_pred - y)
    grad['w2'] = h_relu.T.dot(grad['y_pred'])
    grad['h_relu'] = grad['y_pred'].dot(w2.T)
    grad['h'] = grad['h_relu'].copy()
    grad['h'][h < 0] = 0
    grad['w1'] = x.T.dot(grad['h'])

    # Update weights
    w1 -= learning_rate * grad['w1']
    w2 -= learning_rate * grad['w2']
