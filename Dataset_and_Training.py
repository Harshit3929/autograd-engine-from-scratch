# Creating a random dataset
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4,4,1])


for k in range (10):

    ypred = [n([Value(xi) for xi in x]) for x in xs]
    loss = 0
    for xi, yi in zip(xs, ys):
        xi = [Value(xij) for xij in xi]
        yi = Value(yi)

        pred = n(xi)
        loss += (pred - yi)**2

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss)