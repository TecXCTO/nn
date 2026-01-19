class Network(object):
  def _init__(self, *args, **kwargs):
    #...yada yada, initialize weights and biases...
  def feedforward(self, a):
    """Return the output of the network for an input vector a""n for b, w in zip(self.biases, self.weights):
    a= sigmoid(np.dot(w, a)+b)
    return a
