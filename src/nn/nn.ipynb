{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "adce1336-a4d4-427f-aa63-6c7e72fe5af8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Matrix multiplication\n",
    "A = np.array([[1, 2], [3, 4]])\n",
    "B = np.array([[5], [6]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6d72981b-b5fa-452a-81b5-f0c97004d234",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A @ B = [[17]\n",
      " [39]]\n"
     ]
    }
   ],
   "source": [
    "print(\"A @ B =\", A @ B)          # dot product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e82b74cf-ea8b-4081-b0fb-ca1df5e1099c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A * B = [[ 5 10]\n",
      " [18 24]]\n"
     ]
    }
   ],
   "source": [
    "print(\"A * B =\", A * B)          # element‑wise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f77be24a-ab65-4cfe-a53f-c208cdd760d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "# Artificial Neuron (Perceptron)\n",
    " Concept :\n",
    "  Input vector x, weights w, bias b.\n",
    "  Output y = σ(w·x + b) where σ is an activation (step, sigmoid, ReLU, …).\n",
    "  Training: adjust w, b to minimise error.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "dac42d25-49a7-41b5-b6ed-7457ca45f747",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Code – Vanilla perceptron\n",
    "\n",
    "# import numpy as np\n",
    "\n",
    "class Perceptron:\n",
    "    def __init__(self, dim, lr=0.1):\n",
    "        # cut from here \n",
    "        rn = np.random.randn(dim)\n",
    "        print(\"random number\",rn)\n",
    "        self.w = rn * 0.01\n",
    "        print(\"w when initialize=\",self.w)\n",
    "        # to at then remove # from following line\n",
    "        # self.w = np.random.randn(dim) * 0.01\n",
    "        self.b = 0.0\n",
    "        self.lr = lr\n",
    "\n",
    "    def predict(self, x):\n",
    "        z = np.dot(self.w, x) + self.b\n",
    "        return 1 if z > 0 else 0\n",
    "\n",
    "    def train(self, X, Y, epochs=10):\n",
    "        for epoch in range(epochs):\n",
    "            print(\"epoch=\",epoch)\n",
    "            for x, y in zip(X, Y):\n",
    "                pred = self.predict(x)\n",
    "                error = y - pred\n",
    "                print(\"w=\",self.w,\"b=\",self.b)\n",
    "                self.w += self.lr * error * x\n",
    "                self.b += self.lr * error\n",
    "                print(\"x=\",x, \"=>\", \"prediction of y:\",pred, \"y as target:\", y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "827a7a42-70ee-4f5e-8227-ee85a7f37e1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "random number [2.02383041 0.31513811]\n",
      "w when initialize= [0.0202383  0.00315138]\n",
      "epoch= 0\n",
      "w= [0.0202383  0.00315138] b= 0.0\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.0202383  0.00315138] b= 0.0\n",
      "x= [0 1] => prediction of y: 1 y as target: 0\n",
      "w= [ 0.0202383  -0.09684862] b= -0.1\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [ 0.0202383  -0.09684862] b= -0.1\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 1\n",
      "w= [0.1202383  0.00315138] b= 0.0\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= 0.0\n",
      "x= [0 1] => prediction of y: 1 y as target: 0\n",
      "w= [ 0.1202383  -0.09684862] b= -0.1\n",
      "x= [1 0] => prediction of y: 1 y as target: 0\n",
      "w= [ 0.0202383  -0.09684862] b= -0.2\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 2\n",
      "w= [0.1202383  0.00315138] b= -0.1\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= -0.1\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= -0.1\n",
      "x= [1 0] => prediction of y: 1 y as target: 0\n",
      "w= [0.0202383  0.00315138] b= -0.2\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 3\n",
      "w= [0.1202383  0.10315138] b= -0.1\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.1202383  0.10315138] b= -0.1\n",
      "x= [0 1] => prediction of y: 1 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= -0.2\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= -0.2\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 4\n",
      "w= [0.2202383  0.10315138] b= -0.1\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.1\n",
      "x= [0 1] => prediction of y: 1 y as target: 0\n",
      "w= [0.2202383  0.00315138] b= -0.2\n",
      "x= [1 0] => prediction of y: 1 y as target: 0\n",
      "w= [0.1202383  0.00315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 5\n",
      "w= [0.2202383  0.10315138] b= -0.20000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.20000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.20000000000000004\n",
      "x= [1 0] => prediction of y: 1 y as target: 0\n",
      "w= [0.1202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 0 y as target: 1\n",
      "epoch= 6\n",
      "w= [0.2202383  0.20315138] b= -0.20000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.20315138] b= -0.20000000000000004\n",
      "x= [0 1] => prediction of y: 1 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 7\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 8\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 9\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 10\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 11\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 12\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 13\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 14\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 15\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 16\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 17\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 18\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "epoch= 19\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [0 1] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 0] => prediction of y: 0 y as target: 0\n",
      "w= [0.2202383  0.10315138] b= -0.30000000000000004\n",
      "x= [1 1] => prediction of y: 1 y as target: 1\n",
      "[0 0] => 0 target: 0\n",
      "[0 1] => 0 target: 0\n",
      "[1 0] => 0 target: 0\n",
      "[1 1] => 1 target: 1\n"
     ]
    }
   ],
   "source": [
    "X = np.array([[0,0],[0,1],[1,0],[1,1]])\n",
    "Y = np.array([0,0,0,1])\n",
    "\n",
    "net = Perceptron(dim=2, lr=0.1)\n",
    "net.train(X, Y, epochs=20)\n",
    "\n",
    "for x, y in zip(X, Y):\n",
    "    print(x, \"=>\", net.predict(x), \"target:\", y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d9b1a701-eb01-4b2b-b8cc-421f75a9020c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1]]\n"
     ]
    }
   ],
   "source": [
    "r = np.array([[0],[1]])\n",
    "c = np.array([[0,1]])\n",
    "dcr = np.dot(c, r)\n",
    "print(dcr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a042c1c9-9236-4c50-b1d1-dcffb6dcac8d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 0]\n",
      " [0 1]]\n"
     ]
    }
   ],
   "source": [
    "drc = np.dot(r, c)\n",
    "print(drc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a258cf3b-15d9-4ad8-9116-b407498849b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "dcr = np.dot(c, r)\n",
    "print(dcr)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
