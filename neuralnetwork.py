from matplotlib import pyplot as plt
import numpy as np

#WE ARE GIVEN THE DIMENSIONS, THE LENGTH AND BREADTH OF THE LEAVES OF TWO DIFFERENT TREES, MANGO TREE AND NEEM TREE.
#MANGO TREE IS REPRESENTED BY 1, NEEM TREE IS REPRESENTED BY 0.
#DATA IS IN THE FORM OF [LENGTH, BREADTH, TREE]
data=[[3, 1.5, 1],
      [2, 1, 0],
      [4, 1.5, 1],
      [3, 1, 0],
      [3.5, 0.5, 1],
      [2, 0.5, 0],
      [5.5, 1, 1],
      [1, 1, 0]]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

#REPRESENTATION OF DATA POINTS ON A GRAPH
#BLUE --> NEEM
#RED --> MANGO
plt.axis([0,6,0,6])
plt.grid()
for i in range(len(data)):
    point=data[i]
    color='r'
    if point[2] == 0:
        color = 'b'
    plt.scatter(point[0],point[1], c=color)
plt.show()
    
#training loop
learning_rate=0.1
costs=[]
#randomly initialising weights and the biases
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(10000):
    point=data[i%8]
    
    z=point[0] * w1 + point[1] * w2 + b
    pred=sigmoid(z)

    target=point[2]
    cost=np.square(pred - target)

    dcost_dpred=2*(pred - target)
    dpred_dz=sigmoid_derivative(z)
    dz_dw1=point[0]
    dz_dw2=point[1]
    dz_db=1
    dcost_dz=dcost_dpred * dpred_dz
    dcost_dw1=dcost_dz * dz_dw1
    dcost_dw2=dcost_dz * dz_dw2
    dcost_db=dcost_dz * dz_db

    w1=w1 - learning_rate * dcost_dw1
    w2=w2 - learning_rate * dcost_dw2
    b=b - learning_rate * dcost_db

    if i%100==0:
        cost_sum=0
        for j in range(len(data)):
            point=data[j]
            
            z=point[0] * w1 + point[1] * w2 + b
            pred=sigmoid(z)

            target=point[2]
            cost_sum+=np.square(pred - target)

        costs.append(cost_sum/len(data))

#REPRESENTATION OF THE DECREASE IN SQUARED ERROR COST FUNCTION WITH INCREASE IN NUMBER OF ITERATIONS
plt.plot(costs)
plt.show()

#WE NEED PREDICT THE TREE, WHEN THE DIMENSIONS OF THE LEAF ARE INPUTED BY THE USER.
unknown_tree=eval(input("Enter the input dataset as a list"))
#seeing model predictions
z=unknown_tree[0]*w1 + unknown_tree[1]*w2 + b
if sigmoid(z) > 0.5:
    print("MANGO TREE")
else:
    print("NEEM TREE")

