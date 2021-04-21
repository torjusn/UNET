
'''
TODOLIST

1.
line between paragraphs in latex
https://tex.stackexchange.com/questions/74170/have-new-line-between-paragraphs-no-indentation

2.
#change size of figures and subplots in python
https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib

3.
#write section on multimodal 

4. 
add ridge to nystrom/add handpicked
https://gist.github.com/daien/e393853a9f1dabdf7461

5.
# Try using S from code in sal paper
try pinv + swap B_x, R_t

'''


'''
first try 
code S, code swap to SVD, pinv

Wxx = vt*s*u

FOR S TRY
1. sal code
S = vt*diag(s^(-1/2))*u

2. sal paper
S = Wxx^(-1/2)

3. 
S = Bx*Gamma^(-1/2)*Bx^T

'''

'''
GRAPH FUSION
not mentioned in papers
benefit of diffusing s times, allowed to do considering different energy function?


GRAPH INFERENCE LEARNING
1.
How to define graph structure for image?
G = V, E, X
Vertices, adjacency between nodes, explicit/implicit attributes of vertices

-can use weighted fused graph of similarity
-can extract features by FCN (conv) as in Label prop

2. Definition of subgraph
-neighborhood around node v_i

3.
Point 3: class to node relationship maps vector of path reachability into a weight by two 16 dim fully connected layers into w
Does this mean just using nn.Linear(ch_vectorin, 16), nn.Linear(16, 1)
'''


'''
limitiations of parallell like graph fusion from theory point of view, bad results might be explained in that respect, concern for me to be sure that the algorithm

summarize in slides/short report:
list all architectures i have implemented and which setups im about to test 
for clear overview so far, and if we need the gil architecture in place already or not
'''
