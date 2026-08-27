# Lecture 2: Choosing and juding a numerical method

main question: when seeing a numerical method, how do we know it is a "good" way to solve the problem?

learning outcomes:
- model selection & comparison
- model errors, definitions
- how fast can we reduce the model errors? (truncation errors)

We will use this lecture both as an intercative playground with our past buffon's method in calculating pi to understand many important topics in errors in modeling.

First, let's recall the buffon's method in lecture 1, with the demo below. (do not pay attention to python codings yet).

- how different is the algorithm final answer from the "real pi"? (N=10, N=100, N=10^6)
- if you want to use this model to achieve pi value the 5 digits from real pi, how many N must you choose?
- compare your results with your peer to the left and right, at same N, do your number agree with each other?
- what if we change the parameter of the needle vs spacing?

Let's take a few mins play with it again and write the answer.

<actual answer things now>

1. and 2. you would see when increasing the number of N, you generally have closer and closer value to the pi value.
but how close? if you choose L/D = xxx, you will typically need N=xxx steps to achieve the target.

How do we measure the difference between the calc and final res? We can define the truncation error as

E_T = |V_true - V_calc|

You could also calculate the relative error as

RE_T = |V_true - V_calc| / V_true

How do we measure the reduction of truncation error? we can plot the E_T vs number of points. (you can download the results from above cell as a csv file and upload into here). When using a "fitting" of the curve, we can apprimated say that the error reduces as a power of N^-p. The larger p is, the faster the error (here mostly the truncation error) will be

We can actually use the rate of "convergence" as a measure of model "goodnedd". Now to the voting time:


<Wooclap link: let's make a vote on the buffon's method: do you think it is a good model?
second question, which of the following methods (just equations) would you believe to be a better option than buffons? drop a label question
>

<after the wooclap we can then introduce the "better pi estimation">

- historical polygon approximation (just use number of N to approximate the area or circumference)
- truncation series (gregory-leibniz) pi/4 = 1 - 1/3 + 1/5 (<why is this?>, or the basel problem (I lke basel probem because it's non-negative)
- Ramanujan series (wow!)

<Just these 4 methods. We could introduce now what signature of these methods do you observe?>

Most of them will have a signature of some parameter, but all contain a parameter N meanig steps, so in pythonic design, we could use
```python
def pi_function(N: int, method_name="sss", *additional_params):
   # switch between methods
```

as a common way of writing these functions

The next demonstration will basically show this. Your task is to work in pairs, try to write the basel problem (don't worry if you can't!), you only need the basic python objects. (remember, python list go from 0, and power is "**" not "^")

<when playing with the demo, what did you observe?>
<should see the error actually increase, this something we show the students later, just plot>

## Closing questions

- comparing with all methods, is buffon's a "fast method" or not?
- the error estimation was done by given a ground truth, what if the ground truth is missing?
- why would the error plateau or even increase?
- (you can change the code to use N=10^8 but that will hang in this notebook, how did I make the simuatlion work?)


