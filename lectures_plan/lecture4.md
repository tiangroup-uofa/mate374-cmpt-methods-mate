lecture 4: how do I make computer faster?

Main q: how could we make programs run to adapt large numbers of steps?

Learning outcomes: 
Learn to use “real Python playground “ from molab.
learn how one Python algorithm can be optimized in different ways (use vecotrization,  parallelism, accelerator / gpu) don’t need to memorize but can identify how to make them faster

Last lecture we finally cracked why the error bounce up, but you may realized I capped the programs to only do max N =10^6. In fact if you measure the time of the function you call use time.

In this lecture we will
Use a “real python “ notebook in a molab environment so many real features may be accessed. <please use your Gmail
Acct for signing up> 

The issue with looped python method is that many internal steps from python isn’t fully optimized for numerical results. If you measure the single steps it’s easy to extrapolate to longer simulations. How long would you expect 10^8 steps?

Of course from lecture 2 we know we can work around the model method, but if we cannot figure out a better algorithm first, optimize the way these numbers are calculated, is in general an easier task. A very good example is surprisingly the buffon needle problem we've seen back in lecture 1.

We will provide a few options on this notebook: . numpy provides ways to use a precompiled Fortran code first you could use numpy to put numbers packed into a cpu (make sure we don’t make them confused). You could use np. Sum to calculate the summation of vectorized results 

This approach is generally referred to as vectorization. Does the speed change when the vector shape increases?(yes indeed, but let’s don’t confuse students with cache)

We can further on this method, make the vextorized method use matched memory length. 

Eventually, for a calculation on compute we could have some other ways to make them
Faster. one such way is called just in time (jit) compilation, which compiles a simple Python function code (mostly loop) into machine binary code for the time, so second time will approach the native speed. 

We could see that in the case of the needle problem. How would the result change? 
 it is even possible to use parallelism in jit time as you can realize many of these computations are independent (ie changing order does not affect). 

An even extreme approach is use a hardware accelerator for really heavy computations. How such method would benefit is out of the scope of this lecture itself,  it for certain types of numerical problems including matrix multiplication (happens really frequently in computing the response of material to an external field), gpu will be much much faster than computing on cpu, because many of such operations are simple and can be well put on hardware accelerators. ( we will use the gpu, a very simple implementation of pytorch for the needle problem and see how it goes), 

Many current day pde solver, molecular dynamics, quantum chem and machine learning models would benefit from such 


Closing question for unit 1

We have seen so far 1) how to turn math into code 2) how do we quantify the error and their sources 3) how do we compare between models esp when we don’t know the ground truth 4) practical ways to optimize and accelerate a computational program

Now the practical question: compare 2 numerical methods A and B, if A converges slower to the ground truth, but each step calculation is very cheap, while B can fastly converge to the truth solution, or even is an exact solution, but calculation is very expensive (i.e. many internal steps). Which one should we choose as the "better model" or "better method"? We will actually see this pair of comparison multiple times through this lecture.
