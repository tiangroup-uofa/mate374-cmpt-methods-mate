# Lecture 1: Numerical Mindset for Material engineers

## Learn outcomes 
- be aware of the concepts of computational / numerical and simulations
- when taking about numerical methods / modeling, what should we be aware of?
- common types of numerical and simulation methods in materials engineering
- familiar with interactive class components (wooclap survey / marimo interactive notebooks / demos)
- Sources of error from a interactive demonstration perspective


## Getting to know you!

As mentioned before there will be several types of in-class activities. The first is a wooclap interaction <show activity link>. Let's survey your background and knowledge about coding / numeri / simulation. (results to follow on canvas)

## Why do we need modeling?

### Stages of modeling

<We can then show the 3 stages of modeling, recreated from the old 374 lecture>

<basically problem statement --> phase 1 (new or adaptation of math model) --> phase 2 (construction of an appropriate numerical model / simulation package / machine learning algorithm) --> phase 3 solution of the phase 2 prgram as reulsts>

### Pros and cons of modeling

<what joke should we use? comics>

<as mentioned in lecture 0, link> there are actually 3 types of methods we call "modeling": mathematical, numerical and computational simulations. While you'll learn many different modeling throughout the course, there are several features of modeling we should be aware of:

1. Motivation: modeling is *usually* cheaper than experimental invesitgations
2. Exploration: modeling makes it possible to *access* processes / structures that are impossible to measure using nowaday techniques (e.g. diffusion, atoms motion & rearrangements)
3. better understanding of exp results <can we paraphrase>
4. Design: use modeling to find the optimum
5. Improvable: modeling itself is not static
6. Knowledge: models written down to code, is just another form of reservoir of knowdge (that you, computer and even an AI can have access to)

But most importantly, we should know that modeling comes with disadvantages as well

1. Accuracy / complexity trade off: Models do come with different level accuracy and complexity. Do you want to solve the wavefunction of the whole universe to be able to explain how a trajectory of a basket ball is thrown?
2. Open / closedness duality: 2 kinds of numerical packages exists, do you trust the results of commercial, close source packages (e.g. MATLAB, Ansys, Comsol), or do you trust the correctness of implementation (i.e. no software errors, so called bugs) in open source codes (i.e your own python code)?

How do we strike a balance? by solving a problem, one
should go from a simple method to a complex one, doing **validation**
between. (the above sentences are from old 374)

### Examples of useful modeling and not useful modeling <weird name, please modify>


Simulations of materials have greatly enhanced the way research study and search for useful materials that can potentially impact the world for the better, from the discovery of diffusion laws (Austen experimnts, where the eqns are solved only by pencil and paper), to the huge materials genome project (<more explanation / links>), to recent advancement of machine learning / language model prediction of materials.
<each of the materials above should be linked to one image, so that students are not afraid of too much text, but also citations needs to be added>

Modeling / simulations indeed can help we guide to new materials discovery, but what about we don't fully understand the modeling details / results? What even if the results are actually wrong?

There are 2 examples from the last 5 years I wanted to share.

1) the OceanGate Titan submersible explosion accident (<happened which day, which year?link to wikipedia for the whole event>, full report https://www.ntsb.gov/investigations/AccidentReports/Reports/MIR2536.pdf, and the link to the FEM analysis report https://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=pdf&FileName=CG-019+Spencer+Composites+FEA_Redacted-Rel.pdf&ID=18926631 )

This tragedy happened partly as a combined material / mechanical engineering failure. The carbon fiber composite material used for the vessel pressure compartment wasn't validated to the actual strength and durability.

<key take away “a model can be internally sophisticated yet still fail as engineering evidence if the model assumptions and validation are weak.”>

<some gpt-generated responses Titan’s carbon-fibre pressure hull accumulated delamination damage and eventually failed by local buckling under deep-sea pressure. The problem was not simply that “carbon fibre was bad”: the investigation found that OceanGate had not adequately established the as-built hull’s actual strength, defects, or cycle life, so analyses based on idealized material/geometry assumptions were not enough to guarantee the real structure. i.e. your modeling criteria wasn't up to the actual criteria. FEA answers the model you give it, not necessarily the object you built., what if the model isn't good enough?>

<new simulation https://www.engineering.com/simulation-reveals-exactly-how-titan-submersible-imploded>

2) The infamouse software bug-virus for iranian nuclear development.

We mentioned the potential disadvantage of close-soruced simulation software. This happened in real-world politics. In 2025 <check year> a group of researchers <who> found a 20-year-long bug planted into the national material simulation facility supercomputers <name correct?> in iran atomic facilities. The bug produced a stealth way of introducing numerical errors to the equation-of-state (EOS) simulation of nucleactive materials. (i.e. to answer questions how compressible is that metal in high pressure?)

<show the bugged EOS curve, some eventual bug, link to the article. >

<the bug was very stealth and even infected every single copy of same software on the cluster, so anyone running similar simulation would not notice>

<link to the failure ansys / FEM simulation using the wrong results, from twitter link>

These examples prompt us to ask the questions: if we are going to run any numerical programs, how do we trust the results?

## Errors in modeling

### Sources of errors

The most important thing to remember for this lecture is the source of
errors in modeling.

*modeling error* <can we call it epistemic uncertainty? >: comes from phase I of modeling, that we lack foundamental knowledge of the subject to be studies, for example a physicst using a "vacuum spherical cow", or we mat / chemists use a orbital model for electrons. Basically, "the model is not good", it cannot be improved by ajust parameters of the model

*in phase II*, we use a numerical code to approximate a math idea. error can still occur because of mainly 3 factors:
1) uncertainty abou the input / parameter: uncertainty of the input parameter will cause the final result to differ, i.e. the uncertainty propagation
2) the numerical model itself is not an exact representation (i.e. approximation). Any approximate not to the infinite process is characterized in the truncation error
3) the math / physics system it may involve randomness (stochastical process, or aleatoric uncertainty), in fact many chemistry / materials process fall within this category,

In the final stage (phase III), we will inevitably encounter the limitation from the computer themselves: the round-off error: the error itself came from the finite length of a real number (in computer systems called a floating number. error from the floating numbers will accumulate. Think of you use an old calculator 1/3 --> 0.3333  * 3 = 0.99999)

### Simple demonstration

We will use interactive code demo in this lecture to showcase the sources of the errors. <for all the code demo in this course, we will use the marimo interactive notebook, that you don't need to download an actual python notebook.>

<footnote: why dont I use a local IDE like spyder, or jupyter? Many considerations, but I do focus on the equity of using codes. <this rant should be just very short>>

When thinking about calculating the number of pi, what method can you think of? (pi itself is transdencial number, meaning it will never ends, and the best we can do is to either "measure it" or mathematically "approximate it"). In 1700s <eactly year>, famous physict Buffon invented an algorithm that use random needle throwing to determine the number of Pi.
- <explain the idea quickly>
- <explain it is a random process, how to have randomness? start with your student number as integer in the input box>
- <explain you need to drag the slider and press simulate, to get the numeber>

When interacting with the code below, please see that you can both change the value of the sliders, interact with the button, input value, and even change the code! (frequenty appeared boiler plate codes will be hidden)

<need to add the buffon exp later>

## Closing questions

You can play with the buffon's demo as many time as you wish, but to notice the following aspects

- how different is the algorithm final answer from the "real pi"? (N=10, N=100, N=10^6)
- if you want to use this model to achieve pi value the 5 digits from real pi, how many N must you choose?
- compare your results with your peer to the left and right, at same N, do your number agree with each other?
- what if we change the parameter of the needle vs spacing?

As you can see, they cover many of the error in modeling questions we provided above. We will follow the same route to the question in nect lecture.


