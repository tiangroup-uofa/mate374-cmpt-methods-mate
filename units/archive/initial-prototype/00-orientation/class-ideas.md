# Lecture 1: Computational modeling in material science and engineerings

Learning outcomes:
- understand the meaning of computational modeling in chemical and materials engineering
- what exactly is the modeling doing?
- understand the 3 criteria of modeling (abstraction of physical phenomena --> mathematical equations analysis --> numerical solutions)
- Basic interaction with the marimo interface we'll introduce

## why modeling?

The digital twin of our era: expense between doing experiments vs numerical and mathematical / computational modeling.

(can show the presumably expense figure comparison between doing exp vs computation)

why is mathematical modeling important? and what are we important about it?

(we can take from the past 374 notes, but make them not so explitictly just bullet points

Advantages of Modeling
• Mathematical modeling is cheaper than experimental investigations.
• Mathematical modeling makes it possible to ’see’ or to ’access’ processes which are
impossible to measure, e.g., diffusion or gasification.
• It allows much better understanding of the experimental data or results.
• It can be used to find the optimum in existing industrial equipment or by novel design of
the next generation of devices.
• Math modeling is not static. It can be improved at any time to expand the range of
applications.
• Finally, computer codes can be seen as a reservoir of knowledge.


Possible Disadvantages/Troubles of Modeling
• The main disadvantage of modeling is its complexity in terms of understanding all
steps and algorithms.
• Using a commercial computational engineering software does not allow us to read the
original codes to see the equations used. Thus, we have to validate 1 carefully any
commercial software before we use it to produce final results and to be sure we use the
right model.
• Very often an open-source computational software, e.g., OpenFoam (computational
fluid dynamics software), does not have appropriate comments in the code, which
makes this code hard to understand.
1 In the view of modeling, validation is a comparison of the numerical solution against a known solution
obtained from an analytical solution or experimental data.

• Simple numerical methods are of great importance due to easy implementation
and programming. However too simple is false because often it can not solve
complex equations or problems.
• At the same time, too sophisticated methods take time to be programmed
correctly, and it can cause difficulties with computations due to ’bugs’ made
by the programming.
• Combination of both previous statements is the key: by solving a problem, one
should go from a simple method to a complex one, doing validation
between.

(show the 3 phases between problem --> abstraction in math --> analytical vs numerical and the way of programming)

## What simulations can and cannot do:

We will introduce a few important examples. For instance the "submersive explosion simulation" (had they knew this before) as a preventive measure, and "the fast16 bug that bugs the iranian nuclear development". This is as introduction to the question to the students

In this part we'd love to say that the class going forward will actually be made by
usually a "general purpose problem" and "material science specific problem". Though they share the same essence.

My suggestion to the lecture notes: equations are important, but do not waste time into understanding every tingle part of equations. Sometimes, making an implementation of the equation will be much much better for understanding and visualization

My own phylosophy: numerical and symbolic maths development are basically able to convert almost everything in heavy textbook into live demos, especially with the help of generative AI. 



## Case study: the calculation of Pi

History of finding Pi

We knew the number of pi is somewhere near 3 from ancient times, but how much can people calculate the number of pi?
the simple division rule --> Zu's 7 decimal number --> someone invented much faster way of computing --> we can easily get the decimals up to how many numbers....

In this demo, we will show students the main interactive approach we'll make in this class, the marimo notebook (a bit of short history can be written in the footnote, appendix, etc). Baically say "if you only have phone / ipad, dont think you cannot do numericals!"

In the case of Pi, we could use this example to show 3 different things

1) How do approximation work?

There could be many ways to infer the length of a circle. E.g. we can just measure the total length of angle division, use the taylor expansion, the euler formula (Basel problem), or even Buffon's needle problem

2) Without seeing the actual problem, can you guess which method is fast, more accurate?

3) How do we measure the "Error" of estimation, if we know the exactly solution to Pi now (up to billions of numbers)

this model is to let students understand what the errors in numerical simulation may come from. (comparison is the each group guess the same number of elements, and see if who can get to the actual method accuracy, how long did it take for your simulation to run?)

class activity can be to measure 1) if the accuracy is good, and sources of numerical error

We will formally introduce 1) analysis of model, calculation convergence vs truncation error.

4) Anything weird did you observe in the error? the round-off error? where does it come from?

<a hint on the demos: do not bother to pick 4 demos, actually make them 1 notebook demo and students could choose to see which using the control button>

<the polling from later class could be done by already polled results, no need to use live updates as can be really really slow>

## Numerical simulation and errors

Key point to remember for this course: methods are available everywhere, but not the same

>> taken from old course lectures

Together, the round-off and truncation errors yield the total or true numerical error that
characterizes each numerical solution.
• This true error is the difference between the true (exact if available) solution and the
numerical solution:
Etrue = TrueSolution − NumericalSolution
• The absolute value of the ratio between the true error and the true solution is called the
true relative error:
TrueSolution − NumericalSolution
EtrueR =
TrueSolution
• If the true solution is unknown, which is a very often case, then use an accurate
numerical solution with a very low value of truncation error and double precision by
defining variables with real numbers1
.
1e.g. π = 3.141592654 is a real number.

>> in next lecture we will understand the number more

# lecture 2: Playing with algorithm, numbers and data

learning outcome: combine the lectures with playable / editable notebooks to display the numbers in our computers. (even though they really run just in a browser)
- plus some simple interaction with the codes (they may not need to understand everything, but a good place to show "just change 2 lines and it will work")

the students have taken ENGCOMP101 but that's VERY INTRODUCTORY. I would prefer to have a few rules still supplied. (And they are free to gemini to get it)

including algorithm, chart, implementation. once flow chart is ok, implementation is easier. and must consider 1) software outliner / edge case and 2) numerical regime / domain etc

what we want the students to know in live class demo: they may or maynot know how to do programming but that's ok.

let's take the example of the pi example last lecture. Could explain: what is an algorithm? from the equation how would you add things up? can we directly write that in a python program?

We can first do a demo using the for loop in python. (Note the equations, introduce native python math, for loop, array, python index from 0 etc etc). The function playground should not let students work from scratch, but rather be a time function and run

a more cleaner version of using just 2 lines of numpy summation. let students to measure the time and report

thirdly if we could use the numba jitlite (must use the marimo in browser, if not then forget about it) to compile the same code and measure

all 3 demos should show the exact error where students can use to analysis, measure the absolute error, relative error.

this will be an already good plan so far, and likely takes about half of our lecture time. students should know exactly what we're doing and whe the result error could be. then we talk about the two types of error by studying the number system in computer.

As usual the class could be going a bit deeper, showing the binary of a number how is it done like in the lecture notes (first for the binary integer, and plus the decimals)

a quick python snippet should be good to show them how to convert to binary from a real number and vice versa

*We will ask students to show how the binary numbers can be converted between each other. *

a quick note to the floating point representation (
in old lecture it's this

Numbers
• Generally, there are three systems of numbers in a computing machine:
• Counting numbers 0, 1, 2, 3,..., 32,... Basically, they are integers. This number system is
connected with the index registers of the machine.
• Second system of numbers is the fixed-point number system, like 3.141592, -12.5678, 0.01234.
These numbers have a fixed length. They are basically used in computing or in data tables.
• The third system of numbers is the floating point number system, which is closely related to the
so-called ’scientific notation’. Examples are 0.3141592 · 101
, 1.25678 · 101
....
• In computers, all numbers defined as ’real’ are represented using floating-point notation. Floating point
representation of numbers is used to store large and small numbers. In general case in this
representation, each number is denoted by a pair of numbers as follows:
where e is referred to as exponent, f is fraction or mantissa, b is the base of number system , q is
a fixed integer called the exponent offset or sometimes as ’bias’1
.
1 The value of q is selected in such way, that numbers with both positive and negative exponents
within a reasonable range can be represented with positive value of e.

Binary floating point representation has the form 1:
• To convert a decimal number into a binary floating-point representation, it can be normalized
to the largest power of 2 that is smaller than the number itself. For example:
1 b is a decimal digit

According to the IEEE-754 standard, computers store numbers and carry out calculations in
single precision or in double precision.

what the actual IEEE754 shows for. 
)

## First-lecture “meet the class” diagnostic

Use a small set of non-graded Wooclap questions during the opening syllabus/map block. The purpose is to learn how to proceed through the semester, not to rank students or test syntax. Save only aggregate results and revisit the baseline after the first seminar or before the midterm.

### Wooclap-ready questions

1. **Python/computational experience** *(single choice, 20 s)*
   - I have not used Python.
   - I have edited simple scripts or notebooks.
   - I have used variables, loops, and functions.
   - I have used NumPy, plotting, or another computational tool.
   - I am unsure what counts as Python experience.

   **Use:** decide how much code scaffolding and seminar remediation are needed. Do not turn a high-confidence response into permission to skip the supplied-function workflow.

2. **Which mathematical object feels least familiar?** *(single choice, 20 s)*
   - Functions and graphs
   - Derivatives and integrals
   - Vectors and matrices
   - Probability and statistics
   - None of these / unsure

   **Use:** identify where short just-in-time refreshers are needed. Keep the course moving; do not front-load a mathematics review unit.

3. **Which materials scale or method are you most curious about?** *(multiple choice, 20 s)*
   - Electronic structure / DFT
   - Atomistic simulation / MD
   - Monte Carlo and statistical sampling
   - Phase field / mesoscale models
   - Continuum methods / FEM or finite volume
   - Data-driven models / ML interatomic potentials
   - I do not know yet

   **Use:** choose examples and project prompts while showing the scale–model–numerical-kernel map. Interest is not a prerequisite survey.

4. **A simulation disagrees with an experiment. What should we check first?** *(multiple choice; allow more than one, 30 s)*
   - Units, input data, and parameter values
   - Model assumptions and physical regime
   - Boundary/initial conditions or geometry
   - Code, discretization, convergence, and solver diagnostics
   - Experimental uncertainty and data processing
   - I would need more information before choosing

   **Use:** surface the misconception that there is one universal cause of disagreement. Follow with the course credibility loop and distinguish verification from validation.

5. **What device will you usually use for course computation?** *(single choice, 20 s)*
   - Laptop/desktop with reliable browser access
   - Tablet
   - Phone
   - Shared or lab computer
   - I do not know yet / access may be inconsistent

   **Use:** plan paired work, static fallbacks, and realistic notebook dimensions. Do not use this to assume that every student can install local software.

### Instructor response rules

- If Python experience is mixed, retain supplied function signatures and editable bodies; do not make the first week a blank-page programming boot camp.
- If mathematical confidence is low, add worked diagrams and short refreshers immediately before use.
- If device access is uneven, make the static/PDF path a first-class route and pair students without treating it as remediation.
- If students name DFT, MD, MC, phase field, FEM, and MLIP as equivalent “methods,” return to the map: scale/model family, mathematical representation, and numerical kernel are different layers.
- Use the results to adjust pacing and examples, not to lower the conceptual target.


I would argue this lecture will end with a notation of the floating point representation (so they know there ARE single precition and double precision but we'll leave for next class)

