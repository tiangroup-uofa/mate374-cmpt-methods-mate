Lecture 3: Numbers in Computer Systems

Main question: how do I trust the numbers from my computer, and do they have limitations?

Learning outcomes:
- recall the meaning of floating number in computers 
- Identify sources of round off error in computers, 
- Know the tradeoff between more number precision and result error

In lecture we started from the simple question of throwing the needle to estimate pi and eventually several variations can we increase the number of N Knfinitely? If you’re patient enough to wait (or know the way to optimize the calculation we’ll introduce next lecture), you’ll notice not really, at some point the error climbs up again. So what’s happening?

Our computers (or at least the digital computers as we know nowadays,<footnote there are indeed analog computers that can represent continuous spectra of numbers, cite>) use binary 0-1 digits to represent numbers, in contract to the digital number we write in python. 

To convert between binary and decimal numbers the rule is simple: you solve a polynomial equation that a_i * 2^i = the number in decimal <need a display eqn> and vice versa. The total amount is the same, you only change the representation of binary digits or decimal digits. In python this is easy, you can use bin and dec functions to convert them. <interactive demo: you and your partner think of a number from 100-1000 in mind, tell the partner in binary and do you get the number correct in decimal?

More interestingly even decimals can be represented by this method. Notice 2^-1 -2 etc is just 0.5 0.25 0.125, so we could proximate any real number by the same approach. 

For example pi up to 5 digits in binary could be written as….. we could do this almost forever, but the question is where to define the first digit?

A better solution used more frequently in computers is the floating point numbers, represented by several parts. <complete the description of ieeee solution . Esp floating point single and double precision

in python the ease of writing is usually eliminated by not reworked precision to appear. 
you can check the default binary rep of your number. In older programs like c /c++ the type of numbers that does operation must match like 1/2=0 

python has some really strong numerical packages most prominently numpy. Numpy handles precision transiently like any py program but also easy to explicitly see. 

<demo fr numpy with explicit float32or 64)

Numpy shows something from float8 to float64 what’s the last digit they agree?

Numpy shows a long random walk simulation float 8!to float64 how much error can accumulate?

These hopefully helps you to remember rule of thumb is double precision to this number and single precision to that number.

How many steps will single precision handle?
what aBout double precision?

Closing question:

Float point may not be the only thing that matters, in many systems using floating point (even double precision) is just not enough. For example, we know the smallest decimal number we could get is around <2^-1074>, which is approximated 4.94x10^-324. But why does ingeneral conditions we say double precision only guarantees 15-17 decimals? (from this question you could understand why does the name float come out)
