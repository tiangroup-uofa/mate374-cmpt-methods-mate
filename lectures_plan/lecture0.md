# Lecture 0: Introduction to Computational Methods and Modeling

## Learning outcomes:

- Get familiarize with the course outlines
- Scope and topics of this course
- Key dates, deadline of the course


## Land acknowledgement

The university of alberta acknoledges that we're teaching on the historical lands of the indigenous peopl <please complete this section accordingly to my MATE664 and CHE374 lecture notes>

## Opening words: numerical thinking for materials engineers

### Objectives

Welcome to the fall semester course of MATE 374, a 300-level UG-level
course aimed at providing the fundamentals of numerical and simulation
skills for a material engineering studens. <may need to enrich, see the examples from old mate374 course>

Before jumping into the topics, we should talk about the the objectives. Materials engineering typically involve 3 kinds of "modelling" as we can recall

- mathematical modeling: <meaning please expand, meaning we exttract the formal mathematical relations  / governing equations of complex material system. some people also say these are physical modeling>
- numerical modeling / methods: <describing the approach where the abstractive math methods are represented using computer programs and codes, that we can expand the math to larger systems, subject to the limitaion and precision of computer systems>
- computational materials simulation: <from the instructor's opinion, this is what distinguished MATE from all other engineering subfields. there are dedicated computer programs that implements **complex mathematical equations** for **complex and large-scale numbers of atoms** to calculate / predict properties of materials, without the prerequisite of experimental values, and can be used to guide actual experimental design>

A fun relation between the 3 paradigms can be seen in a figure shown below. <a phd-comic-like illustration>

You and your friends from chemical and materials engineering (CME) may have taken or heard of another fantasctic course CHE 374 (<add prof name, link>). So why do we learn the same thing again?

The instructor realizes that there are many powerful MATE-specific concepts (e.g. materials phase diagram construction and analysis, continuum diffusion modeling, molecular to atomistic simulations, etc.) may worth separate attention from a general purpose numerical course. In fact, during the semester we will see the emphasis is spanned over both regimes of numerical modeling and computational simulations, with the underlying mathematical modeling as the main thread.

### Main skills we except

With the fast evolution of numerical modeling software and packages in
the past 20 years, the field of computational methods have seen huge
changes. Most importantly, the pre-matlab era (<I believe before 1990s>
where the numerical packages are written by languages like FORTRAN / C/C++ <footnote: FORTRAN is still one of the instructor's favourite languages>), to the Matlab (a commercial software started from <which year? by which company?>) dominated regimes for engineering and science teachings (<citations?>), to the rise of open-source, FAIR numerical programming paradigm (most notably python with its <which adj to use? versatile, or ground-breaking?> numpy/scipy/scikit-learn stack, and others like R <stats> / Julia <physics>, <say the rise from late 2000 onwards>). Recently, the rise of text/code-completion softwares (we tend to avoid use the word model here, because it may be confused by the meaning of model in this course. <they are colloquially known as the "large-language models">) made the barrier of writing numerical / software packages less and less daunting to entry- and senior level students and researchers alike.

So, a natural question is: what's the scope of this course and what skills do I expect you to gain after completing the course, even if you will never actually engage in scientific programming in rest life? I think the idea is clear:

- This is not merely a "python course": python as of 2026 is my favourite versatile open source scientific programming, which is coupled with much more mature other sofrware ecosystem.
- This is not merely a "learn to code" course: as will see in the lecture websites, we will use the cutting-edge python-in-a-browser (<footnote: formally the pyodide on webassembly>) so you can try the codes virtually anywhere, on most compatible devices
- I DO WANT TO show you that, however combining with the above points, MY PHILOSOPHIE is: there are no mathematical equations that cannot be written down to computer programs, and the easiest way to showcase that is via a python interactive demo that you can play with + change codes

So, when skimming out all the essentially coding knowledges, what I want you all to realize is we know how to **analyze** the choice of model, **optimize** the parameter, **verify** the results, **critisize** the error. These to me will likely be more important than remembering the syntax of the specific programming language.

### Getting to know us

I started in UofA in 2025 as an assistant prof, I;m both affiliated with uofa CME and a fellow at Amii. My main research is on adapting AI/ML for accelerating materials simulation and discovery. I'm thrilled to teach the younger generation some useful skills <well last part is awkward please revise>

TA: Prince Ezeano (M.Sc. in my group, started 2025)

## Course outline
### Course schedule

main class: MWF 12:00 - 12:50 <NRW which classroom?>
seminar (TA-guided): F <which classroom>

Office hour: Monday afternoon 13:30 - 14:15, DICE 12-245. No appointment needed

TA: Prince ezeano <email>

### Textbook and resources

Please refer to the course website as the main ground truth for the MATE374 course. There are however numerous online courses and resources however the instructor strongly **recommend** to check some other resources, including

- Kuisalaas: numerical python 3 <please help complete, add the accessible url based on uofa eproxy>
- Sullivan online course <please add link>
- William Callister Jr materials science and engineering an introduction (for engineering problems, available at library)


### Theme-based modules

The MATE374 will try a slightly different approach than conventional "numerical" or "computational" course, that we wanted to split the course into individual themed "units" instead of spearate lectures. The topics will be grouped into following

<this will eventually be a table, I'll just through the words>

unit name | what problems | num / compt methods we focus on

1. Orientataion & numerical thinking | comparison of models / error/ precision and limits | finite precision / error analysis / parameter blahblah
2. How do we find answer? |  find equilibrium of materials | root finding and simple optimization
3. How do we handle larger systems? | atomistic systems / equilibrium of positions | systems of linear algebra / matrix operations
4. How do we handle data? | Getting trend from historical data | interpolation / regression / differential and integration and simple machine learning
6. How do materials systems evolve over time? | Simple evolution | ODE
7. Material evolution over time and space | diffusion in materials | PDE intro
8. Simulating materials at different scales | demonstration of materials simulation from continuum / monte carlo --> simple MD --> intro to atomistic & ML

### Course workload and breakdown

The course will have 2 types of assessments: assignments and group project

there will be ONE (1) midterm exam (time TBD, in the week of Oct 28) and ONE (1) final exam

breakdown
assignment: 20%
group project: 20% (same score for all group members)
midterm exam: 20% (50 min)
final exam: 40% (2 hr)

Assignment grading: there will FIVE (5) assignments released on canvas, each given about 2 weeks for completion. late panelties: 25% points every 24 hours after deadline. Missing one assignment due to other issues? No problem, we will take 4 out of the 5 highest assignments. **In case of emergency** please send to me and TA no less than 24h before the initial deadline and will be reviewed case-by-case

Projects: will be released close to midterm. students are allowed to work in group of approximated 5 on a specific materials simulation problem. Final result and report will be handout.

Format of midterm and final exams: open book tests (can use laptop /
tablets to look for course materials), completion of exam will need to complete and run python codes within the time limits. Internet access to course materials, python nodebooks, and python API resources are allowed without limitation, but no access to generative AI during exam permitted. (instructor / proctor will review).

### Academic integrity and usage of generative AI

As an computational course which has significant components with code generation, I strongly recommend students at least try out some generative AI assisted coding writing. similar to stated above, using AI to help with coding writing during assessment and final project is allowed, but the students must demonstrate there own understanding of the codes and solutions in assignement hand in and reports.

When stating the usage of AI-assisted code generation, please comment on your assignment / report the langauge model / tool you used.

instructors recommendation: we have made a course-custom Gemini AI bot <link https://gemini.google.com/gem/7d47e82eecb2> that you could access using UofA's credentials. No input / materials will be used for training / improving the model and the AI tool was used to help your coding tasks during this course.

### Other misc topic

- **accomodation**: please contact the instructor and TA in advance (i.e. at least 2 weeks before accomodated exam) if you need accomodation (even if you don't need instructor's approval) (we wanted to make sure the materials during accomodation exams cna be correctly accessed)
- Per UofA regulation, there will be no deferred mid term exams. If you're sick during the midterm exam, please send instructor and TA information ASAP. Moving weights form midterm to final will only be reviewed case by case
- the final letter grade of a engineering 300-level course will be subject to the instructor's <discretion? not sure the wording>
- Formula sheet? There will be **no** official formula sheets during mid and final exams as they are intended to be open book However we strongly recommend to write down your collections of notes / methods for better understanding.
- where to get support <add a few links>













