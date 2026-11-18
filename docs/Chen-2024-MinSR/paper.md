Article

https://doi.org/10.1038/s41567-024-02566-1

# Empowering deep neural quantum states through efficient optimization

Received: 20 March 2023

Accepted: 28 May 2024

Published online: 1 July 2024

Check for updates

Ao Chen  & Markus Heyl

Computing the ground state of interacting quantum matter is a long-standing challenge, especially for complex two-dimensional systems. Recent developments have highlighted the potential of neural quantum states to solve the quantum many-body problem by encoding the many-body wavefunction into artificial neural networks. However, this method has faced the critical limitation that existing optimization algorithms are not suitable for training modern large-scale deep network architectures. Here, we introduce a minimum-step stochastic-reconfiguration optimization algorithm, which allows us to train deep neural quantum states with up to 10<sup>6</sup> parameters. We demonstrate our method for paradigmatic frustrated spin-1/2 models on square and triangular lattices, for which our trained deep networks approach machine precision and yield improved variational energies compared to existing results. Equipped with our optimization algorithm, we find numerical evidence for gapless quantum-spin-liquid phases in the considered models, an open question to date. We present a method that captures the emergent complexity in quantum many-body problems through the expressive power of large-scale artificial neural networks.

It has been an ever-persisting quest in condensed-matter and quantum many-body physics to capture the essence of quantum many-body systems that is covered behind their exponential complexity. Although many numerical methods have been developed to access the quantum many-body problem with strong interactions, it still remains an extraordinary challenge to obtain accurate ground-state solutions, especially for complex and large two-dimensional systems. The respective challenges depend on the method utilized, such as the ‘curse of dimensionality’ in exact diagonalization<sup>1</sup>, the notorious sign problem<sup>2</sup> in quantum Monte Carlo approaches<sup>3</sup> or the growth of entanglement and matrix contraction complexity in tensor network methods<sup>4</sup>. One of the paradigmatic instances of such complex two-dimensional quantum matter is the putative quantum-spin-liquid (QSL) phase in frustrated magnets<sup>5</sup>. Although a large variety of different numerical methods have been applied, the nature of many of the presumed QSLs still remains debated, such as the prototypical frustrated Heisenberg J –J magnets on square<sup>6–12</sup> or triangular lattices<sup>13–22</sup>.

Recently, neural quantum states (NQSs) have been introduced as a promising alternative for solving the quantum many-body problem by means of artificial neural networks<sup>23</sup>. This approach has already seen tremendous progress for QSLs<sup>24–26</sup>. However, this method also faces an outstanding challenge critically limiting its capabilities and its potential to date. Due to the rugged quantum landscape<sup>27</sup> with many saddle points, it is typically necessary to utilize stochastic reconfiguration (SR)<sup>28</sup> in the optimization. SR is a quantum generalization of natural gradient descent<sup>29</sup> and has a $\mathcal { O } ( N _ { \mathfrak { p } } ^ { 3 } )$ complexity for a network with $N _ { \mathfrak { p } }$ parameters, which impedes the training of deep networks. Consequently, the current applications of NQS mainly focus on shallow networks, such as a restricted Boltzmann machine $( \mathsf { R B M } ) ^ { 2 3 , 3 0 }$ or shallow convolutional neural networks $( \mathbf { C N N S } ) ^ { 2 5 , 3 1 }$ with no more than ten layers and around $1 0 ^ { 3 }$ parameters. Many efforts have been made to overcome the optimization difficulty in deep NQS based on either iterative solvers<sup>23</sup>, approximate optimizers<sup>32–36</sup> or large-scale supercomputers<sup>37,38</sup>. However, the cost of SR still represents the key limitation in increasing the network size and, thereby, fully materializing the exceptional power of artificial neural networks for outstanding physics problems.

In this work, we introduce an alternative training algorithm for NQS, which we term the minimum-step stochastic reconfiguration (MinSR). We show that the optimization cost in MinSR is reduced massively while it remains as accurate as SR. Concretely, the training cost of MinSR is only linear in $N _ { \mathfrak { p } } ,$ which represents an enormous acceleration compared to SR. This, in turn, allows us to push the NQS towards the deep era by training deep networks with up to 64 layers and 10<sup>6</sup> parameters. We apply our resulting algorithm to paradigmatic two-dimensional quantum spin systems, such as the spin-1/2 Heisenberg J – $- J _ { 2 }$ model, both to demonstrate the resulting accuracies for large system sizes beyond what is achievable with other computational methods and to address an outstanding question relating to the gaps in the model’s QSL phases.

## Results

## Minimum-step stochastic reconfiguration

In the NQS approach, a neural network is utilized to encode and compress the many-body wavefunction. In a system with N spin-1/2 degrees of freedom, the Hilbert space can be spanned by the S spin configuration basis $\vert \sigma \rangle = \vert \sigma _ { 1 } , \dots , \sigma _ { N } \rangle$ with $\sigma _ { i } = \uparrow \thinspace \mathbf { o r } \downarrow$ . An NQS with parameters θ maps every σ at the input to a wavefunction component $\psi _ { \theta , o }$ at the output<sup>23</sup>, as shown in Fig. 1a. The full quantum state is then given by the superposition $\begin{array} { r } { | \varPsi _ { \theta } \rangle = \sum _ { \sigma } \psi _ { \theta , \sigma } | \sigma \rangle } \end{array}$ . When searching for ground states based on a variational Monte Carlo method (VMC), θ is optimized to minimize the variational energy $E _ { \theta } =  \varPsi _ { \theta } | \mathcal { H } | \varPsi _ { \theta }  /  \varPsi _ { \theta } | \varPsi _ { \theta } $

The standard numerical approach for finding the minimal variational energy for NQS is SR. This is done by approximately implementing imaginary-time evolution. Thus, as the training progresses, the contributions from eigenstates with higher energies are systematically reduced, thereby pushing the state towards the ground state step by step. In every training step, this requires minimizing the quantum distance d between the new variational state $| \psi _ { \theta + \delta \theta } \rangle$ and the exact imaginary-time evolved state $\mathbf { e } ^ { - \mathcal { H } \delta \tau } | \psi _ { \theta } \rangle$ , where δτ is the imaginary-time interval.

As proven in the Supplementary Information, the quantum distance d can be estimated for a group of samples σ with $P _ { \sigma } \propto | \psi _ { \sigma } | ^ { 2 }$ as $\begin{array} { r } { d ^ { 2 } = \sum _ { \sigma } \bigl | \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } \bigr | ^ { 2 } } \end{array}$ where ∑ is performed on spin configurations in samples. We adopt the following notation: $\overline { { O } } _ { \sigma k } = ( O _ { \sigma k } - \langle O _ { \sigma k } \rangle ) / \sqrt { N _ { s } }$ with $O _ { \sigma k } = { \frac { 1 } { \psi _ { \sigma } } } { \frac { \partial \psi _ { \sigma } } { \partial \theta _ { k } } }$ , and $\overline { { \epsilon } } _ { \sigma } = - \delta \tau \big ( E _ { \mathrm { l o c } , \sigma } - \big < E _ { \mathrm { l o c } , \sigma } \big > \big ) / \sqrt { N _ { s } }$ with local energy $\begin{array} { r } { E _ { \mathrm { l o c } , \sigma } = \sum _ { \sigma ^ { \prime } } \frac { \psi _ { \sigma ^ { \prime } } } { \psi _ { \sigma } } H _ { \sigma \sigma ^ { \prime } } , } \end{array}$ where $N _ { s }$ is the number of samples and $\langle . . . \rangle$ represents the mean value over the given set of samples.

Thus, the quantum distance d can be rewritten as $d = | | \overline { { O } } \delta \theta - \overline { { \epsilon } } | |$ if we treat δθ and ϵ as vectors and O as a matrix. As a key consequence, we introduce a new linear equation

$$
\overline { { { O } } } \delta \theta = \overline { { { \epsilon } } } ,\tag{1}
$$

whose least-squares solution minimizes the quantum distance d and leads to the SR equation. Conceptually, one can understand the left-hand side of this equation as the change of the variational state induced by an optimization step of the parameters, and the right-hand side as the change of the exact imaginary-time evolving state. The traditional SR solution minimizing their difference is

$$
{ \delta \theta } = S ^ { - 1 } \overline { { { O } } } ^ { \dagger } \overline { { { \epsilon } } } \quad \mathrm { w i t h } S = \overline { { { O } } } ^ { \dagger } \overline { { { O } } } .\tag{2}
$$

As illustrated in Fig. 1a, the matrix S in equation (2) plays an important role as the quantum metric in $\mathsf { V M C } ^ { 2 9 , 3 9 , 4 0 }$ , which links the variations in the Hilbert space and the parameter space. However, inverting the matrix S, which has $N _ { \mathfrak { p } } \times N _ { \mathfrak { p } }$ elements, has $\mathcal { O } ( N _ { \mathfrak { p } } ^ { 3 } )$ complexity, and this a major difficulty when optimizing deep NQSs with large $N _ { \mathfrak { p } } .$ To reduce the cost of SR, we focus on a specific optimization case of a deep network with a large number of parameters $N _ { \mathfrak { p } }$ but a relatively small amount of batch samples $N _ { s } ,$ , as occurs in most deep learning research. In this case, as shown in Fig. 1b, the rank of the $N _ { \mathfrak { p } } \times N _ { \mathfrak { p } }$ matrix S is at most $N _ { \mathrm { s } } ,$ meaning that S contains much less information than its capacity. As a more efficient way to express the information of the quantum metric, + we introduce the neural tangent kernel ${ \cal T } = \overline { { { O } } } \overline { { { O } } }$ (ref. 41), which has the same non-zero eigenvalues as S but the matrix size reduces from $N _ { \mathfrak { p } } \times N _ { \mathfrak { p } }$ to $N _ { \mathrm { s } } \times N _ { \mathrm { s } }$

As derived in Methods, we propose a new method termed MinSR using T as the compressed matrix,

$$
{ \delta } \theta = \overline { { { O } } } ^ { \dagger } T ^ { - 1 } \overline { { { \epsilon } } } \quad \mathrm { w i t h } \ T = \overline { { { O } } } \overline { { { O } } } ^ { \dagger } ,\tag{3}
$$

which is mathematically equivalent to the traditional SR solution but only has $\mathcal { O } ( N _ { \mathrm { p } } N _ { \mathrm { s } } ^ { 2 } + N _ { \mathrm { s } } ^ { 3 } )$ complexity. For large $N _ { \mathfrak { p } ^ { \jmath } }$ , it provides a tremendous acceleration with a time cost proportional to $N _ { \mathfrak { p } }$ instead of $N _ { \mathfrak { p } } ^ { 3 } .$ Therefore, it can be viewed as a natural reformulation of traditional SR, which is particularly useful in the limit $N _ { \mathfrak { p } } \gg N _ { \mathfrak { s } }$ , as relevant in deep learning situations. For a performance comparison, Extended Data Fig. 1 shows the time cost and accuracy of different optimization methods.

## Benchmark models

To demonstrate the exceptional performance of MinSR, we consider in the following the paradigmatic spin-1/2 Heisenberg $J _ { 1 } \ – J _ { 2 }$ model on a square lattice. This choice serves two purposes. On the one hand, this model serves as a standard benchmark system in various NQS studies and provides a convenient comparison to other state-of-the-art methods. On the other hand, it represents a paradigmatic reference case of QSLs in frustrated magnets, as an outstanding question regarding the nature of the QSL phase is whether it is gapped or gapless. The Hamiltonian of the system is given by

$$
\mathcal { H } = J _ { 1 } \sum _ { \langle i , j \rangle } \pmb { S } _ { i } \cdot \pmb { S } _ { j } + J _ { 2 } \sum _ { \langle \langle i , j \rangle \rangle } \pmb { S } _ { i } \cdot \pmb { S } _ { j } ,\tag{4}
$$

where ${ \pmb S } _ { i } = ( S _ { i } ^ { x } , S _ { i } ^ { y } , S _ { i } ^ { z } )$ with $S _ { i } ^ { x } , S _ { i } ^ { y } , S _ { i } ^ { z }$ spin-1/2 operators at site $i , \langle i , j \rangle$ and $\langle \langle i , j \rangle \rangle$ indicate pairs of nearest-neighbour and next-nearest-neighbour sites, respectively, and $J _ { 1 }$ is chosen to be equal to 1 for simplicity in this work.

We will specifically focus on two points in the parameter space: $J _ { 2 } / J _ { 1 } = 0$ and $J _ { 2 } / J _ { 1 } { = } 1 / 2 . \mathrm { A t } J _ { 2 } / J _ { 1 } { = } 0$ , the Hamiltonian reduces to the non-frustrated Heisenberg model. $\mathbf { A t } J _ { 2 } / J _ { 1 } { = } 1 / 2$ , the $J _ { 1 } { \cdot } J _ { 2 }$ model becomes strongly frustrated close to the maximally frustrated point where the system resides in a QSL phase<sup>24</sup>, which imposes a great challenge for existing numerical methods, including $\mathsf { N Q S } ^ { 3 1 , 4 2 }$ . Two different designs of residual neural networks (ResNet), whose details we describe in Methods, will be employed for variationally learning the ground states of these benchmark models. A direct comparison with exact diagonalization results for the 6 × 6 square lattice can be found in Extended Data Fig. 2, which shows that our network can even approach machine precision on modern GPU and TPU hardware.

For a non-frustrated Heisenberg model of a 10 × 10 square lattice, a deep NQS trained by MinSR provides an unprecedentedly precise result that is better than all existing variational methods, as shown in Fig. 2a. The adopted reference ground-state energy per site is $E _ { \scriptscriptstyle \mathrm { G S } } / N = - 0 . 6 7 1 5 5 2 6 7 ( 5 )$ , as given by a simulation based on a stochastic series expansion<sup>43</sup> performed by ourselves, instead of the commonly used reference E/N = −0.671549(4) from ref. 44 because our best NQS variational energy E/N = −0.67155260(3) provides even better accuracy compared to this common reference energy. Thanks to the deep network architecture and the efficient MinSR method, the relative error of the variational energy $\epsilon _ { \mathrm { r e l } } { = } ( E - E _ { \mathrm { G S } } ) / | E _ { \mathrm { G S } } |$ drops much faster than for the one-layer RBM as $N _ { \mathfrak { p } }$ increases and finally reaches a level of $1 0 ^ { - 7 }$ , greatly outperforming existing results.

![](images/fig-01-minsr.jpg)  
Fig. 1 | Illustration of NQS and MinSR. a, In the NQS approach, an artificial neural network is used to represent a quantum many-body state. A change of the network parameters for the NQS leads to a new quantum state, whose distance to the previous NQS is given by the quantum metric $\boldsymbol { S } \in \mathbb { C } ^ { N _ { \mathsf { p } } \times N _ { \mathsf { p } } }$ , where $N _ { \mathfrak { p } }$ is the number of variational parameters. b, The quantum metric ${ \cal { S } } = \overline { { { \cal { O } } } } ^ { \dagger } \overline { { { \cal { O } } } }$ can be decomposed into a smaller matrix $\overline { { O } } \in \mathbb { C } ^ { N _ { \mathsf { S } } \times N _ { \mathsf { P } } }$ with $N _ { \mathfrak { s } } \ll N _ { \mathfrak { p } }$ the number of  
Monte Carlo samples. The optimization of an NQS involves the inversion of the quantum metric S, which is equivalent to determining its non-zero eigenvalues λ with $\dot { \iota } { = } 1 , { \ldots } , N _ { s } .$ . In MinSR, a neural tangent kernel $T = \overline { { O } } \overline { { O } } ^ { \dag } \in \mathbb { C } ^ { N _ { 5 } \times N _ { 5 } }$ is introduced with identical eigenvalues λ and, therefore, the essential information of S.

To attain the next level of complexity, we will now focus on the frustrated $J _ { 1 } { - } J _ { 2 }$ model, whose accurate ground-state solution has remained a key challenge for all available computational approaches. Figure 2b shows that, for a 10 × 10 square lattice, our method based on MinSR allows us to reach ground-state energies below what is possible with any other numerical scheme so far. In this context, the Marshall sign rule (MSR) limit shows the energy one can obtain without considering any frustration. As shown in the figure, the use of deep NQS becomes absolutely crucial as the shallow CNN is not guaranteed to beat the MSR limit. Most importantly, the variational energy we obtained was reduced upon increasing the network size for both networks trained by MinSR. We finally trained unprecedentedly large networks with 64 convolutional layers in ResNet1 and more than one million parameters in ResNet2, to attain the best variational energy $E / N { = } { - } 0 . 4 9 7 6 9 2 1 ( 4 )$ , which outperforms all existing numerical results. The extraordinary variational outcomes allow us to accurately estimate the ground-state energy $E _ { \scriptscriptstyle \mathrm { G S } } / N = - 0 . 4 9 7 7 1 5 ( 9 )$ by zero-variance extrapolation, as described in Methods. Compared with the previous best result<sup>24</sup>, $\epsilon _ { \mathrm { r e l } }$ in our biggest network is around 4 times lower, suggesting that our deep NQS result is substantially more accurate. From this, we conclude that the deep NQS trained by MinSR is superior even in the frustrated case, which was argued to be challenging for NQS on a general level<sup>45</sup>. The variational energies of different methods in this prototypical model are summarized in Extended Data Table 1.

Finally, we aim to provide evidence that our approach still exhibits advantageous performance compared to other computational methods upon further increasing the system size. Figure 2c presents the variational energy obtained for a 16 × 16 square lattice and compares the results with existing results in the literature. One can clearly see that our approach yields the best variational energy $E / N { = } { - } 0 . 4 9 6 7 1 6 3 ( 8 )$ for the frustrated $J _ { 1 } { \cdot } J _ { 2 }$ model on such a large lattice. Compared with the best existing variational result given in ref. 37, $\epsilon _ { \mathrm { r e l } }$ in this work is stil $2 . 5 \times 1 0 ^ { - }$ <sup>4</sup> lower. In summary, the deep NQS trained by MinSR provides

c

![](images/fig-02a-heisenberg-10x10.jpg)

![](images/fig-02b-j1j2-10x10.jpg)  
Fig. 2 | Relative error of the variational energy $\mathbf { \sigma } ^ { \prime } \epsilon _ { \mathrm { r e l } } = \mathbf { \varepsilon } ( { \pmb { E } } - { \pmb { E } } _ { \mathrm { c s } } ) / \vert { \pmb { E } } _ { \mathrm { c s } } \vert$ for a square lattice, wher $: { \pmb { { \cal E } } } _ { { \bf G S } }$ is the exact ground-state energy estimated by stochastic series expansion in the non-frustrated case and zero-variance extrapolation in the frustrated case. a, Non-frustrated 10 × 10 Heisenberg model. The variational energies obtained in this work by using a deep ResNet trained with MinSR are compared to previous results in the literature including an RBM<sup>23</sup>, shallow $\mathsf { C N N } ^ { 3 1 }$ and RBM with a Lanczos step $( \mathsf { R B M + L S } ) ^ { 3 8 }$ . As no tensor network (TN) data are available for the periodic boundary condition, the best result with an open boundary condition is included as a dashed line<sup>51</sup>. b, Frustrated $1 0 \times 1 0 J _ { 1 } J _ { 2 }$

![](images/fig-03-triplet-gap.jpg)  
Fig. 3 | Energy gap Δ between the ground state with total spin S = 0 and the excited state with S = 1 as a function of inverse linear length 1/L at the maximally frustrated point. The inset includes the behaviour of the rescaled gap Δ × L versus 1/L.

results for large frustrated models that are not only on a par with other state-of-the-art methods but can substantially outperform them.

## Energy gaps of a QSL

Although so far we have focused on demonstrating the exceptional performance of the MinSR method, we now take the next step by addressing an outstanding physical question regarding the $I _ { 1 } \mathbf { - } J _ { 2 }$ Heisenberg model considered. Concretely, we utilize the combination of the deep NQS and MinSR to study the gaps for two famous QSL candidates in the $J _ { 1 } \ – J _ { 2 }$ model on a square lattice and on a triangular lattice. In these systems, several works in the literature<sup>6–22</sup> have shown the existence of QSL phases, although the energy gaps in the thermodynamic limit, especially for the triangular lattice, still remain debated. Figure 3 present an extrapolation of the energy gaps between states with total spin ${ \cal S } = 0$ and $S { = } 1 \mathsf { t o }$ the thermodynamic limit within the most frustrated regime in which QSL candidates reside. As explained in Extended Data Figs. 3 and 4, the energy is estimated by NQS trained by MinSR with a Lanczos step and zero-variance extrapolation to increase accuracy. In the Supplementary Information, we provide the spin and dimer structure factors to support the existence of a QSL phase on the triangular lattice and compare gap estimates with and without zero-variance extrapolation.

![](images/fig-02c-j1j2-16x16.jpg)  
model $\mathbf { a t } J _ { 2 } / J _ { 1 } = 0 . 5 .$ The results obtained in this work with MinSR for two designs of ResNet are compared to previous results in the literature for a shallow CNN<sup>31</sup>, $_ { \mathsf { R B M + L S } } 3 8$ , group convolutional neural network $( \mathbf { G C N N } ) ^ { 2 6 }$ and medium CNN<sup>37</sup>. Further results from methods other than NQS are included as dashed lines, such as a tensor network<sup>9</sup>, the Gutzwiller wavefunction with two Lanczos steps $( \mathbf { G } \mathbf { W } \mathbf { F } + 2 \mathbf { L } \mathbf { S } ) ^ { 8 }$ , and a combination of the pair product state and RBM (PP+RBM)<sup>24</sup>. As a further reference, the so-called MSR limit is included. This was obtained from an NQS trained for a wavefunction where the sign structure was not learned but rather fixed by the MSR. c, Frustrated $1 6 \times 1 6 J _ { 1 } – J _ { 2 }$ model $\mathbf { a t } J _ { 2 } / J _ { 1 } { = } 0 . 5$

On the square lattice, the gaps are measured in the total spin $S = 1$ sector and momentum k = (π, π) (M-point) at the most frustrated point $J _ { 2 } / J _ { 1 } = 0 .$ 5 for different system sizes, including $\left. 5 \times 6 , 8 \times 8 , 1 0 \times 1 0 , 1 2 \times 1 2 \right.$ 16 × 16 and $2 0 \times 2 0$ . As shown by the small fitting error in Fig. 3 with $\Delta = a + b / L + c / L ^ { 2 }$ , the vanishing gap $\pmb { \mathscr { \Delta } } = 0 . 0 0 ( 3 )$ in the thermodynamic limit provides an unprecedented precision and is so far the most accurate extrapolation at this most frustrated point. In addition to the direct extrapolation of the energy gap Δ, we support our finding of a vanishing gap in the inset of Fig. 3, which display $\varDelta \times \var L$ as a function of 1/L (ref. 24). Although a finite gap would imply a divergent $\pmb { \varDelta } \times \pmb { { L } } _ { \ast }$ , we observe a constant value, further corroborating our conclusion of a gapless phase in the thermodynamic limit. Combined with the large lattice sizes used, this result shows strong evidence of gapless QSLs as suggested by refs. 10–12,24 in contrast to the conclusion of the gapped QSLs in ref. 6.

The triangular J $\phantom { } _ { 1 } \overline { { - J _ { 2 } } }$ model has even stronger frustration compared to the square one, leading to larger variational errors in different meth ods and more controversy regarding the nature of the QSLs. To target the QSLs in this model, we also studied the most frustrated point at $J _ { 2 } / J _ { 1 } { = } 0 . 1 2 5$ . The gaps were measured for the S = 1 and $k = ( 4 \pi / 3 , 0 )$ state on lattices $6 \times 6 , 6 \times 9 , 1 2 \times 1 2$ and 18 × 18 for the triangular lattice. Due to the larger variational error on the triangular lattice compared to the square case, a linear fitting $\pmb { \varDelta } = \ b { a } + \ b { b } / L$ was utilized instead of the quadratic one to prevent overfitting. For a lattice with unequal extents $L _ { x }$

and $L _ { y }$ in different dimensions, L is defined as ${ \sqrt { L _ { x } L _ { y } } } .$ . Our data matches well with the linear relation $\varDelta \propto 1 / L$ as expected for Dirac spin liquids, and the vanishing gap at the thermodynamic limit is $\pmb { \varDelta } = - 0 . 0 5 ( 6 )$ Furthermore, we also performed an extrapolation of $\varDelta \times \var L$ (inset of

Fig. 3). We found a finite Δ × L upon increasing the system size L, indicating a vanishing gap in the thermodynamic limit. We take these results as strong numerical evidence suggesting the existence of a gapless QSL as also indicated in refs. 13,16,20,22 instead of a gapped QSL in refs. 14,15,21. Consequently, these numerical results demonstrate the exceptional computational power of the MinSR method applied to NQS wavefunctions, especially for the challenging regime of frustrated quantum magnets in two dimensions.

## Discussion

To date, there have been tremendous efforts in solving quantum many-body problems in two major directions, studying the simplified Hilbert space given by specific physical backgrounds on classical computers and traversing the full Hilbert space on quantum computers. In this work, we present another promising approach that is supported by deep NQSs. This method allows us to approximate the complexity of quantum many-body problems through the emergent expressive power of large-scale neural networks.

For the future, we envision promising research directions, for instance, studying fermionic systems including the celebrated Hubbard model<sup>46,47</sup> or ab initio quantum chemistry<sup>48</sup>, in which the traditional methods have limited accuracy, especially in the strongly interacting regime. Moreover, it is key to point out that the MinSR method is not at all restricted to NQS. As a general optimization method in VMC, it can also be applied to other variational wavefunctions, like tensor networks, so that a more complex ansatz can be introduced in these conventional methods to enhance the expressivity. It will also be of great importance to exploit the expressive power of large-scale variational wavefunctions through a suitable design that would lower the computational cost and increase the accuracy.

We can further envision the application of MinSR beyond the scope of physics for general machine learning tasks, if a suitable space for optimization like the Hilbert space in physics can be defined for which we can construct an equation like equation (1). In reinforcement learning tasks, for instance, obtaining gradients from the action in the environment is usually the most time-consuming part of the training, so a MinSR-like natural policy gradient<sup>49</sup> can provide more accurate optimization directions without substantially increased time cost and greatly improve the training efficiency, even for very deep neural networks. Recently, a method inspired by MinSR has already found applications in general machine learning tasks<sup>50</sup>.

## Online content

Any methods, additional references, Nature Portfolio reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41567-024-02566-1.

## References

1. Lin, H., Gubernatis, J., Gould, H. & Tobochnik, J. Exact diagonalization methods for quantum systems. Comput. Phys. 7, 400 (1993).

2. Troyer, M. & Wiese, U.-J. Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations. Phys. Rev. Lett. 94, 170201 (2005).

3. Ceperley, D. & Alder, B. Quantum Monte Carlo. Science 231, 555 (1986).

4. Schollwöck, U. The density-matrix renormalization group in the age of matrix product states. Ann. Phys. 326, 96 (2011).

5. Balents, L. Spin liquids in frustrated magnets. Nature 464, 199 (2010).

6. Jiang, H.-C., Yao, H. & Balents, L. Spin liquid ground state of the spin- <sup>1</sup> square J<sub>1</sub>-J<sub>2</sub> Heisenberg model. Phys. Rev. B 86, 024424 2 (2012).

7. Wang, L., Poilblanc, D., Gu, Z.-C., Wen, X.-G. & Verstraete, F. Constructing a gapless spin-liquid state for the spin-1/2 J<sub>1</sub>–J<sub>2</sub> Heisenberg model on a square lattice. Phys. Rev. Lett. 111, 037202 (2013).

8. Hu, W.-J., Becca, F., Parola, A. & Sorella, S. Direct evidence for a gapless Z<sub>2</sub> spin liquid by frustrating Néel antiferromagnetism. Phys. Rev. B 88, 060402 (2013).

9. Gong, S.-S., Zhu, W., Sheng, D. N., Motrunich, O. I. & Fisher, M. P. A. Plaquette ordered phase and quantum phase diagram in the spin- $- { \frac { 1 } { 2 } } J _ { 1 } - J _ { 2 }$ square Heisenberg model. Phys. Rev. Lett. 113, 027201 (2014).

10. Wang, L. & Sandvik, A. W. Critical level crossings and gapless spin liquid in the square-lattice spin-1/2 J<sub>1</sub>–J<sub>2</sub> Heisenberg antiferromagnet. Phys. Rev. Lett. 121, 107202 (2018).

11. Liu, W.-Y. et al. Gapless spin liquid ground state of the spin- $- { \frac { 1 } { \gamma } } J _ { 1 } - J _ { 2 }$ Heisenberg model on square lattices. Phys. Rev. B 98, 241109 (2018).

12. Ferrari, F. & Becca, F. Gapless spin liquid and valence-bond solid in the J<sub>1</sub>–J<sub>2</sub> Heisenberg model on the square lattice: insights from singlet and triplet excitations. Phys. Rev. B 102, 014417 (2020).

13. Kaneko, R., Morita, S. & Imada, M. Gapless spin-liquid phase in an extended spin 1/2 triangular Heisenberg model. J. Phys. Soc. Jpn 83, 093707 (2014).

14. Zhu, Z. & White, S. R. Spin liquid phase of the $\begin{array} { r } { S = \frac 1 2 J _ { 1 } - J _ { 2 } } \end{array}$ Heisenberg model on the triangular lattice. Phys. Rev. B 92, 041105 (2015).

15. Hu, W.-J., Gong, S.-S., Zhu, W. & Sheng, D. N. Competing spin-liquid states in the spin -<sup>1</sup> Heisenberg model on the triangular 2 lattice. Phys. Rev. B 92, 140403 (2015).

16. Iqbal, Y., Hu, W.-J., Thomale, R., Poilblanc, D. & Becca, F. Spin liquid nature in the Heisenberg J<sub>1</sub>–J<sub>2</sub> triangular antiferromagnet. Phys. Rev. B 93, 144411 (2016).

17. Saadatmand, S. N. & McCulloch, I. P. Symmetry fractionalization in the topological phase of the spin- $\frac { 1 } { \gamma } J _ { 1 } - J _ { 2 }$ triangular Heisenberg model. Phys. Rev. B 94, 121111 (2016).

18. Wietek, A. & Läuchli, A. M. Chiral spin liquid and quantum criticality in extended s = <sup>1</sup> Heisenberg models on the triangular lattice. Phys. Rev. B 95, 035141 (2017).

19. Gong, S.-S., Zhu, W., Zhu, J.-X., Sheng, D. N. & Yang, K. Global phase diagram and quantum spin liquids in a spin - <sup>1</sup> triangular antiferromagnet. Phys. Rev. B 96, 075116 (2017).

20. Hu, S., Zhu, W., Eggert, S. & He, Y.-C. Dirac spin liquid on the spin-1/2 triangular Heisenberg antiferromagnet. Phys. Rev. Lett. 123, 207203 (2019).

21. Jiang, Y.-F. & Jiang, H.-C. Nature of quantum spin liquids of the s = <sup>1</sup> Heisenberg antiferromagnet on the triangular 2 lattice: a parallel DMRG study. Phys. Rev. B 107, L140411 (2023).

22. Sherman, N. E., Dupont, M. & Moore, J. E. Spectral function of the J –J Heisenberg model on the triangular lattice. Phys. Rev. B 107, 165146 (2023).

23. Carleo, G. & Troyer, M. Solving the quantum many-body problem with artificial neural networks. Science 355, 602 (2017).

24. Nomura, Y. & Imada, M. Dirac-type nodal spin liquid revealed by refined quantum many-body solver using neural-network wave function, correlation ratio, and level spectroscopy. Phys. Rev. X 11, 031034 (2021).

25. Astrakhantsev, N. et al. Broken-symmetry ground states of the Heisenberg model on the pyrochlore lattice. Phys. Rev. X 11, 041021 (2021).

26. Roth, C., Szabó, A. & MacDonald, A. H. High-accuracy variational Monte Carlo for frustrated magnets with deep neural networks. Phys. Rev. B 108, 054410 (2023).

27. Bukov, M., Schmitt, M. & Dupont, M. Learning the ground state of a non-stoquastic quantum Hamiltonian in a rugged neural network landscape. SciPost Phys. 10, 147 (2021).

28. Sorella, S. Green function Monte Carlo with stochastic reconfiguration. Phys. Rev. Lett. 80, 4558 (1998).

29. Stokes, J., Izaac, J., Killoran, N. & Carleo, G. Quantum natural gradient. Quantum 4, 269 (2020).

30. Nomura, Y. Helping restricted Boltzmann machines with quantum-state representation by restoring symmetry. J. Phys.: Condens. Matter 33, 174003 (2021).

31. Choo, K., Neupert, T. & Carleo, G. Two-dimensional frustrated J<sub>1</sub>–J<sub>2</sub> model studied with neural network quantum states. Phys. Rev. B 100, 125124 (2019).

32. Sharir, O., Levine, Y., Wies, N., Carleo, G. & Shashua, A. Deep autoregressive models for the eficient variational simulation of many-body quantum systems. Phys. Rev. Lett. 124, 020503 (2020).

33. Yang, L. et al. Deep learning-enhanced variational Monte Carlo method for quantum many-body physics. Phys. Rev. Res. 2, 012039(R) (2020).

34. Hibat-Allah, M., Ganahl, M., Hayward, L. E., Melko, R. G. & Carrasquilla, J. Recurrent neural network wave functions. Phys. Rev. Res. 2, 023358 (2020).

35. Inui, K., Kato, Y. & Motome, Y. Determinant-free fermionic wave function using feed-forward neural networks. Phys. Rev. Res. 3, 043126 (2021).

36. Zhang, W., Xu, X., Wu, Z., Balachandran, V. & Poletti, D. Ground state search by local and sequential updates of neural network quantum states. Phys. Rev. B 107, 165149 (2023).

37. Liang, X. et al. Deep learning representations for quantum many-body systems on heterogeneous hardware. Mach. Learn.: Sci. Technol. 4, 015035 (2023).

38. Chen, H., Hendry, D. G., Weinberg, P. E. & Feiguin, A. Systematic improvement of neural network quantum states using Lanczos. In Proc. Advances in Neural Information Processing Systems Vol. 35 (eds Oh, A. H. et al.) 7490–7503 (Curran Associates, 2022).

39. Mazzola, G., Zen, A. & Sorella, S. Finite-temperature electronic simulations without the Born–Oppenheimer constraint. J. Chem. Phys. https://doi.org/10.1063/1.4755992 (2012).

40. Park, C.-Y. & Kastoryano, M. J. Geometry of learning neural quantum states. Phys. Rev. Res. 2, 023232 (2020).

41. Jacot, A., Gabriel, F. & Hongler, C. Neural tangent kernel: convergence and generalization in neural networks. In Proc. Advances in Neural Information Processing Systems Vol. 31 (eds Bengio, S. et al.) 8571–8580 (Curran Associates, 2018).

42. Liang, X. et al. Solving frustrated quantum many-particle models with convolutional neural networks. Phys. Rev. B 98, 104426 (2018).

43. Sandvik, A. W. Stochastic series expansion method with operator-loop update. Phys. Rev. B 59, R14157 (1999).

44. Sandvik, A. W. Finite-size scaling of the ground-state parameters of the two-dimensional Heisenberg model. Phys. Rev. B 56, 11678 (1997).

45. Westerhout, T., Astrakhantsev, N., Tikhonov, K. S., Katsnelson, M. I. & Bagrov, A. A. Generalization properties of neural network approximations to frustrated magnet ground states. Nat. Commun. 11, 1593 (2020).

46. Luo, D. & Clark, B. K. Backflow transformations via neural networks for quantum many-body wave functions. Phys. Rev. Lett. 122, 226401 (2019).

47. Moreno, J. R., Carleo, G., Georges, A. & Stokes, J. Fermionic wave functions from neural-network constrained hidden states. Proc. Natl Acad. Sci. USA 119, e2122059119 (2022).

48. Hermann, J. et al. Ab initio quantum chemistry with neural-network wavefunctions. Nat. Rev. Chem. 7, 692 (2023).

49. Kakade S. M. A natural policy gradient. In Proc. Advances in Neural Information Processing Systems Vol. 14 (eds by Dietterich, T. et al.) 1531–1538 (MIT Press, 2001).

50. Chen, Y., Xie, H. & Wang, H. Eficient numerical algorithm for large-scale damped natural gradient descent. Preprint at https:// arxiv.org/abs/2310.17556 (2023).

51. He, L. et al. Peps++: towards extreme-scale simulations of strongly correlated quantum many-particle models on Sunway Taihulight. IEEE Trans. Parallel Distrib. Syst. 29, 2838 (2018).

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons. org/licenses/by/4.0/.

© The Author(s) 2024

## Methods

## Derivation of the MinSR equation

MinSR was derived based on the observation that equation (1) is underdetermined when $N _ { \mathrm { s } } < N _ { \mathrm { p } } .$ . To obtain a unique δθ solution, we employed the least-squares minimum-norm condition, which is widely used for underdetermined linear equations. To be specific, we chose, among all solutions with minimum residual error ||Oδθ − ϵ|| , the one minimizing the norm of the variational step ∣∣δθ∣∣, which helps to reduce higher-order effects, prevent overfitting and improve stability. We called this method MinSR due to the additional minimum-step condition. In this section, we adopt two different approaches, namely the Lagrangian multiplier method and the pseudo-inverse method, to derive the MinSR formula in equation (3).

Lagrangian multiplier. The MinSR solution can be derived by minimizing the variational step $\textstyle \sum _ { k } | \delta \theta _ { k } | ^ { 2 }$ under the constraint of minimum residual error $\begin{array} { r } { \sum _ { \sigma } \lvert \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } \rvert ^ { 2 } } \end{array}$ . To begin, we assume that the minimum residual error is 0, which can always be achieved by letting $N _ { \mathrm { s } } { < } N _ { \mathrm { p } }$ and assuming a typical situation in VMC that $\overline { { O } } _ { \sigma k }$ values obtained by different samples are linearly independent. This leads to constraints $\begin{array} { r } { \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } = 0 } \end{array}$ for each σ. The Lagrange function is then given by

$$
\mathcal { L } ( \{ \delta \theta _ { k } \} , \{ \alpha _ { \sigma } \} ) = \sum _ { k } | \delta \theta _ { k } | ^ { 2 } - \left[ \sum _ { \sigma } \alpha _ { \sigma } ^ { * } \sum _ { k } ( \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } ) + \mathrm { h . c . } \right] ,\tag{5}
$$

where $\alpha _ { \sigma }$ is the Lagrangian multiplier. Written in matrix form, the Lagrangian function becomes

$$
\mathcal { L } ( \delta \theta , \alpha ) = \delta \theta ^ { \dagger } \delta \theta - \alpha ^ { \dagger } ( \overline { { O } } \delta \theta - \overline { { \epsilon } } ) - ( \delta \theta ^ { \dagger } \overline { { O } } ^ { \dagger } - \overline { { \epsilon } } ^ { \dagger } ) \alpha .\tag{6}
$$

From $\partial \mathcal { L } / \partial ( \delta \theta ^ { \dagger } ) = 0 .$ , one obtains

$$
\delta \theta = \overline { { { O } } } ^ { \dagger } \alpha .\tag{7}
$$

Putting equation (7) back into $\overline { { { O } } } \delta \theta = \overline { { { \epsilon } } } ,$ one can solve α as

$$
\alpha = ( \overline { { { O } } } \overline { { { O } } } ^ { \dagger } ) ^ { - 1 } \overline { { { \epsilon } } } .\tag{8}
$$

Combining equation (8) with equation (7), one obtains the final solution as

$$
{ \delta } \theta = \overline { { O } } ^ { \dagger } ( \overline { { O } } \overline { { O } } ^ { \dagger } ) ^ { - 1 } \overline { { \epsilon } } ,\tag{9}
$$

which is the MinSR formula in equation (3). A similar derivation also applies when O, δθ and ϵ are all real.

In our simulations, the residual error is non-zero, which differs from our previous assumption. This is because the inverse in equation (9) is replaced by a pseudo-inverse with finite truncation to stabilize the solution in the numerical experiments.

Pseudo-inverse. To simplify the notation, we use $A = \overline { { O } } , x = \delta \theta$ and $b = \overline { { { \epsilon } } } .$ . We will prove that for a linear equation Ax = b,

$$
x = A ^ { - 1 } b = { ( A ^ { \dagger } A ) } ^ { - 1 } A ^ { \dagger } b = A ^ { \dagger } { ( A A ^ { \dagger } ) } ^ { - 1 } b\tag{10}
$$

is the least-squares minimum-norm solution, where the matrix inverse is pseudo-inverse.

First, we prove x = A<sup>−1</sup>b is the solution we need. The singular value decomposition of A gives

$$
A = U \Sigma V ^ { \dagger } ,\tag{11}
$$

where U and V are unitary matrices, and Σ is a diagonal matrix with $\sigma _ { i } = \Sigma _ { i i } = 0$ if and only if i > r with r the rank of A. The least-squares solution is given by minimizing

$$
\begin{array} { l } { \displaystyle | | A x - b | | ^ { 2 } = | | U \Sigma V ^ { \dagger } x - b | | ^ { 2 } } \\ { \displaystyle = | | \Sigma x ^ { \prime } - b ^ { \prime } | | ^ { 2 } } \\ { \displaystyle = \sum _ { i = 1 } ^ { r } \left( \sigma _ { i } x _ { i } ^ { \prime } - b _ { i } ^ { \prime } \right) ^ { 2 } + \sum _ { i = r + 1 } ^ { N _ { s } } b _ { i } ^ { \prime 2 } , } \end{array}\tag{12}
$$

where $x ^ { \prime } = V ^ { \dagger } x , \ b ^ { \prime } = U ^ { \dagger } b$ and $N _ { s }$ is the dimension of b, and the second step is because applying a unitary matrix does not change the norm of a vector. Therefore, all the least-squares solutions take the form

$$
x _ { i } ^ { \prime } = { \left\{ \begin{array} { l l } { b _ { i } ^ { \prime } / \sigma _ { i } , } & { i \leq r , } \\ { \quad } \\ { { \mathrm { a n y v a l u e } } , } & { i > r . } \end{array} \right. }\tag{13}
$$

Among all these possible solutions, the one that minimizes $| | \boldsymbol { x } | | = | | \boldsymbol { x ^ { \prime } } | | \mathbf { i s }$

$$
x _ { i } ^ { \prime } = \left\{ { \begin{array} { l l } { b _ { i } ^ { \prime } / \sigma _ { i } , } & { i \leq r , } \\ { } \\ { 0 , } & { i > r . } \end{array} } \right.\tag{14}
$$

With the following definition of a pseudo-inverse

$$
\begin{array} { r } { A ^ { - 1 } = V \Sigma ^ { + } U ^ { \dagger } , } \end{array}
$$

$$
\begin{array} { r } { \Sigma _ { i j } ^ { + } = \delta _ { i j } \times \left\{ \begin{array} { l l } { 1 / \sigma _ { i } , } & { \sigma _ { i } > 0 , } \\ { 0 , } & { \sigma _ { i } = 0 , } \end{array} \right. } \end{array}\tag{15}
$$

we have $x ^ { \prime } = \Sigma ^ { + } b ^ { \prime }$ , so the final solution is

$$
x = V x ^ { \prime } = V \Sigma ^ { + } U ^ { \dagger } b = A ^ { - 1 } b .\tag{16}
$$

Furthermore, we show the following equality

$$
A ^ { - 1 } = { ( A ^ { \dagger } A ) } ^ { - 1 } A ^ { \dagger } = A ^ { \dagger } { ( A A ^ { \dagger } ) } ^ { - 1 } .\tag{17}
$$

With the singular value decomposition of A in equation (11), equation (17) can be directly proved by

$$
\begin{array} { l } { { { ( A ^ { \dagger } A ) } ^ { - 1 } A ^ { \dagger } = { ( V \Sigma U ^ { \dagger } U \Sigma V ^ { \dagger } ) } ^ { - 1 } V \Sigma U ^ { \dagger } } } \\ { { \ \qquad = V ( { \Sigma ^ { + } } ) ^ { 2 } V ^ { \dagger } V \Sigma U ^ { \dagger } } } \\ { { \ \qquad = V \Sigma ^ { + } U ^ { \dagger } } } \\ { { \ \qquad = A ^ { - 1 } , } } \end{array}\tag{18}
$$

and

$$
\begin{array} { l } { { A ^ { \dagger } ( A A ^ { \dagger } ) ^ { - 1 } = V \Sigma U ^ { \dagger } \bigl ( U \Sigma V ^ { \dagger } V \Sigma U ^ { \dagger } \bigr ) ^ { - 1 } } } \\ { { \ } } \\ { { \ } } \\ { { \qquad = V \Sigma U ^ { \dagger } U ( \Sigma ^ { + } ) ^ { 2 } U ^ { \dagger } } } \\ { { \ \qquad = V \Sigma ^ { + } U ^ { \dagger } } } \\ { { \ \qquad = A ^ { - 1 } . } } \end{array}\tag{19}
$$

In the derivation, the shapes of diagonal matrices Σ and $\Sigma ^ { + }$ are not fixed but assumed to match their neighbour matrices to make the matrix multiplication valid.

Equation (17) shows that the SR solution in equation (2) and MinSR solution in equation (3) are both equivalent to the pseudo-inverse solution $\delta \theta = \overline { { O } } ^ { - 1 } \overline { { \epsilon } }$ , which justifies MinSR as a natural alternative to SR when $N _ { \mathrm { s } } { < } N _ { \mathrm { p } } .$

## MinSR solution

Numerical solution. In this section, we focus on how to solve the MinSR equation numerically:

$$
\delta \theta = \overline { { { O } } } ^ { \dagger } T ^ { - 1 } \overline { { { \epsilon } } } .\tag{20}
$$

The whole computation, starting from ${ \cal T } = \overline { { { O } } } \overline { { { O } } }$ , should be executed under double-precision arithmetic to ensure that small eigenvalues are reliable.

Then a suitable pseudo-inverse should be applied to obtain a stable solution. In practice, the Hermitian matrix T is first diagonalized as $T = U D U$ , and the pseudo-inverse is given by

$$
T ^ { - 1 } = U D ^ { + } U ^ { \dagger } ,\tag{21}
$$

where $D ^ { + }$ is the pseudo-inverse of the diagonal matrix $D ,$ numerically given by a cutoff below which the eigenvalues are regarded as 0, that is

$$
\lambda _ { i } ^ { + } = \left\{ \begin{array} { l l } { 1 / \lambda _ { i } , } & { | \lambda _ { i } | \geq r _ { \mathrm { p i n v } } | \lambda _ { \mathrm { m a x } } | + a _ { \mathrm { p i n v } } , } \\ { 0 , } & { | \lambda _ { i } | < r _ { \mathrm { p i n v } } | \lambda _ { \mathrm { m a x } } | + a _ { \mathrm { p i n v } } , } \end{array} \right.\tag{22}
$$

where $\lambda _ { i }$ and $\lambda _ { i } ^ { + }$ are the diagonal elements of D and $D ^ { + } , \lambda _ { \operatorname* { m a x } }$ is the largest value among $\lambda _ { i } ,$ , and $r _ { \mathrm { p i n v } }$ and $a _ { \mathrm { p i n } }$ are the relative and absolute pseudo-inverse cutoffs. In most cases, we choose $r _ { \mathrm { p i n v } } = 1 0 ^ { - 1 2 }$ and $a _ { \mathrm { p i n v } } = 0$ . Furthermore, we modify the aforementioned direct cutoff to a soft one<sup>52</sup>:

$$
\lambda _ { i } ^ { + } = \left[ \lambda _ { i } \left( 1 + \left( \frac { r _ { \mathrm { p i n v } } | \lambda _ { \mathrm { m a x } } | + a _ { \mathrm { p i n v } } } { | \lambda _ { i } | } \right) ^ { 6 } \right) \right] ^ { - 1 }\tag{23}
$$

to avoid abrupt changes when the eigenvalues cross the cutoff during optimization.

Complex neural networks. Our original MinSR formula equation (3) can be applied when the network is real or complex holomorphic. In our ResNet2 architecture, however, the neural network parameters are real but the network outputs can be complex, in which case equation (3) cannot be directly applied. For other non-holomorphic networks, a complex parameter can be taken as two independent real parameters but this problem still occurs. To obtain the MinSR equation in these special cases, notice that the quantum distance d between $| \psi _ { \theta + \delta \theta } \rangle$ and $\mathsf { e } ^ { \mathrm { i } \mathcal { H } \delta \tau } | \psi _ { \theta } \rangle$ can be reformulated as

$$
\begin{array} { r l } & { d ^ { 2 } = | | \overline { { O } } \delta \theta - \overline { { \epsilon } } | | ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad = | | \mathrm { R e } ( \overline { { O } } ) \delta \theta - \mathrm { R e } ( \overline { { \epsilon } } ) | | ^ { 2 } + | | \mathrm { I m } ( \overline { { O } } ) \delta \theta - \mathrm { I m } ( \overline { { \epsilon } } ) | | ^ { 2 } , } \end{array}\tag{24}
$$

assuming $\overline { o }$ and ϵ are complex while δθ is real. By defining

$$
\overline { { { O } } } ^ { \prime } = \left( \begin{array} { c } { { \mathrm { R e } \overline { { { O } } } } } \\ { { \mathrm { I m } \overline { { { O } } } } } \end{array} \right) , \quad \overline { { { \epsilon } } } ^ { \prime } = \left( \begin{array} { c } { { \mathrm { R e } \overline { { { \epsilon } } } } } \\ { { \mathrm { I m } \overline { { { \epsilon } } } } } \end{array} \right) ,\tag{25}
$$

one can rewrite the quantum distance again as $d ^ { 2 } = | | \overline { { O } } ^ { ' } \delta \theta - \overline { { \epsilon } } ^ { ' } | | ^ { 2 }$ with all entities real. The MinSR solution, in this case, is similarly given by

$$
{ \delta } \theta = \overline { { O } } ^ { ' \dagger } T ^ { - 1 } \overline { { \epsilon } } \quad \mathrm { w i t h } \ T ^ { \prime } = \overline { { O } } ^ { ' } \overline { { O } } ^ { ' \dagger } .\tag{26}
$$

Similar arguments can also provide the SR equation in the non-holomorphic case as

$$
\delta \theta = { { S } ^ { \prime - 1 } } F ^ { \prime }\tag{27}
$$

$$
{ \mathrm { w i t h } } S ^ { \prime } = \overline { { O } } ^ { \prime \dagger } \overline { { O } } ^ { \prime } = { \mathrm { R e } } S , F ^ { \prime } = \overline { { O } } ^ { \prime \dagger } \overline { { \epsilon } } ^ { \prime } = { \mathrm { R e } } F ,
$$

where ${ \cal { S } } = \overline { { { \cal { O } } } } ^ { \dagger } \overline { { { \cal { O } } } }$ and $F = \overline { { O } } ^ { \dagger } \bar { \epsilon }$ are the same as for the ordinary SR solution. This solution agrees with the widely used non-holomorphic SR solution<sup>53</sup>.

## Neural quantum states

In this work, we adopt two different designs of ResNets. Several techniques are also applied to reduce the error.

ResNet1. The first architecture, as suggested in ref. 54, has two convolutional layers in each residual block, each given by a layer normalization, a ReLU activation function and a convolutional layer sequentially. All the convolutional layers are real-valued with the same number of channels and kernel size. After the forward pass through all residual blocks, a final activation function $f ( x ) = \cosh x ( x > 0 ) , 2 - \cosh x ( x < 0 )$ is applied, which resembles the cosh�x) activation in RBM but can also give negative outputs so that the whole network is able to express sign structures while still being real-valued. In the non-frustrated case, ∣f(x)∣ is used as the final activation function to make all outputs positive. After the final activation function, the outputs ${ \bf \nabla } _ { \bf i } \psi _ { i }$ are used to compute the wavefunction as $\begin{array} { r } { \psi _ { \sigma } ^ { \mathrm { n e t } } = \prod _ { i } ( v _ { i } / t ) } \end{array}$ , where t is a rescaling factor updated in every training step. t is used to prevent a data overflow after the product.

ResNet2. The second design of ResNet basically follows ref. 26. In this architecture, the residual blocks are the same as ResNet1 but the normalization layers are removed. In the last layer, two different kinds of activations can be applied. For real-valued wavefunctions, we chose f�x) = sinh�x) + 1. For complex-valued wavefunctions, we split all channels in the last layer into two groups and employ $f ( x _ { 1 } , x _ { 2 } ) = \exp ( x _ { 1 } + \mathrm { i } x$ <sub>2</sub>). A rescaling factor t is also inserted in suitable places in f to prevent an overflow.

Finally, a sum is performed to obtain the wavefunction. Considering the possible non-zero momentum q, the wavefunction is given by

$$
\psi _ { \sigma } ^ { \mathrm { n e t } } = \sum _ { i } \mathrm { e } ^ { - \mathrm { i } \mathbf { q } \cdot \mathbf { r } _ { i } } \sum _ { c } \upsilon _ { c , i } ,\tag{28}
$$

where $\upsilon _ { c , \astrosun }$ is the last-layer neuron at channel c and site i, and $r _ { i }$ is the real-space position of site i. This definition ensures that the whole NQS has a momentum q.

In summary, ResNet1 performs better when one applies transfer learning from a small lattice to a larger one, but ResNet2, in general, has better accuracy and stability. Moreover, ResNet2 allows one to implement non-zero momentum, which is key to finding low-lying excited states.

Sign structure. On top of the raw output from the neural network $\psi _ { \sigma } ^ { \mathrm { n e t } }$ the $\mathsf { M S R } ^ { 5 5 }$ is applied to wavefunctions on a square lattice, which serves as the exact sign structure for the non-frustrated Heisenberg model but is still the approximate sign structure in the frustrated region around $J _ { 2 } / J _ { 1 } { \approx } 0 . 5 .$ The sign structure representing the $1 2 0 ^ { \circ }$ magnetic order is also applied for the triangular lattice. Although these sign structures are additional physical inputs for specific models, the generality is not reduced because it has been shown that simple sign structures such as MSR can be exactly solved by an additional sign network<sup>56,57</sup>.

Symmetry. Symmetry plays an important role in improving the accuracy and finding low-lying excited states for $\mathsf { N Q S } ^ { 3 0 , 5 \& }$ . In this work, we apply symmetry on top of the well-trained $\psi _ { \sigma } ^ { \mathrm { n e t } }$ to project variational states onto suitable symmetry sectors. Assuming the system permits a symmetry group of order ∣G∣ represented by operators T with characters $\omega _ { i } ,$ the symmetrized wavefunction is then defined as<sup>30,59</sup>

$$
\psi _ { \sigma } ^ { \mathrm { s y m m } } = \frac { 1 } { | G | } \sum _ { i } \omega _ { i } ^ { - 1 } \psi _ { T _ { i } \sigma } ^ { \mathrm { n e t } } .\tag{29}
$$

With translation symmetry already enforced by the CNN architecture, the remaining symmetries applied by equation (29) are the point group symmetry, which is $C _ { 4 \upsilon }$ for the square lattice and $D _ { 6 }$ for the triangular lattice, and the spin inversion symmetry σ → −σ (refs. 60–64).

## Zero-variance extrapolation

The variational wavefunction provides an inexact estimate of the ground-state energy due to the variational error. Fortunately, in VMC one can compute the energy variance

$$
\sigma ^ { 2 } = \left. \mathcal { H } ^ { 2 } \right. - \left. \mathcal { H } \right. ^ { 2 }\tag{30}
$$

as an estimate of the variational error. Hence, an extrapolation to zero energy variance gives a better estimate of the ground-state energy $r ^ { 6 5 , 6 6 }$ which has been successfully applied to NQS in refs. 30,37. In the following, we adopt the derivation in ref. 66 to show how to perform the extrapolation.

Assuming the normalized variational state |ψ⟩ deviates only slightly from the exact ground state $\left| \psi _ { \mathrm { g } } \right.$ , one can express it as

$$
\left| \psi \right. = \sqrt { 1 - \lambda ^ { 2 } } \left| \psi _ { \mathrm { g } } \right. + \lambda \left| \psi _ { \mathrm { e } } \right. ,\tag{31}
$$

where $| \psi _ { \mathrm { e } } \rangle$ represents the error in the variational state orthogonal to the ground state and λ is a small positive number indicating the error strength. Denoting $E _ { \mathrm { g } } = \left. \psi _ { \mathrm { g } } | \mathcal { H } | \psi _ { \mathrm { g } } \right. , E _ { \mathrm { e } } = \left. \psi _ { \mathrm { e } } | \mathcal { H } | \psi _ { \mathrm { e } } \right.$ and $\left. \mathcal { H } ^ { 2 } \right. _ { \mathrm { e } } = \left. \psi _ { \mathrm { e } } | \mathcal { H } ^ { 2 } | \psi _ { \mathrm { e } } \right.$ , one can express the variational energy as

$$
E = \langle \psi | { \mathcal { H } } | \psi \rangle = E _ { \mathrm { g } } + \lambda ^ { 2 } ( E _ { \mathrm { e } } - E _ { \mathrm { g } } ) .\tag{32}
$$

Similarly, the energy variance can be written as

$$
\sigma ^ { 2 } = \lambda ^ { 2 } \left( \left. \mathcal { H } ^ { 2 } \right. _ { \mathrm { e } } - 2 E _ { \mathrm { g } } E _ { \mathrm { e } } + E _ { \mathrm { g } } ^ { 2 } \right) + \mathcal { O } ( \lambda ^ { 4 } ) .\tag{33}
$$

If the error state $| \psi _ { \mathrm { e } } \rangle$ does not change substantially in different training attempts, there is a linear relation

$$
( E - E _ { \mathrm { g } } ) \propto \sigma ^ { 2 }\tag{34}
$$

for small λ, so a linear extrapolation to $\sigma ^ { 2 } = 0$ gives $E = E _ { \mathrm { g } } .$

As shown in Extended Data Fig. 3, the ratio $( E - E _ { \mathrm { g } } ) / \bar { \sigma } ^ { 2 }$ also remains nearly unchanged for different lattice sizes and symmetry sectors. This empirical conclusion is adopted to estimate the ratio in the large lattice from smaller ones so as to reduce the error and the time cost.

Lanczos step. The Lanczos step is a popular method in VMC for improving the variational accura $\mathsf { \overrightarrow { y } } ^ { 6 7 }$ . It is also used in $\mathsf { N Q S } ^ { 2 6 , 3 8 }$

The key idea of a Lanczos step is to construct new states $\left| \psi _ { \mathsf { p } } \right.$ orthogonal to the well-trained variational wavefunction $\left| \psi _ { 0 } \right.$ and to minimize the energy of the new state formed by a linear combination of |ψ ⟩ and $\left| \psi _ { \mathsf { p } } \right.$ . The new energy is then guaranteed to be lower than the initial energy.

Only one Lanczos step is applied in this work, so we have one state |ψ ⟩ satisfying ⟨ψ |ψ ⟩ = 0 given by

$$
\left. \psi _ { 1 } \right. = \frac { \mathcal { H } - E _ { 0 } } { \sigma } \left. \psi _ { 0 } \right. ,\tag{35}
$$

where $E _ { 0 } = \langle \psi _ { 0 } | \mathcal { H } | \psi _ { 0 } \rangle$ ⟩ and $\sigma ^ { 2 } = \left. \psi _ { 0 } | \mathcal { H } ^ { 2 } | \psi _ { 0 } \right. - E _ { 0 } ^ { 2 } .$ . The linear combination of $| \psi _ { 0 } \rangle$ ⟩ and $| \psi _ { 1 } \rangle$ can be written as

$$
\begin{array} { r } { | \psi _ { \alpha } \rangle = | \psi _ { 0 } \rangle + \alpha | \psi _ { 1 } \rangle , } \end{array}\tag{36}
$$

whose energy is

$$
E _ { \alpha } = E _ { 0 } + \frac { \langle \psi _ { \alpha } | \left( \mathcal { H } - E _ { 0 } \right) | \psi _ { \alpha } \rangle } { \langle \psi _ { \alpha } | \psi _ { \alpha } \rangle } = E _ { 0 } + \sigma \frac { \alpha ^ { 2 } \mu _ { 3 } + 2 \alpha } { \alpha ^ { 2 } + 1 } ,\tag{37}
$$

where

$$
\mu _ { n } = \frac { \left. \psi _ { 0 } \right. \left( \mathcal { H } - E _ { 0 } \right) ^ { n } \left. \psi _ { 0 } \right. } { \sigma ^ { n } } .\tag{38}
$$

The minimal energy is achieved at

$$
\alpha _ { \ast } = \frac { \mu _ { 3 } - \sqrt { \mu _ { 3 } ^ { 2 } + 4 } } { 2 } ,\tag{39}
$$

and the lowest energy is

$$
E _ { \alpha _ { * } } = E _ { 0 } + \sigma \frac { \alpha _ { * } ^ { 2 } \mu _ { 3 } + 2 \alpha _ { * } } { \alpha _ { * } ^ { 2 } + 1 } = E _ { 0 } + \sigma \alpha _ { * } .\tag{40}
$$

Initial guess of α. A direct way to compute ${ \bf { \dot { \mu } } } _ { \pmb { \mu } }$ is by measuring suitable quantities as expectation values of the initial state $\left| \psi _ { 0 } \right.$ . However, the measurement becomes more accurate if it is performed with a state $\left| \psi _ { \alpha _ { 0 } } \right.$ closer to the ground state<sup>67</sup>.

In this paper, we estimate the suitable $\alpha _ { 0 }$ to obtain a $\left| \psi _ { \alpha _ { 0 } } \right.$ closer to the true ground state compared to $| \psi _ { 0 } \rangle$ . Then, from equation (37), one can compute $\pmb { \mu } _ { 3 }$ as

$$
\mu _ { 3 } = \frac { \left( \alpha _ { 0 } ^ { 2 } + 1 \right) \left( E _ { \alpha _ { 0 } } - E _ { 0 } \right) / \sigma - 2 \alpha _ { 0 } } { \alpha _ { 0 } ^ { 2 } } ,\tag{41}
$$

where $E _ { \alpha _ { 0 } }$ can be measured by Monte Carlo sampling. The optimal α can be derived from $\pmb { \mu } _ { 3 }$ by equation (39), and the lowest energy is then given by equation (40).

Energy variance. To compute the energy variance of $| \psi _ { \alpha } \rangle$ , we start with an intermediate quantity

$$
v _ { \alpha } = \frac { \langle \psi _ { \alpha } | \left( \mathcal { H } - E _ { 0 } \right) ^ { 2 } | \psi _ { \alpha } \rangle } { \sigma ^ { 2 } \langle \psi _ { \alpha } | \psi _ { \alpha } \rangle } = \frac { \alpha ^ { 2 } \mu _ { 4 } + 2 \alpha \mu _ { 3 } + 1 } { \alpha ^ { 2 } + 1 } .\tag{42}
$$

Like the previous case, one can measure $\upsilon _ { \alpha _ { 0 } }$ by Monte Carlo sampling and determine ${ \bf { \sigma } } _ { \mu }$ as

$$
\mu _ { 4 } = \frac { \left( \alpha _ { 0 } ^ { 2 } + 1 \right) \upsilon _ { \alpha _ { 0 } } - 2 \alpha _ { 0 } \mu _ { 3 } - 1 } { \alpha _ { 0 } ^ { 2 } } .\tag{43}
$$

Then $\upsilon _ { \alpha _ { * } }$ can be computed given $\pmb { \mu } _ { 3 }$ and $\mu _ { 4 } ,$ which gives the required energy variance as

$$
\sigma _ { \alpha _ { \ast } } ^ { 2 } = \frac { \langle \psi _ { \alpha _ { \ast } } | \big ( \mathcal { H } - E _ { \alpha _ { \ast } } \big ) ^ { 2 } | \psi _ { \alpha _ { \ast } } \rangle } { \langle \psi _ { \alpha _ { \ast } } | \psi _ { \alpha _ { \ast } } \rangle } = \sigma ^ { 2 } v _ { \alpha _ { \ast } } - \big ( E _ { 0 } - E _ { \alpha _ { \ast } } \big ) ^ { 2 } .\tag{44}
$$

## Data availability

This research does not rely on any external datasets. The data shown in Figs. 2 and 3 and the obtained neural network weights are available via Zenodo at https://zenodo.org/doi/10.5281/zenodo.7657551 (ref. 68).

## Code availability

We provide the code needed to reproduce our main results via Zenodo at https://zenodo.org/doi/10.5281/zenodo.7657551 (ref.68) and via GitHub at https://github.com/ChenAo-Phys/MinSR.

## References

52. Schmitt, M. & Reh, M. jvmc: Versatile and performant variational Monte Carlo leveraging automated diferentiation and GPU acceleration. SciPost Physics Codebase https://www. scipost.org/SciPostPhysCodeb.2?acad\_field\_slug= politicalscience (2021).

53. Vicentini F. et al. Netket 3: machine learning toolbox for many-body quantum systems. SciPost Physics Codebase https://www.scipost.org/10.21468/SciPostPhysCodeb.7?acad\_ field\_slug=politicalscience (2021).

54. He, K., Zhang, X., Ren, S. & Sun, J. Identity mappings in deep residual networks. In Proc. Computer Vision – ECCV 2016 (eds Leibe, B. et al.) 630–645 (Springer, 2016).

55. Marshall, W. & Peierls, R. E. Antiferromagnetism. Proc. R. Soc. Lond. Ser. A 232, 48 (1955).

56. Szabó, A. & Castelnovo, C. Neural network wave functions and the sign problem. Phys. Rev. Res. 2, 033075 (2020).

57. Chen, A., Choo, K., Astrakhantsev, N. & Neupert, T. Neural network evolution strategy for solving quantum sign structures. Phys. Rev. Res. 4, L022026 (2022).

58. Choo, K., Carleo, G., Regnault, N. & Neupert, T. Symmetries and many-body excitations with neural-network quantum states. Phys. Rev. Lett. 121, 167204 (2018).

59. Reh, M., Schmitt, M. & Gärttner, M. Optimizing design choices for neural quantum states. Phys. Rev. B 107, 195115 (2023).

60. Westerhout, T. lattice-symmetries: A package for working with quantum many-body bases. J. Open Source Softw. 6, 3537 (2021).

61. Liang, X., Dong, S.-J. & He, L. Hybrid convolutional neural network and projected entangled pair states wave functions for quantum many-particle states. Phys. Rev. B 103, 035138 (2021).

62. Ferrari, F., Becca, F. & Carrasquilla, J. Neural Gutzwiller-projected variational wave functions. Phys. Rev. B 100, 125131 (2019).

63. Wu, D. et al. Variational benchmarks for quantum many-body problems. Preprint at https://arxiv.org/abs/2302.04919 (2023).

64. Rende, R., Viteritti, L. L., Bardone, L., Becca,F. & Goldt, S. A simple linear algebra identity to optimize large-scale neural network quantum states. https://arxiv.org/abs/2310.05715 (2023).

65. Kwon, Y., Ceperley, D. M. & Martin, R. M. Efects of backflow correlation in the three-dimensional electron gas: quantum Monte Carlo study. Phys. Rev. B 58, 6800 (1998).

66. Kashima, T. & Imada, M. Path-integral renormalization group method for numerical study on ground states of strongly correlated electronic systems. J. Phys. Soc. Jpn 70, 2287 (2001).

67. Sorella, S. Generalized Lanczos algorithm for variational quantum Monte Carlo. Phys. Rev. B 64, 024512 (2001).

68. Chen, A. & Heyl, M. Empowering deep neural quantum states through eficient optimization: data and code. Zenodo https://zenodo.org/doi/10.5281/zenodo.7657551 (2023).

## Acknowledgements

We gratefully acknowledge M. Schmitt for help in improving the manuscript. We also thank T. Neupert, C. Roth, M. Bukov, F. Vicentini,

W.-Y. Liu and X. Liang for fruitful discussions. This project has received funding from the European Research Council under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 853443). This work was supported by the German Research Foundation (Project No. 492547816; TRR 360). We also acknowledge the Gauss Centre for Supercomputing e.V. (www. gauss-centre.eu) for funding this project by providing computing time through the John von Neumann Institute for Computing on the GCS Supercomputer JUWELS at Jülich Supercomputing Centre. We gratefully acknowledge the scientific support and high-performance computing resources provided by the Erlangen National High Performance Computing Center (NHR) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (NHR Project No. nqsQuMat). NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (Grant No. 440719683).

## Author contributions

A.C. proposed the method and performed numerical simulations. M.H. provided computing resources and supervised the simulations. Both authors took part in the design of numerical experiments, the writing of the manuscript and the critical review of the paper.

## Funding

Open access funding provided by Universität Augsburg.

## Competing interests

The authors declare no competing interests.

## Additional information

Extended data is available for this paper at https://doi.org/10.1038/s41567-024-02566-1.

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41567-024-02566-1.

Correspondence and requests for materials should be addressed to Ao Chen or Markus Heyl.

Peer review information Nature Physics thanks Guglielmo Mazzola and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

Reprints and permissions information is available at www.nature.com/reprints.

![](images/extended-data-fig-01a-solver-time.jpg)

![](images/extended-data-fig-01b-residual-error.jpg)

![](images/extended-data-fig-01c-training-convergence.jpg)  
Extended Data Fig. 1 | Performance evaluation of various optimization methods on the 10 × 10 square Heisenber ${ \pmb J } _ { 1 } { \pmb J } _ { 2 }$ model. a, Time cost of solving the equation $\overline { { O } } \delta \theta = \overline { { \epsilon } }$ for the different optimization methods for different numbers of Monte-Carlo samples $N _ { s }$ and variational parameters $N _ { p } ,$ measured at $J _ { 2 } = 0$ on the ResNet1 architecture and an A100 80G GPU. The time cost of other  
contributions in a training step on 4 parallel A100 GPUs is presented as the black star, showing that the time cost of SR becomes the bottleneck for training deep networks if MinSR is not employed. b, Relative residual error $| | \overline { { O } } \delta \theta - \overline { { \epsilon } } | | / | | \overline { { \epsilon } } | | \mathrm { f o r }$ $N _ { s } { = } 1 0 ^ { 4 } { \mathrm { a t } } J _ { 2 } { = } 0$ on ResNet1. $\mathbf { \nabla } . \mathbf { c } ,$ Training curvea of ResNet2 $\mathbf { a t } J _ { 2 } / J _ { 1 } = 0 . 5$ with $N _ { s } = 1 0 ^ { 4 }$ and $N _ { p } { \approx } 1 0 ^ { 6 }$ comparing different optimizers.

![](images/extended-data-fig-02-wavefunction-amplitudes.jpg)  
Extended Data Fig. 2 | NQS wave function amplitudes for the Heisenberg and $J _ { 1 } \mathbf { - } J _ { 2 }$ models on a 6 × 6 square lattice obtained by ResNet1 with 64 layers and 146320 parameters by means of MinSR. The ED wave function amplitudes obtained by the lattice-symmetries package [64] are shown as the black dotted lines, and the spin configurations are sorted according to the descending order of ED amplitudes. All wave function amplitudes are shown for the Heisenberg model, while for the J -J model only one point is plotted among 10000 successive points. In the inset, we show the infidelity with different  
numerical precision. The infidelity of a ResNet1 with 13750 parameters which approaches the size limit of SR (pinv) is also presented for comparison. This shows that the deep NQS trained by MinSR can approach TF32 precision in the non-frustrated case and BF16 precision in the frustrated case on the 6 × 6 lattice with a Hilbert space dimension of 15804956 after applying symmetry, while the shallow NQS trained by traditional SR cannot. In our further tests, the shallow network trained by traditional SR can only approach such precision on the 4 × 6 lattice with a much smaller Hilbert space dimension of 15578.

![](images/extended-data-fig-03-square-6x6.jpg)

![](images/extended-data-fig-03-square-8x8.jpg)

![](images/extended-data-fig-03-square-10x10.jpg)

![](images/extended-data-fig-03-square-12x12.jpg)

![](images/extended-data-fig-03-square-16x16.jpg)

![](images/extended-data-fig-03-square-20x20.jpg)  
Extended Data Fig. 3 | Zero-variance extrapolation for the square $J _ { 1 } J _ { 2 }$ model $\mathbf { a t } J _ { 2 } / J _ { 1 } = 0 .$ 5 with different lattice sizes. The three data points in the same fitting are obtained by two different sizes of real-valued ResNet2 with 34944 and 139008 parameters and a Lanczos step on the larger one. The ground state sector $\mathbf { i s } S = 0 \mathbf { a n d } k \mathbf { = } \Gamma \mathbf { = } ( 0 , 0 )$ , and the excited state sector is $S = 1 \mathrm { a n d } p = M = ( \pi , \pi ) $ The error bars show the standard deviations. On the largest 20 × 20 lattice, the estimation of the slope is inaccurate in direct linear fitting due to the uncertainty of data points. Consequently, we utilize an empirical assumption that the slope  
remains nearly unchanged for different system sizes and symmetry sectors. Excluding too small lattices in which the estimation of slopes is inaccurate due to too close data points, the tendency of unchanged slopes is obvious for the slopes on $L = 1 0 , 1 2$ , 16, respectively 0.17(2), 0.18(4), 0.15(4) for S = 0, and 0.14(3), 0.16(2), $0 . 1 6 ( 2 ) \mathrm { f o r } S = 1 .$ The unchanged slope is also observed in existing literature [8, 16, 37]. Consequently, we employ the average of the aforementioned slopes as the slope on L = 20 to mitigate the error.

![](images/extended-data-fig-04-triangular-6x6.jpg)

![](images/extended-data-fig-04-triangular-12x12.jpg)  
Extended Data Fig. 4 | Zero-variance extrapolation for the triangular J -J model at $J _ { 2 } / J _ { 1 } = 0 . 1 2 5$ with different lattice sizes. The three data points in the same fitting are obtained by two sizes of complex-valued ResNet2 with 34944 and 139008 parameters and a Lanczos step on the larger one. The ground state sector is $S = 0 , k = \Gamma = ( 0 , 0 )$ , and the excited state sector is $S = 1 , k = K = ( 4 \pi / 3 , 0 ) ,$

![](images/extended-data-fig-04-triangular-6x9.jpg)

![](images/extended-data-fig-04-triangular-18x18.jpg)  
The error bars show the standard deviations. Similar to the square lattice case, the slope on the 18 × 18 lattice is approximated by the slope on the 12 × 12 lattice to mitigate the error. The data on the 18 × 18 lattice is generated by networks with 139008 parameters and a Lanczos step.

As a reference, the ground state energy estimated by zero-variance extrapolation in this work is − 0.497715(9).

Extended Data Table 1 | Variational ground state energies in the 10 × 10 square $\pmb { J } _ { 1 } \pmb { - J } _ { 2 }$ model with PBC at $\mathbf { \mathit { J } } _ { 2 } / \mathbf { \mathit { J } } _ { 1 } { = } 0 . 5$
<table><tr><td>Wave function</td><td>Parameters</td><td>Reference</td><td>E/N</td><td> $\sigma ^ { 2 } / N$ </td></tr><tr><td>Tensor Network</td><td></td><td></td><td></td><td></td></tr><tr><td>DMRG</td><td>8192 SU(2) or 32000 U(1) states</td><td>[9]</td><td>-0.495530</td><td></td></tr><tr><td> $\mathrm { P E P S } + \mathrm { C N N }$ </td><td>Bond dimension 4 in  $\mathrm { P E P S } + 3 5 3 1$  real in CNN</td><td>[61]</td><td>-0.495502</td><td></td></tr><tr><td>GWF</td><td></td><td></td><td></td><td></td></tr><tr><td>Determinant</td><td>5 real</td><td>[8]</td><td>-0.49521(1)</td><td> $\approx 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Determinant + 2LS</td><td>5 real</td><td>[8]</td><td>-0.49755(1)</td><td> $\approx 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Determinant + RBM</td><td>810 complex</td><td>[62]</td><td>-0.49575(3)</td><td></td></tr><tr><td> $\mathrm { P P } + \mathrm { R B M }$ </td><td>10000 real in PP + 1616 complex in RBM</td><td>[24]</td><td>-0.497629(1)</td><td> $4 . 9 ( 2 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Pure NQS</td><td></td><td></td><td></td><td></td></tr><tr><td>CNN</td><td>3838 complex</td><td>[31]</td><td>-0.49516(1)</td><td></td></tr><tr><td>RNN</td><td>Unknown</td><td>[34,63]</td><td>-0.49560(4)</td><td> $2 . 9 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>RBM</td><td>30000 complex</td><td>[38]</td><td>-0.49580(2)</td><td></td></tr><tr><td> $\mathrm { R B M } + \mathrm { L S }$ </td><td>30000 complex</td><td>[38]</td><td>-0.4968(4)</td><td></td></tr><tr><td> $\mathrm { C N N } + \mathrm { L S }$ </td><td>106529 real</td><td>[37]</td><td>-0.497468</td><td> $\approx 1 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GCNN</td><td>≈ 75000 real</td><td>[26]</td><td>-0.497437(7)</td><td></td></tr><tr><td>ViT(MinSR)</td><td>267720 real</td><td>[64]</td><td>-0.497634(1)</td><td></td></tr><tr><td>ResNet(MinSR)</td><td>1071488 real</td><td></td><td>This work -0.4976921(4)</td><td> $1 . 8 0 ( 4 ) \times 1 0 ^ { - 4 }$ </td></tr></table>

## CONTENTS

A. Fubini-Study distance 2   
B. Computational complexity 2   
C. Structure factor 4   
D. Gap estimation with and without zero-variance extrapolation 5   
References 6

## A. FUBINI-STUDY DISTANCE

The original definition of the Fubini-Study distance between the variational state $| \Psi _ { \theta ^ { \prime } } \rangle = | \Psi _ { \theta + \delta \theta } \rangle$ and the imaginarytime evolving state $| \Psi ^ { \prime } \rangle = e ^ { - \mathcal { H } \delta \tau } | \Psi _ { \theta } \rangle$ is [1–3]

$$
d ( \Psi _ { \theta ^ { \prime } } , \Psi ^ { \prime } ) = \operatorname { a r c c o s } \frac { \mid \langle \Psi _ { \theta ^ { \prime } } | \Psi ^ { \prime } \rangle \mid } { \mid \mid \Psi _ { \theta ^ { \prime } } \mid \mid \cdot \mid \mid \Psi ^ { \prime } \mid \mid } .\tag{A.1}
$$

Assuming that $| \Psi _ { \theta ^ { \prime } } \rangle = | \Psi _ { \theta } \rangle + | \delta \Psi _ { \theta } \rangle$ and $\left| \Psi ^ { \prime } \right. = \left| \Psi _ { \theta } \right. + \left| \delta \Psi _ { H } \right.$ where $\left| \delta \Psi _ { \theta } \right.$ and $ { \left. \delta \Psi _ { H } \right. }$ are both small quantities, one can expand the FS distance to the lowest order of $\left| \delta \Psi \right. / \left| \left| \Psi \right| \right|$ as

$$
d ^ { 2 } ( \Psi _ { \theta ^ { \prime } } , \Psi ^ { \prime } ) = \Big ( \langle \delta \tilde { \Psi } _ { \theta } | - \langle \delta \tilde { \Psi } _ { H } | \Big ) \Big ( | \delta \tilde { \Psi } _ { \theta } \rangle - | \delta \tilde { \Psi } _ { H } \rangle \Big ) ,\tag{A.2}
$$

where

$$
| \delta \tilde { \Psi } \rangle = \frac { | \delta \Psi \rangle } { | | \Psi | | } - \frac { \langle \Psi | \delta \Psi \rangle | \Psi \rangle } { | | \Psi | | ^ { 3 } }\tag{A.3}
$$

is the normalized increment perpendicular to the original state for $\left| \delta \Psi _ { \boldsymbol { \theta } } \right.$ and $| \delta \Psi _ { H } \rangle$

The change of $\left| \Psi _ { \boldsymbol { \theta } } \right.$ is induced by the change of parameters as

$$
\left| \delta \Psi _ { \theta } \right. = \sum _ { \sigma } \sum _ { k } \frac { \partial \psi _ { \sigma } } { \partial \theta _ { k } } \delta \theta _ { k } \left| \sigma \right. = \sum _ { \sigma } \psi _ { \sigma } \sum _ { k } O _ { \sigma k } \delta \theta _ { k } \left| \sigma \right. ,\tag{A.4}
$$

where $\begin{array} { r } { O _ { \sigma k } = \frac { 1 } { \psi _ { \sigma } } \frac { \partial \psi _ { \sigma } } { \partial \theta _ { k } } } \end{array}$ . According to the imaginary-time evolution $| \Psi ^ { \prime } \rangle = e ^ { - \mathcal { H } \delta \tau } | \Psi _ { \theta } \rangle$ , the change of the state $\vert \Psi ^ { \prime } \rangle$ to the first order of $\delta \tau$ is

$$
\left| \delta \Psi _ { H } \right. = - \mathcal { H } \delta \tau \left| \Psi _ { \theta } \right. = - \delta \tau \sum _ { \sigma } \psi _ { \sigma } E _ { \mathrm { l o c } , \sigma } \left| \sigma \right. ,\tag{A.5}
$$

where $\begin{array} { r } { E _ { \mathrm { l o c } , \sigma } = \sum _ { \sigma ^ { \prime } } \frac { \psi _ { \sigma ^ { \prime } } } { \psi _ { \sigma } } H _ { \sigma \sigma ^ { \prime } } } \end{array}$ . Putting Eq. (A.4) and $\mathrm { E q . ( A . 5 ) }$ into $\mathrm { E q . ( A . 3 ) }$ , we obtain

$$
| \delta \tilde { \Psi } _ { \boldsymbol { \theta } } \rangle = \sqrt { N _ { s } } \sum _ { \sigma } \frac { \psi _ { \sigma } } { | | \Psi | | } \sum _ { k } \overline { { { O } } } _ { \sigma k } \delta \theta _ { k } | \sigma \rangle ,\tag{A.6}
$$

$$
\left| \delta \tilde { \Psi } _ { H } \right. = \sqrt { N _ { s } } \sum _ { \sigma } \frac { \psi _ { \sigma } } { \left| \left| \Psi \right. \right| } \overline { { { \epsilon } } } _ { \sigma } \left| \sigma \right. ,\tag{A.7}
$$

where $\overline { { O } } _ { \sigma k } = ( O _ { \sigma k } - \langle O _ { \sigma k } \rangle ) / \sqrt { N _ { s } } , \overline { { \epsilon } } _ { \sigma } = - \delta \tau ( E _ { \mathrm { l o c } , \sigma } - \langle E _ { \mathrm { l o c } , \sigma } \rangle ) / \sqrt { N _ { s } } .$

Substituting Eq. (A.6) and Eq. (A.7) into Eq. (A.2), the FS distance becomes

$$
\begin{array} { l } { \displaystyle d ^ { 2 } ( \Psi _ { \theta ^ { \prime } } , \Psi ^ { \prime } ) = N _ { s } \sum _ { \sigma } \frac { \left| \psi _ { \sigma } \right| ^ { 2 } } { \left| \left| \Psi \right| \right| ^ { 2 } } \left| \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } \right| ^ { 2 } } \\ { = N _ { s } \left. \left| \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } \right| ^ { 2 } \right. } \\ { = \displaystyle \sum _ { \sigma \operatorname* { i n s a m p l e s } } \left| \sum _ { k } \overline { { O } } _ { \sigma k } \delta \theta _ { k } - \overline { { \epsilon } } _ { \sigma } \right| ^ { 2 } , } \end{array}\tag{A.8}
$$

which proves the equation used in the main text.

## B. COMPUTATIONAL COMPLEXITY

In this section, we discuss the computational complexity of NQS optimization. The epochs required for convergence can vary for diferent systems, so we will limit the discussion to one training iteration. The computational cost is contributed by four major parts, namely the Monte Carlo sampling, the local energy computation, the gradient computation, and the MinSR equation. The quantities necessary for the complexity analysis are the number of lattice sites $N _ { \mathrm { s i t e } } .$ , the number of Monte Carlo samples $N _ { s } ,$ the number of parameters $N _ { p } .$ , and the order $| G |$ of the symmetry group. Furthermore, we denote the time cost of a single forward pass as $F ,$ , which depends on $N _ { p } , N _ { \mathrm { s i t e } } ,$ and also on the details of the chosen network architecture. The CNN architecture utilized in this work exhibits $\mathcal { O } ( F ) =$ $\mathcal { O } ( N _ { p } N _ { \mathrm { s i t e } } )$ [4].

The Monte Carlo sampling requires $N _ { s }$ symmetrized forward passes in every update, and $\mathcal { O } ( N _ { \mathrm { s i t e } } )$ updates to reduce the correlation between samples, leading to a complexity $\mathcal { O } ( N _ { s } \left| G \right| F N _ { \mathrm { s i t e } } )$ . For every Monte Carlo sample $\sigma ,$ the local energy requires computing the wave function $\psi _ { \sigma ^ { \prime } }$ for every $\sigma ^ { \prime }$ with $\langle { \sigma ^ { \prime } } | \mathcal { H } | { \sigma } \rangle \ne 0$ , which, in the typical case of local interactions, amounts to $\mathcal { O } ( N _ { \mathrm { s i t e } } )$ forward passes with symmetry. This again gives the complexity $\mathcal { O } ( N _ { s } \left| G \right| F N _ { \mathrm { s i t e } } )$ To compute the gradient or the Jacobian matrix, one needs to perform a symmetrized forward and backward pass for all Monte Carlo samples. The time costs of forward and backward passes are similar, so the computational complexity is $\mathcal { O } ( N _ { s } \vert G \vert F )$ . As discussed in the main text, the complexity of solving the MinSR equation is $\mathcal { O } ( N _ { p } N _ { s } ^ { 2 } + N _ { s } ^ { \hat { 3 } } )$ . The time cost of this part is usually not a bottleneck, since the simple matrix computation is much faster than the forward pass of deep neural networks.

In summary, the dominant computational complexity, as also indicated in Extended Fig. 1(a), is $\mathcal { O } ( N _ { s } \left| G \right| F N _ { \mathrm { s i t e } } )$ given by the Monte Carlo sampling and the local energy computation. Which one becomes the bottleneck depend on several details, including hardware memory, sample correlation, Hamiltonian connectivity, etc.

Considering $\mathcal { O } ( F ) = \mathcal { O } ( N _ { p } N _ { \mathrm { s i t e } } )$ in the presently utilized CNN, the complexity becomes $\mathcal { O } ( N _ { p } N _ { s } | G | N _ { \mathrm { s i t e } } ^ { 2 } )$ . The dependence of $N _ { p }$ on $N _ { \mathrm { s i t e } }$ is, in general, an open question in the study of $\mathrm { N Q S }$ . For the physical models considered in our work, specifically in the frustrated quantum magnets, we find that a larger $N _ { p }$ is required to achieve the same level of accuracy upon increasing the system size $N _ { \mathrm { s i t e } }$ . Empirically, we observe that the dependence is roughly linear. However, let us point out that this likely depends on the details of the model. For our case, such an N -dependence of $N _ { p }$ can be directly related to the reduction of the finite-size gap upon increasing the system size. From our own empiric observations, we find that for some models $N _ { p }$ can also remain unchanged, $\mathrm { e . g . }$ , for systems with a large and finite gap. Overall, these considerations lead to an overall $\mathcal { O } ( N _ { \mathrm { s i t e } } ^ { 2 } )$ or $\mathcal { O } ( N _ { \mathrm { s i t e } } ^ { 3 } )$ complexity, respectively. However, let us also point out that these empirical complexity observations can also depend on the details of the chosen network architecture. In this context it is of particular importance to put a specific focus in future research to the identification of suitable architectures with improved complexity behavior in NQS.

Reducing memory complexity. The local energy and the gradient computations are bottlenecks of the memory cost. As the local energy can be computed by splitting the inputs into small batches, here we discuss how to reduce the memory complexity when computing the matrix T.

Directly computing the matrix $T$ requires storing the value of O with shape $N _ { s } \times N _ { p }$ , which may cause overflow issues on common GPUs for large $N _ { p }$ . To alleviate this problem, we compute the elements of $T$ as

$$
T _ { \sigma \sigma ^ { \prime } } = \sum _ { k } \overline { { O } } _ { \sigma k } \overline { { O } } _ { \sigma ^ { \prime } k } ^ { * } = \sum _ { l } \sum _ { k ^ { ( l ) } } \overline { { O } } _ { \sigma k ^ { ( l ) } } \overline { { O } } _ { \sigma ^ { \prime } k ^ { ( l ) } } ^ { * } = \sum _ { l } T _ { \sigma \sigma ^ { \prime } } ^ { ( l ) } ,\tag{B.1}
$$

where l iterates over diferent layers, $k ^ { ( l ) }$ iterates over parameters in layer $l ,$ and $\begin{array} { r } { T _ { \sigma \sigma ^ { \prime } } ^ { ( l ) } = \sum _ { k ^ { ( l ) } } \overline { { O } } _ { \sigma k ^ { ( l ) } } \overline { { O } } _ { \sigma ^ { \prime } k ^ { ( l ) } } ^ { * } } \end{array}$ . Consequently, $\overline { O }$ only has to be stored for each layer instead of the whole network and its memory can be freed after obtaining $T _ { \sigma \sigma ^ { \prime } } ^ { ( l ) }$ , which can be achieved by performing backward pass layer by layer. The memory complexity is hence reduced from $\mathcal { O } ( N _ { s } N _ { p } )$ to $\mathcal { O } ( N _ { s } N _ { p } ^ { ( l ) } )$ , where e Np $N _ { p } ^ { ( l ) }$ is the number of parameters in layer l. For more details, one can check the structured derivatives method in Ref. [5].

Another major source of memory cost comes from storing intermediate values in the computation of gradients. For a network with $n _ { l }$ layers and $n _ { y } ^ { ( l ) }$ neurons in each layer, the memory complexity is $\mathcal { O } \left( n _ { y } ^ { ( \bar { l } ) } n _ { l } \right)$ which is comparable to $\mathcal { O } ( N _ { s } N _ { p } )$ . This can be solved by the gradient checkpointing method [6], in which the intermediate values are only stored for every $\sqrt { n _ { l } }$ layers. During the backward pass, the missing intermediate values can be recovered from the stored values. The additional cost in time only amounts to one additional forward pass, but helps to drastically reduce the memory complexity to $\mathcal { O } \left( n _ { y } ^ { ( l ) } \sqrt { n _ { l } } \right)$ .

The aforementioned structured derivatives and gradient checkpointing methods introduce additional time costs but do not alter the time complexity. With these two methods, the memory complexity of MinSR is reduced to $\mathcal { O } \left( N _ { s } N _ { p } ^ { ( l ) } + n _ { y } ^ { ( l ) } \sqrt { n _ { l } } \right)$

## C. STRUCTURE FACTOR

There exists several literature focusing on the triangular $J _ { 1 ^ { - } } J _ { 2 }$ model and supporting the existence of a quantum spin liquid (QSL) phase [7–16]. Here we also provide the spin and dimer structure factors on the $1 2 \times 1 2$ triangular lattice to confirm the existence of QSLs. The structure factor is given by

$$
S ( \mathbf { q } ) = \frac { 1 } { N _ { \mathrm { s i t e } } } \sum _ { i , j } C _ { i j } e ^ { i \mathbf { q } \cdot ( \mathbf { r } _ { i } - \mathbf { r } _ { j } ) } ,\tag{C.1}
$$

where $C _ { i j }$ is the spin-spin correlation $\langle \mathbf { S } _ { i } \cdot \mathbf { S } _ { j } \rangle$ or the dimer-dimer correlation $\langle ( \mathbf { S } _ { i } \cdot \mathbf { S } _ { i + \hat { x } } ) ( \mathbf { S } _ { j } \cdot \mathbf { S } _ { j + \hat { x } } ) \rangle - \langle \mathbf { S } _ { i } \cdot \mathbf { S } _ { i + \hat { x } } \rangle \langle \mathbf { S } _ { j } \cdot \mathbf { S } _ { j + \hat { x } } \rangle$ in which we choose the dimer in x direction. As shown in Fig. 1, the spin structure peak, representing the antiferromagnetic (AFM) order, is weakened when $J _ { 2 } / J _ { 1 }$ increases from 0.060 to 0.125, which is expected when the system goes from an ordered phase to a QSL phase. On the other hand, the dimer structure factor representing the valence-bond solid (VBS) order remains nearly unchanged.

To further confirm the existence of a QSL phase, we measure the system-size dependence of the correlation ratio [17]

$$
R = 1 - S ( \mathbf { Q } _ { \mathrm { p e a k } } + \delta \mathbf { q } ) / S ( \mathbf { Q } _ { \mathrm { p e a k } } ) ,\tag{C.2}
$$

where $\mathbf { Q } _ { \mathrm { p e a k } }$ represents the peak position in k-space, which is $\mathbf { Q } _ { \mathrm { p e a k } } = ( 4 \pi / 3 , 0 )$ for the spin structure and $\mathbf { Q } _ { \mathrm { p e a k } } =$ $( 2 \pi / 3 , 0 )$ for the dimer structure, and $\mathbf { Q } _ { \mathrm { p e a k } } + \delta \mathbf { q }$ is the neighbor of the peak position with a second highest structure factor. The correlation ratio of the spin structure has a crossing point at $J _ { 2 } / J _ { 1 } \approx 0 . 0 6 3$ , which can be taken as an indicator for a quantum phase transition from an AFM phase to a QSL phase. The correlation ratio of the dimer structure, instead, remains small and does not exhibit a crossing point, indicating a vanishing VBS order in the studied regime.

![](images/supplementary-fig-01-structure-factors.jpg)  
FIG. 1. The spin structure factor (upper) and the dimer structure factor (lower) in the $1 2 \times 1 2$ triangular $J _ { 1 ^ { - } } J _ { 2 }$ model.

![](images/supplementary-fig-02-correlation-ratios.jpg)  
FIG. 2. The correlation ratio R of the spin and dimer structure on the $6 \times 6$ and $1 2 \times 1 2$ triangular lattice.

## D. GAP ESTIMATION WITH AND WITHOUT ZERO-VARIANCE EXTRAPOLATION

In this section, we present the estimation of energy gaps with and without zero-variance extrapolation and perform a comparison between them. The result with variance extrapolation shows converged $\Delta \times L$ when $L \to \infty$ , indicating $\Delta \to 0$ in the thermodynamic limit.

For the result without zero-variance extrapolation, the displayed data requires a more detailed discussion. Excluding for the moment the data point for the largest system size, our data also seems to be consistent with our previous result of a gapless phase as shown in the middle panel of Fig. 3. However, the unextrapolated $1 8 \times 1 8$ data point appears to be a complete outlier in the right panel.

The key problem in this situation is that the variational accuracy of the excited state is usually worse than that of the ground state. This diference becomes even more significant in complex systems at larger system sizes, leading to larger estimated gaps in these systems. When operating with fixed computational resources, as one increases the system size, this has the unfortunate consequence that the gap is more and more biased towards larger values. Thus, any extrapolation of such data to the thermodynamic limit sufers in this sense from a systematic bias toward larger

![](images/supplementary-fig-03-extrapolated.jpg)

![](images/supplementary-fig-03-unextrapolated-without-largest.jpg)

![](images/supplementary-fig-03-unextrapolated-with-largest.jpg)  
FIG. 3. The system size dependence of the energy gaps in the triangular $J _ { 1 ^ { - } } J _ { 2 }$ model at J ${ } ; / J _ { 1 } = 0 . 1 2 5$

values.

Although this deviation may not apply to all diferent circumstances, it is still quite common in the variational Monte Carlo. For instance, in Ref. [18] the overestimation of energy gaps is observed when the zero-variance extrapolation is not performed. In particular, when aiming at deciding whether a gap closes or not, it is of key importance to eliminate this bias, which requires an extrapolation as we have been performing.

[1] G. Fubini, Sulle metriche definite da una forma hermitiana: nota (Ofice graf. C. Ferrari, 1904).

[2] E. Study, K¨urzeste wege im komplexen gebiet, Mathematische Annalen 60, 321 (1905).

[3] C.-Y. Park and M. J. Kastoryano, Geometry of learning neural quantum states, Phys. Rev. Research 2, 023232 (2020).

[4] K. He and J. Sun, Convolutional neural networks at constrained time cost, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015).

[5] R. Novak, J. Sohl-Dickstein, and S. S. Schoenholz, Fast finite width neural tangent kernel, in International Conference on Machine Learning (2022).

[6] T. Chen, B. Xu, C. Zhang, and C. Guestrin, Training deep nets with sublinear memory cost (2016), arXiv:1604.06174 [cs.LG].

[7] R. Kaneko, S. Morita, and M. Imada, Gapless spin-liquid phase in an extended spin $1 / 2$ triangular heisenberg model, Journal of the Physical Society of Japan 83, 093707 (2014).

[8] Z. Zhu and S. R. White, Spin liquid phase of the $\begin{array} { r } { s = { \frac { 1 } { 2 } } \ J _ { 1 } - J _ { 2 } } \end{array}$ heisenberg model on the triangular lattice, Phys. Rev. B 92, 041105 (2015).

[9] W.-J. Hu, S.-S. Gong, W. Zhu, and D. N. Sheng, Competing spin-liquid states in the $\mathrm { s p i n } - \frac { 1 } { 2 }$ heisenberg model on the triangular lattice, Phys. Rev. B 92, 140403 (2015).

[10] Y. Iqbal, W.-J. Hu, R. Thomale, D. Poilblanc, and F. Becca, Spin liquid nature in the heisenberg $J _ { 1 } \mathrm { ~ - ~ } J _ { 2 }$ triangular antiferromagnet, Phys. Rev. B 93, 144411 (2016).

[11] S. N. Saadatmand and I. P. McCulloch, Symmetry fractionalization in the topological phase of the spin- <sup>1</sup> $J _ { 1 } - J _ { 2 }$ triangular heisenberg model, Phys. Rev. B 94, 121111 (2016).

[12] A. Wietek and A. M. L¨auchli, Chiral spin liquid and quantum criticality in extended $s = \textstyle { \frac { 1 } { 2 } }$ heisenberg models on the triangular lattice, Phys. Rev. B 95, 035141 (2017).

[13] S.-S. Gong, W. Zhu, J.-X. Zhu, D. N. Sheng, and K. Yang, Global phase diagram and quantum spin liquids in $\mathrm { a \ s p i n { - } { \frac { 1 } { 2 } } }$ triangular antiferromagnet, Phys. Rev. B 96, 075116 (2017).

[14] S. Hu, W. Zhu, S. Eggert, and Y.-C. He, Dirac spin liquid on the spin-1/2 triangular heisenberg antiferromagnet, Phys. Rev. Lett. 123, 207203 (2019).

[15] Y.-F. Jiang and H.-C. Jiang, Nature of quantum spin liquids of the $\begin{array} { r } { s = \frac { 1 } { 2 } } \end{array}$ heisenberg antiferromagnet on the triangular lattice: A parallel dmrg study, Phys. Rev. B 107, L140411 (2023).

[16] N. E. Sherman, M. Dupont, and J. E. Moore, Spectral function of the $J _ { 1 } - J _ { 2 }$ heisenberg model on the triangular lattice, Phys. Rev. B 107, 165146 (2023).

[17] Y. Nomura and M. Imada, Dirac-type nodal spin liquid revealed by refined quantum many-body solver using neural-network wave function, correlation ratio, and level spectroscopy, Phys. Rev. X 11, 031034 (2021)

[18] W.-J. Hu, F. Becca, A. Parola, and S. Sorella, Direct evidence for a gapless $Z _ { 2 }$ spin liquid by frustrating n´eel antiferromagnetism, Phys. Rev. B 88, 060402 (2013)
