# open_spiel

## Background

- Decouple problem specification (what to solve) and solver techniques (how to solve).
- Specify the database tuning problem as a single-player game.
- Solve with computational game solving / reinforcement learning techniques.
- Implemented in [OpenSpiel](https://github.com/deepmind/open_spiel) [^openspiel].
- More background material: [^sandholm]

## Folder contents

- database.h
  - Header for database tuning problem specification.
  - Corresponding OpenSpiel location: open_spiel/games/
- database.cc
  - Implementation for database tuning problem specification.
  - Corresponding OpenSpiel location: open_spiel/games/
- database_game.cc
  - Solver for the database tuning problem.
  - Corresponding OpenSpiel location: open_spiel/examples/
- CMakeLists.txt
  - Build system for this project.
  - You probably will need to modify this file:
    - If you add dependencies.
    - If you use OpenSpiel's JAX / PyTorch / TensorFlow / etc. components. 

## References

[^openspiel]: OpenSpiel: A Framework for Reinforcement Learning in Games.

    ```
    @article{LanctotEtAl2019OpenSpiel,
      title     = {{OpenSpiel}: A Framework for Reinforcement Learning in Games},
      author    = {Marc Lanctot and Edward Lockhart and Jean-Baptiste Lespiau and
                   Vinicius Zambaldi and Satyaki Upadhyay and Julien P\'{e}rolat and
                   Sriram Srinivasan and Finbarr Timbers and Karl Tuyls and
                   Shayegan Omidshafiei and Daniel Hennes and Dustin Morrill and
                   Paul Muller and Timo Ewalds and Ryan Faulkner and J\'{a}nos Kram\'{a}r
                   and Bart De Vylder and Brennan Saeta and James Bradbury and David Ding
                   and Sebastian Borgeaud and Matthew Lai and Julian Schrittwieser and
                   Thomas Anthony and Edward Hughes and Ivo Danihelka and Jonah Ryan-Davis},
      year      = {2019},
      eprint    = {1908.09453},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG},
      journal   = {CoRR},
      volume    = {abs/1908.09453},
      url       = {http://arxiv.org/abs/1908.09453},
    }
    ```

[^sandholm]: https://www.cs.cmu.edu/~sandholm/cs15-888F21/
