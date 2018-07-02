# Rainy Day

## Description

This is an interview project.

The project poses a problem, in which the protagonist is driving a car in a street at a night when rain is pouring, and faces a moral dilemma. His crush, his benefactor, and one dying patient is stuck under a roof, waiting for help. "You" would need to decide who would be the persons in the car and leave and who would stay in the pouring rain.

Implicit condition: Everyone, including the patient, is able to drive away, at least at this moment.

## Approach

This problem is solved by a popular algorithm in Reinforcement Learning - the state-value function. This algorithm is very powerful and has been applied to robotics, self-driving cars. Here a simple version is used, where the number of states are pretty limited, given only four parties are in this state.

## Environment Dependencies

NumPy is needed for the running of this file.
Python 3.0 is required.

## Running the File

```
python rainy-night.py
```
