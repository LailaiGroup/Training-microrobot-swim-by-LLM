# Python script for 'Training microrobots to swim by a large language model'

[This repository]([https://www.google.com](https://github.com/ZhulailaiFluidLab/Training_microswimmer_by_LLM)) includes the python script of the following paper:

Z. Xu, L. Zhu*. Training microrobots to swim by a large language model. Phys. Rev. Appl., 23, 044058, 2025. [Publisher](https://journals-aps-org.libproxy1.nus.edu.sg/prapplied/abstract/10.1103/PhysRevApplied.23.044058)

*If you find our script useful for your research, please acknowledge our work by citing it!*

@article{xu2025training,\
  title={Training microrobots to swim by a large language model},\
  author={Xu, Zhuoqun and Zhu, Lailai},\
  journal={Phys. Rev. Appl.},\
  year={2025}\
}

## Project dependencies

Ensure your Python version is at least 3.11.4. Install the required libraries for this project:

```bash
pip install openai==0.28
pip install numpy
pip install retrying
```
## How to decide which script to run

```bash
LLM_control_microswimmers.py is used to control three-link and three-sphere swimmers, and LLM_control_microswimmers_four_DOF.py is used to control four-link and four-sphere swimmers.
```


## How to run this script
```bash
python3 LLM_control_microswimmers.py / LLM_control_microswimmers_four_DOF.py
```

Before running this script, you need to enter your API KEY on line 10 / 12. You can obtain your API KEY by registering on the [Open API website](https://platform.openai.com/api-keys). Please note that you must make a deposit before you can use the API KEY.


## Select the swimmer to control

```bash
In the file LLM_control_microswimmers.py, line 803:
swimmer = ThreeLink() # for the Purcell's (three-link) swimmer
swimmer = ThreeSphere() # for Najafi-Golestanian's (three-sphere) swimmer
```

```bash
In the file LLM_control_microswimmers_four_DOF.py, line 616:
swimmer = FourLink() # for the four-link swimmer
swimmer = FourSphere() # for the four-sphere swimmer
```

## Change swimmer's moving direction (positive or negative)

```text
In the file LLM_control_microswimmers.py, when you want to change the moving direction (positive or negative) for the three-link and three-sphere swimmers:
1. Change the first sentence of the prompt (see line 766).
2. Modify the history-clear section (see lines 783-790).
（If the swimmer is moving in the positive direction, use lines 788-790; if the swimmer is moving in the negative direction, use lines 784-786.）
```

```text
In the file LLM_control_microswimmers_four_DOF.py, when you want to change the moving direction (positive or negative) for the four-link and four-sphere swimmers:
1. Change the first sentence of the prompt (see line 578).
2. Modify the history-clear section (see lines 594-601).
（If the swimmer is moving in the positive direction, use lines 598-601; if the swimmer is moving in the negative direction, use lines 594-597.）
```
## Change the number of historical demonstrations

```bash
In the file LLM_control_microswimmers.py, line 684:
self.history_length = xxx # 
```
```bash
In the file LLM_control_microswimmers_four_DOF.py, line 496:
self.history_length = xxx # 
```
（'self.history_length' in the script corresponds to the 'n_ht' variable shown in Figure 3 of the manuscript. For the four-link and four-sphere swimmers, 'self.history_length' is consistently set to 20.）



<!--There are two class "class ThreeLink" and "class Threesphere" respectively introduce the environment of purcell swimmer and NG's swimmer. -->
