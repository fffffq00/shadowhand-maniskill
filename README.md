# shadowhand-maniskill
Shadowhand in ManiSkill

This repo stores URDF and SRDF configurations for Shadowhand that can be loaded into [ManiSkill](https://github.com/haosulab/ManiSkill) or [SAPIEN](https://github.com/haosulab/SAPIEN). 

I made some changes in the configuration files that come from [shadow-robot](https://github.com/shadow-robot), including the loading files for the visual bodies and the simplification of the collision bodies. 

The `shadowhand.urdf` is the original configuration and `shadowhand_simple.urdf` is the modified configuration.


# RUN Test
This repository uses Python version 3.12.2.
```
pip install maniskill==3.0.0b20  #it should install all dependencies.
```
Running `python robots/shadowhand.py` will load the robot into an empty maniskill environment, and then you can perform simple interactions with it by typing on the keyboard. You can take a look at the code for details, it is very simple. 


# File description 

## urdf_loader.py
> from [urdf_loader](https://github.com/haosulab/SAPIEN/blob/3.0.0dev/python/py_package/wrapper/urdf_loader.py) in SPAIEN, slightly modified to add robots without collision bodies.

## vis-in-sapien.py
> using the basic environment of SAPIEN, load two same robots, one of which has no collision body, which can be used to compare the difference between the collision body and the visual body.


## robots/shadowhand.py
> load shadowhand into ManiSkill.


# Note
SAPIEN does not allow too many ignored collision groups when loading the robot's srdf file([urdf_loader](https://github.com/haosulab/SAPIEN/blob/3.0.0dev/python/py_package/wrapper/urdf_loader.py#L589)), so I only added a small number of groups.

The mesh files of multiple shadowhand versions have not been removed for the time being. I will clean up the irrelevant code in the future.