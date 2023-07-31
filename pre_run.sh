#!/usr/bin/bash

if [ ! -d "output" ];
then
	mkdir output
fi


if [[ $# -eq 0 ]] ; then
	echo 'Please provide config file. Example :'
	echo './topography.py config.py'
	exit 1
fi

conf=$1

echo 'Topography'
python3 topography.py $conf &&
echo 'Initial conditions' &&
python3 initial_conditions.py $conf &&
echo 'Truth' &&
python3 truth.py $conf &&
echo 'Pseudo-observations' &&
python3 pseudo_observations.py $conf &&
echo 'Model error covariance matrix Q' &&
python3 offlineQ.py $conf
